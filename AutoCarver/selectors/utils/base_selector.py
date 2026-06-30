"""Tools to select the best Quantitative and Qualitative features.

The selector mirrors the :class:`BaseDiscretizer` / :class:`BaseCarver` shape: a
sklearn estimator built from a :class:`Features` set, a per-type budget and a
pluggable set of ``measures`` / ``filters`` (the swappable *decision boundary*).
Inspect the per-feature measure/filter values through the
:attr:`BaseSelector.summary` property, as on :class:`BaseCarver`.

Speed comes from :meth:`BaseMeasure.compute_all`: every feature of a given type
is scored in a single batched call (see
:mod:`AutoCarver.selectors.measures._vectorized`) instead of a per-feature Python
loop. Selection is **exhaustive** — every feature is scored exactly; there is no
chunk sampling.
"""

from abc import ABC
from collections.abc import Iterable
from typing import Self, TypeVar

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from AutoCarver.features import BaseFeature, Features, get_versions
from AutoCarver.selectors.filters import BaseFilter, NonDefaultValidFilter, ValidFilter
from AutoCarver.selectors.measures import BaseMeasure, ModeMeasure, NanMeasure
from AutoCarver.selectors.utils.pretty_print import format_ranked_features


class BaseSelector(BaseEstimator, TransformerMixin, ABC):
    """Pipeline of measures/filters that pre-selects features by association with a target.

    Subclasses (:class:`ClassificationSelector`, :class:`RegressionSelector`)
    only declare the target type and the task-appropriate default measures.

    Examples
    --------
    See `Selectors examples <https://autocarver.readthedocs.io/en/latest/index.html>`_
    """

    __name__ = "BaseSelector"

    # whether the target is qualitative (classification) or quantitative
    # (regression); ``None`` on the plain base selector (no reorientation).
    _target_is_qualitative: bool | None = None

    def __init__(
        self,
        features: Features | list[BaseFeature],
        n_best_per_type: int,
        *,
        measures: list[BaseMeasure] | None = None,
        filters: list[BaseFilter] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        features : Features
            A set of :class:`Features` to select from.

        n_best_per_type : int
            Number of quantitative and/or qualitative :class:`Features` to select.

        measures : list[BaseMeasure], optional
            Association measures (the swappable decision boundary). Defaults to a
            task-appropriate set provided by the subclass. ``NanMeasure`` and
            ``ModeMeasure`` are always added if missing.

        filters : list[BaseFilter], optional
            Redundancy filters. Defaults to the task-appropriate set; the
            validity filters are always added if missing.
        """
        # features
        self.features: Features = features if isinstance(features, Features) else Features.from_list(features)

        # number of features selected per type
        self.n_best_per_type = n_best_per_type
        if not 0 < int(self.n_best_per_type) <= len(self.features):
            raise ValueError("Must set 0 < n_best_per_type <= len(features)")

        # measures and filters (with task defaults + validity/outlier defaults)
        self.measures = self._initiate_measures(measures)
        self.filters = self._initiate_filters(filters)

        # fit state
        self.is_fitted = False
        self._selected: list[BaseFeature] = []
        self.target_name = None
        self._summaries: list[pd.DataFrame] = []

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:
        """Returns the name of the selector"""
        _ = N_CHAR_MAX
        return self.__name__

    def __sklearn_is_fitted__(self) -> bool:
        """Hook used by :func:`sklearn.utils.validation.check_is_fitted`."""
        return self.is_fitted

    @property
    def summary(self) -> pd.DataFrame:
        """Per-feature association table: every scored feature with its measures,
        filter values and rank, ranked best-first (available after :meth:`fit`).

        Mirrors :attr:`BaseCarver.summary`: display it (e.g. in a notebook or by
        printing) to inspect the measure/filter values that drove the selection.
        """
        if not self._summaries:
            return pd.DataFrame()
        return pd.concat(self._summaries, ignore_index=True)

    # ------------------------------------------------------------------
    # measure / filter initiation
    # ------------------------------------------------------------------

    def _default_measures(self) -> list[BaseMeasure]:
        """Task-appropriate default association measures (subclass overrides)."""
        return []

    def _default_filters(self) -> list[BaseFilter]:
        """Task-appropriate default redundancy filters."""
        from AutoCarver.selectors.filters import SpearmanFilter, TschuprowtFilter

        return [TschuprowtFilter(), SpearmanFilter()]

    def _initiate_measures(self, requested_measures: list[BaseMeasure] | None = None) -> list[BaseMeasure]:
        """Builds the measure list: task defaults, mandatory outlier defaults, orientation."""
        measures = list(requested_measures) if requested_measures is not None else self._default_measures()

        # always include the default outlier measures (Mode then Nan, prepended)
        for default in (ModeMeasure(), NanMeasure()):
            if all(measure.__name__ != default.__name__ for measure in measures):
                measures = [default] + measures

        return self._orient_measures(measures)

    def _initiate_filters(self, requested_filters: list[BaseFilter] | None = None) -> list[BaseFilter]:
        """Builds the filter list, always including the validity filters."""
        filters = list(requested_filters) if requested_filters is not None else self._default_filters()

        # always include the validity filters (Valid then NonDefaultValid, prepended)
        for default in (ValidFilter(), NonDefaultValidFilter()):
            if all(filter_.__name__ != default.__name__ for filter_ in filters):
                filters = [default] + filters

        return filters

    def _orient_measures(self, measures: list[BaseMeasure]) -> list[BaseMeasure]:
        """Reverses reversible measures so each handles the right feature type for the target.

        e.g. in regression (quantitative target) Kruskal-Wallis is reversed so it
        scores *qualitative* features against the continuous target. No-op on the
        plain base selector (``_target_is_qualitative is None``).
        """
        if self._target_is_qualitative is None:
            return measures

        for measure in measures:
            if measure.is_default:
                continue
            if not self._y_matches(measure) and measure.is_reversible:
                measure.reverse_xy()
            if not self._y_matches(measure):
                raise ValueError(f"[{self}] measure {measure} does not match the target type")
        return measures

    def _y_matches(self, measure: BaseMeasure) -> bool:
        """Whether a measure's target-type matches this selector's target."""
        return measure.is_y_qualitative if self._target_is_qualitative else measure.is_y_quantitative

    # ------------------------------------------------------------------
    # sklearn API: fit / transform / select
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        """Scores, ranks and filters features; stores the selected ones.

        Parameters
        ----------
        X : pd.DataFrame
            Dataset to select from.
        y : pd.Series
            Target the association is evaluated against.
        """
        if isinstance(y, pd.Series):
            self.target_name = y.name

        # clearing any previously computed measures/filters and summaries
        self._summaries = []
        self._initiate_features_measures(self.features, remove_default=True)

        # splitting features by type and selecting the best of each
        typed = get_typed_features(self.features)
        best_features = self._select_quantitatives(typed["quantitatives"], X, y)
        best_features += self._select_qualitatives(typed["qualitatives"], X, y)

        self._selected = best_features
        self.is_fitted = True
        return self

    @property
    def selected_features(self) -> Features:
        """The selected :class:`Features` (available after :meth:`fit`)."""
        check_is_fitted(self)
        return Features.from_list(self._selected) if self._selected else self._selected  # type: ignore

    def transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Restricts ``X`` to the selected features' columns."""
        _ = y
        check_is_fitted(self)
        return X[get_versions(self._selected)]

    # ------------------------------------------------------------------
    # selection internals
    # ------------------------------------------------------------------

    def _select_quantitatives(
        self, quantitatives: list[BaseFeature], X: pd.DataFrame, y: pd.Series
    ) -> list[BaseFeature]:
        """Selects the best quantitative features"""
        best_quantitatives: list[BaseFeature] = []
        if len(quantitatives) > 0:
            measures = get_quantitative_metrics(self.measures)
            filters = get_quantitative_metrics(self.filters)
            best_quantitatives = self._select_features(quantitatives, X, y, measures, filters)
        return best_quantitatives

    def _select_qualitatives(self, qualitatives: list[BaseFeature], X: pd.DataFrame, y: pd.Series) -> list[BaseFeature]:
        """Selects the best qualitative features"""
        best_qualitatives: list[BaseFeature] = []
        if len(qualitatives) > 0:
            measures = get_qualitative_metrics(self.measures)
            filters = get_qualitative_metrics(self.filters)
            best_qualitatives = self._select_features(qualitatives, X, y, measures, filters)
        return best_qualitatives

    def _initiate_features_measures(self, features: Iterable[BaseFeature], remove_default: bool = True) -> None:
        """Resets per-feature measures/filters before/within selection."""
        for feature in features:
            if remove_default:
                feature.measures = {}
                feature.filters = {}
            else:
                remove_non_default_metrics_from_features(feature)

    def _select_features(
        self,
        features: list[BaseFeature],
        X: pd.DataFrame,
        y: pd.Series,
        measures: list[BaseMeasure],
        filters: list[BaseFilter],
    ) -> list[BaseFeature]:
        """Applies default gates, then exhaustively ranks/filters every feature."""

        # default (outlier/validity) measures + filters act as gates
        apply_measures(features, X, y, measures, default_measures=True)
        features = apply_filters(features, X, filters, default_filters=True)

        # non-default measures/filters do the ranking
        measures = remove_default_metrics(measures)
        filters = remove_default_metrics(filters)

        # keeping only default metrics on features before final ranking
        self._initiate_features_measures(features, remove_default=False)

        # exhaustively selecting the best features
        best_features = get_best_features(features, X, y, measures, filters, self.n_best_per_type)

        # storing the per-feature association table for the `summary` property
        formatted_measures = format_ranked_features(features)
        if not formatted_measures.empty:
            self._summaries.append(formatted_measures)

        return best_features


def get_typed_features(features: Features) -> dict[str, list[BaseFeature]]:
    """returns quantitative and qualitative features from list of features"""
    return {
        "quantitatives": [feature for feature in features if is_quantitative(feature)],
        "qualitatives": [feature for feature in features if is_qualitative(feature)],
    }


def is_quantitative(feature: BaseFeature) -> bool:
    """checks if feature is quantitative"""
    return feature.is_quantitative and not feature.is_fitted


def is_qualitative(feature: BaseFeature) -> bool:
    """checks if feature is qualitative"""
    return feature.is_qualitative or feature.is_fitted


_MetricT = TypeVar("_MetricT", BaseMeasure, BaseFilter)


def get_qualitative_metrics(metrics: list[_MetricT]) -> list[_MetricT]:
    """returns filtered list of measures/filters that apply on qualitative features"""
    return [metric for metric in metrics if metric.is_x_qualitative]


def get_quantitative_metrics(metrics: list[_MetricT]) -> list[_MetricT]:
    """returns filtered list of measures/filters that apply on quantitative features"""
    return [metric for metric in metrics if metric.is_x_quantitative]


def get_default_metrics(metrics: list[_MetricT]) -> list[_MetricT]:
    """returns filtered list of measures/filters that are default"""
    return [metric for metric in metrics if metric.is_default]


def remove_default_metrics(metrics: list[_MetricT]) -> list[_MetricT]:
    """returns filtered list of measures/filters that are non-default"""
    return [metric for metric in metrics if not metric.is_default]


def remove_non_default_metrics_from_features(feature: BaseFeature) -> None:
    """removes non-default measures/filters from a feature"""
    measures = dict(feature.measures)
    for measure_name, measure in feature.measures.items():
        if not measure.get("info", {}).get("is_default"):
            measures.pop(measure_name)

    filters = dict(feature.filters)
    for filter_name, measure in feature.filters.items():
        if not measure.get("info", {}).get("is_default"):
            filters.pop(filter_name)

    feature.measures = measures
    feature.filters = filters


def remove_duplicates(features: list[BaseFeature]) -> list[BaseFeature]:
    """removes duplicated features, keeping its first appearance"""
    return [features[i] for i in range(len(features)) if features[i] not in features[:i]]


def sort_features_per_measure(features: list[BaseFeature], measure: BaseMeasure) -> list[BaseFeature]:
    """sorts features according to specified measure"""
    ranked = False
    for feature in features:
        if make_rank_name(measure) in feature.measures:
            ranked = True

    reverse = not measure.info.get("higher_is_better")
    if ranked:
        reverse = False

    return sorted(features, key=lambda feature: get_feature_rank(feature, measure), reverse=reverse)


def get_feature_rank(feature: BaseFeature, measure: BaseMeasure) -> float:
    """gives rank of feature according to measure"""
    if make_rank_name(measure) not in feature.measures:
        return get_measure_value(feature, measure)
    return get_measure_rank(feature, measure)


def get_measure_rank(feature: BaseFeature, measure: BaseMeasure) -> int:
    """gives rank of feature according to measure"""
    return feature.measures[make_rank_name(measure)]["value"]


def get_measure_value(feature: BaseFeature, measure: BaseMeasure) -> float:
    """gives value of measure for specified feature"""
    value = feature.measures[measure.__name__]["value"]
    if measure.is_absolute:
        value = abs(value)
    if np.isnan(value):
        value = float("-inf")
    return value


def apply_measures(
    features: list[BaseFeature],
    X: pd.DataFrame,
    y: pd.Series,
    measures: list[BaseMeasure],
    default_measures: bool = False,
) -> None:
    """Measures association between every feature and ``y`` in batched calls.

    Each measure scores all ``features`` at once via
    :meth:`BaseMeasure.compute_all` (vectorized for built-ins, per-feature
    fallback for custom measures), preserving the ``feature.measures`` contract.
    """
    used_measures = remove_default_metrics(measures)
    if default_measures:
        used_measures = get_default_metrics(measures)

    for measure in used_measures:
        # type guard (raises TypeError on mismatch)
        for feature in features:
            check_measure_mismatch(feature, measure)

        # batched association for all features at once
        results = measure.compute_all(X, y, features)
        for feature in features:
            feature.measures[measure.__name__] = results[feature.version]


def apply_filters(
    features: list[BaseFeature],
    X: pd.DataFrame,
    filters: list[BaseFilter],
    default_filters: bool = False,
    n_best: int | None = None,
) -> list[BaseFeature]:
    """Filters out too correlated features (least relevant first)"""
    used_filters = remove_default_metrics(filters)
    if default_filters:
        used_filters = get_default_metrics(filters)

    # the n_best early-stop is only sound for the *last* filter in the chain:
    # nothing drops features after it, so its first n_best kept are exactly the
    # final survivors that selection keeps (earlier filters must still see all)
    last = len(used_filters) - 1

    filtered = features[:]
    for i, measure in enumerate(used_filters):
        for feature in features:
            check_measure_mismatch(feature, measure)
        filtered = measure.filter(X, filtered, n_best=n_best if i == last else None)

    return filtered


def check_measure_mismatch(feature: BaseFeature, measure: BaseMeasure | BaseFilter) -> None:
    """checks for mismatched data types between feature and measure"""
    if not (
        (is_quantitative(feature) and measure.is_x_quantitative)
        or (is_qualitative(feature) and measure.is_x_qualitative)
    ):
        raise TypeError(
            f"Type mismatch, provided feature {feature}, with {measure} that has "
            f"is_x_quantitative={measure.is_x_quantitative}"
        )


def get_best_features(
    features: list[BaseFeature],
    X: pd.DataFrame,
    y: pd.Series,
    measures: list[BaseMeasure],
    filters: list[BaseFilter],
    n_best: int,
) -> list[BaseFeature]:
    """gives best features according to provided measures"""
    if not all(measure.is_sortable for measure in measures):
        raise ValueError("All provided measures should be sortable")

    apply_measures(features, X, y, measures)

    best_features = []
    for measure in measures:
        best_features += select_with_measure(X, features, measure, filters, n_best)

    return remove_duplicates(best_features)


def select_with_measure(
    X: pd.DataFrame,
    features: list[BaseFeature],
    measure: BaseMeasure,
    filters: list[BaseFilter],
    n_best: int,
) -> list[BaseFeature]:
    """Selects the ``n_best`` features of the DataFrame, by association with the target"""
    sorted_features = sort_features_per_measure(features, measure)
    sorted_features.reverse()

    filtered_features = apply_filters(sorted_features, X, filters, n_best=n_best)

    for rank, feature in enumerate(filtered_features):
        feature.measures.update(make_rank_info(rank, measure, n_best, len(filtered_features)))

    return select_from_rank(filtered_features, measure)


def select_from_rank(features: list[BaseFeature], measure: BaseMeasure) -> list[BaseFeature]:
    """Selects the ``n_best`` features of the DataFrame, by association with the target"""
    return [feature for feature in features if feature.measures.get(make_rank_name(measure), {}).get("valid")]


def make_rank_name(measure: BaseMeasure) -> str:
    """makes a name for the rank info"""
    return f"{measure.__name__.replace('Measure', '')}Rank"


def make_rank_info(rank: int, measure: BaseMeasure, n_best: int, n_features: int) -> dict:
    """makes a dict with rank and measure info"""
    return {
        make_rank_name(measure): {
            "value": rank,
            "threshold": n_features - n_best,
            "valid": rank < n_best,
            "info": {"is_default": False, "higher_is_better": False},
        }
    }
