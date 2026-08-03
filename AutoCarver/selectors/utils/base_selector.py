"""Tools to select the best Quantitative and Qualitative features.

The selector mirrors the :class:`BaseDiscretizer` / :class:`BaseCarver` shape: a
sklearn estimator built from a :class:`Features` set, a total ``n_best_features``
budget and a :class:`SelectionConfig` carrying per-type ``measures`` / ``filters``
(the swappable *decision boundary*).
Inspect the per-feature measure/filter values through the
:attr:`BaseSelector.summary` property, as on :class:`BaseCarver`.

Speed comes from :meth:`BaseMeasure.compute_all`: every feature of a given type
is scored in a single batched call (see
:mod:`AutoCarver.selectors.measures._vectorized`) instead of a per-feature Python
loop. Selection is **exhaustive** — every feature is scored exactly; there is no
chunk sampling.
"""

from abc import ABC
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Self, TypeVar
from warnings import warn

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from AutoCarver.features import BaseFeature, Features, get_versions
from AutoCarver.selectors.filters import BaseFilter, NonDefaultValidFilter, ValidFilter
from AutoCarver.selectors.measures import BaseMeasure, ModeMeasure, NanMeasure
from AutoCarver.selectors.utils.pretty_print import format_ranked_features


@dataclass
class SelectionConfig:
    """Behavioral configuration applied to a :class:`BaseSelector`.

    Measures and filters are declared **per feature type** so the routing is
    explicit: a qualitative-only measure can no longer silently leave the
    quantitative features unranked.

    Each list defaults to ``None`` meaning *use the selector's task-appropriate
    default for that type*. Pass an explicit list to override it. The default
    gate measures (:class:`NanMeasure`, :class:`ModeMeasure`) and validity
    filters (:class:`ValidFilter`, :class:`NonDefaultValidFilter`) are always
    added if missing — include your own instance to change their threshold.

    ``verbose`` prints the per-type selection counts at the end of
    :meth:`BaseSelector.fit`.
    """

    qualitative_measures: list[BaseMeasure] | None = None
    quantitative_measures: list[BaseMeasure] | None = None
    qualitative_filters: list[BaseFilter] | None = None
    quantitative_filters: list[BaseFilter] | None = None
    verbose: bool = False


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
        n_best_features: int | None = None,
        *,
        config: SelectionConfig | None = None,
    ) -> None:
        """
        Parameters
        ----------
        features : Features
            A set of :class:`Features` to select from.

        n_best_features : int, optional
            Total number of :class:`Features` to select, split across feature
            types proportionally to how many of each were passed (see
            :func:`split_budget`). ``None`` (the default) applies no cap: every
            feature passing the default gates is kept.

        config : SelectionConfig, optional
            Per-type measures/filters and ``verbose``. Defaults to the
            task-appropriate measures/filters provided by the subclass.
        """
        # features
        self.features: Features = features if isinstance(features, Features) else Features.from_list(features)

        # total number of features to select, split across types at fit time
        self.n_best_features = n_best_features
        if n_best_features is not None and int(n_best_features) <= 0:
            raise ValueError(f"[{self}] n_best_features must be > 0, or None for no selection")

        self.config = config if config is not None else SelectionConfig()

        # per-type measures/filters, keyed like get_typed_features
        self.measures: dict[str, list[BaseMeasure]] = {
            "qualitatives": self._resolve_measures(self.config.qualitative_measures, "qualitatives"),
            "quantitatives": self._resolve_measures(self.config.quantitative_measures, "quantitatives"),
        }
        self.filters: dict[str, list[BaseFilter]] = {
            "qualitatives": self._resolve_filters(self.config.qualitative_filters, "qualitatives"),
            "quantitatives": self._resolve_filters(self.config.quantitative_filters, "quantitatives"),
        }

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
        """Per-feature association table, ranked best-first (available after :meth:`fit`).

        One row per feature (per ranking measure when several apply to a type). The
        ranking measure and redundancy filter are *named* in the ``measure`` /
        ``filter`` columns so qualitative and quantitative features share the
        ``association`` and ``redundancy`` columns instead of producing a ragged frame.

        Mirrors :attr:`BaseCarver.summary`: display it (e.g. in a notebook or by
        printing) to inspect the measure/filter values that drove the selection.
        """
        if not self._summaries:
            return pd.DataFrame()

        frame = pd.concat(self._summaries, ignore_index=True)
        if self.is_fitted:
            selected = {feature.version for feature in self._selected}
            frame["selected"] = [feature.version in selected for feature in frame["feature"]]
        return frame

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

    def _kind_getter(self, kind: str) -> Callable[[list], list]:
        """Metric predicate for a feature-type key ('qualitatives'/'quantitatives')."""
        return get_qualitative_metrics if kind == "qualitatives" else get_quantitative_metrics

    def _other_kind(self, kind: str) -> str:
        """Name of the other per-type slot, for error messages."""
        return "quantitative" if kind == "qualitatives" else "qualitative"

    def _type_default_measures(self, kind: str) -> list[BaseMeasure]:
        """Task-default ranking measures that apply to ``kind`` (fresh instances)."""
        oriented = self._orient_measures(self._default_measures())
        return remove_default_metrics(self._kind_getter(kind)(oriented))

    def _type_default_filters(self, kind: str) -> list[BaseFilter]:
        """Task-default redundancy filters that apply to ``kind`` (fresh instances)."""
        return remove_default_metrics(self._kind_getter(kind)(self._default_filters()))

    def _resolve_measures(self, requested: list[BaseMeasure] | None, kind: str) -> list[BaseMeasure]:
        """Builds the measure list for one feature type: gates then ranking measures."""
        gates: list[BaseMeasure] = []
        if requested is None:
            ranking = self._type_default_measures(kind)
        else:
            oriented = self._orient_measures(list(requested))
            gates = get_default_metrics(oriented)
            ranking = remove_default_metrics(oriented)

            # a non-default measure placed in the wrong per-type slot is a hard error
            applicable = self._kind_getter(kind)(ranking)
            mismatched = [measure for measure in ranking if measure not in applicable]
            if mismatched:
                raise ValueError(
                    f"[{self}] {mismatched[0]} does not apply to {kind[:-1]} features; "
                    f"move it to `{self._other_kind(kind)}_measures`"
                )

            # without a ranking measure nothing would be selected: fall back to the default
            if not ranking:
                ranking = self._type_default_measures(kind)
                if ranking:
                    warn(
                        f"[{self}] no ranking measure applies to {kind[:-1]} features -- "
                        f"falling back to {ranking[0]}().",
                        UserWarning,
                        stacklevel=3,
                    )

        # the gate measures are mandatory; a user-supplied instance wins (its threshold)
        for default in (ModeMeasure(), NanMeasure()):
            if all(gate.__name__ != default.__name__ for gate in gates):
                gates = [default] + gates

        return gates + ranking

    def _resolve_filters(self, requested: list[BaseFilter] | None, kind: str) -> list[BaseFilter]:
        """Builds the filter list for one feature type: gates then redundancy filters."""
        gates: list[BaseFilter] = []
        if requested is None:
            redundancy = self._type_default_filters(kind)
        else:
            gates = get_default_metrics(requested)
            redundancy = remove_default_metrics(requested)

            applicable = self._kind_getter(kind)(redundancy)
            mismatched = [filter_ for filter_ in redundancy if filter_ not in applicable]
            if mismatched:
                raise ValueError(
                    f"[{self}] {mismatched[0]} does not apply to {kind[:-1]} features; "
                    f"move it to `{self._other_kind(kind)}_filters`"
                )
            # no fallback here: running without a redundancy filter is a valid choice

        # always include the validity filters (Valid then NonDefaultValid, prepended)
        for default in (ValidFilter(), NonDefaultValidFilter()):
            if all(gate.__name__ != default.__name__ for gate in gates):
                gates = [default] + gates

        return gates + redundancy

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

        # splitting features by type and apportioning the total budget across them
        typed = get_typed_features(self.features)
        budget = split_budget(self.n_best_features, {kind: len(feats) for kind, feats in typed.items()})

        best_features = self._select_kind("quantitatives", typed, X, y, budget)
        best_features += self._select_kind("qualitatives", typed, X, y, budget)

        self._selected = best_features
        self.is_fitted = True

        if self.config.verbose:
            self._print_report(typed)
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

    def _select_kind(
        self,
        kind: str,
        typed: dict[str, list[BaseFeature]],
        X: pd.DataFrame,
        y: pd.Series,
        budget: dict[str, int],
    ) -> list[BaseFeature]:
        """Selects the best features of one type, within its share of the budget."""
        features = typed[kind]
        if len(features) == 0:
            return []
        return self._select_features(features, X, y, self.measures[kind], self.filters[kind], budget[kind])

    def _print_report(self, typed: dict[str, list[BaseFeature]]) -> None:
        """Prints per-type selection counts (``config.verbose``)."""
        selected = {feature.version for feature in self._selected}
        for kind, features in typed.items():
            if len(features) > 0:
                kept = sum(feature.version in selected for feature in features)
                print(f"[{self}] selected {kept}/{len(features)} {kind[:-1]} feature(s)")

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
        n_best: int,
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

        # exhaustively selecting the best features; get_best_features returns the union of
        # each measure's top-n_best, so it is truncated back to this type's budget
        best_features = get_best_features(features, X, y, measures, filters, n_best)[:n_best]

        # storing the per-feature association table for the `summary` property
        formatted_measures = format_ranked_features(features, measures, filters)
        if not formatted_measures.empty:
            self._summaries.append(formatted_measures)

        return best_features


def get_typed_features(features: Features) -> dict[str, list[BaseFeature]]:
    """returns quantitative and qualitative features from list of features"""
    return {
        "quantitatives": [feature for feature in features if is_quantitative(feature)],
        "qualitatives": [feature for feature in features if is_qualitative(feature)],
    }


def split_budget(n_best: int | None, counts: dict[str, int]) -> dict[str, int]:
    """Splits a total selection budget across feature types, proportionally.

    Largest-remainder apportionment: each type gets ``n_best * count / total``
    rounded down, and the leftover seats go to the largest fractional parts.
    ``None`` (or a budget larger than the feature count) means no cap.
    """
    total = sum(counts.values())
    if n_best is None or total == 0 or n_best >= total:
        return dict(counts)

    exact = {kind: n_best * count / total for kind, count in counts.items()}
    budget = {kind: int(value) for kind, value in exact.items()}
    leftover = n_best - sum(budget.values())
    for kind in sorted(exact, key=lambda kind: exact[kind] - budget[kind], reverse=True)[:leftover]:
        budget[kind] += 1
    return budget


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
