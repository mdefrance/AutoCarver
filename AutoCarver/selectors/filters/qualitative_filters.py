"""Filters based on association measures between Qualitative features."""

import pandas as pd

from AutoCarver.features import BaseFeature, get_versions
from AutoCarver.selectors.filters.base_filters import BaseFilter
from AutoCarver.selectors.measures import CramervMeasure, TschuprowtMeasure
from AutoCarver.selectors.measures._vectorized import factorize_column, pairwise_chi2
from AutoCarver.utils.extend_docstring import extend_docstring


class QualitativeFilter(BaseFilter):
    """Computes max association between X and X (qualitative) excluding features
    that are correlated to a feature more associated with the target
    (defined by the ranks).
    """

    __name__ = "QualitativeFilter"

    is_x_qualitative = True

    @extend_docstring(BaseFilter.filter)
    def filter(self, X: pd.DataFrame, ranks: list[BaseFeature], n_best: int | None = None) -> list[BaseFeature]:
        # factorizing each column once and reusing the codes across every pair
        # (the scalar path re-ran pd.crosstab + factorize on every pair -> O(P^2)
        # pandas calls; here it is one bincount per pair over cached int codes)
        self._codes_cache: dict[str, tuple] = {}

        # filtering ranks to avoid correlation with already removed features
        filtered_ranks = ranks[:]

        # iterating over each feature by target association order
        filtered: list[BaseFeature] = []
        for feature in ranks:
            # maximum correlation with a better feature
            correlation_with, worst_correlation = self._compute_worst_correlation(X, feature, filtered_ranks)

            # checking for too much correlation
            valid = self._validate(worst_correlation)

            # update feature accordingly (update stats)
            self._update_feature(feature, worst_correlation, valid, info={"correlation_with": correlation_with})

            # keeping feature
            if valid:
                filtered += [feature]

                # any feature kept past n_best ranks >= n_best -> never selected,
                # so once n_best are kept the remaining pairs are wasted work
                if n_best is not None and len(filtered) >= n_best:
                    break

            # removing feature from ranks
            else:
                filtered_ranks.remove(feature)

        self._codes_cache = {}
        return filtered

    def _compute_worst_correlation(
        self, X: pd.DataFrame, feature: BaseFeature, rank: list[BaseFeature]
    ) -> tuple[str, float]:
        """Computes maximum association between a feature and features
        more associated to the target (according to ranks)
        """

        # initiating worst correlation
        correlation_with, worst_correlation = "itself", 0.0

        # features more associated with target
        current_feature_index = get_versions(rank).index(feature.version)
        better_features = rank[:current_feature_index]

        # iterating over each better feature
        for better_feature in better_features:
            # computing association with the better feature
            correlation = self._pairwise_association(X, feature.version, better_feature.version)

            # updating association if it's greater than previous better features
            if correlation > worst_correlation:
                worst_correlation, correlation_with = correlation, better_feature.version

            # breaking loop if too correlated
            if not self._validate(worst_correlation):
                break

        return correlation_with, worst_correlation

    def _pairwise_association(self, X: pd.DataFrame, version_a: str, version_b: str) -> float:
        """Vectorized feature/feature association, reusing the measure's effect-size map.

        ``pairwise_chi2`` gives the raw chi² + observation count; ``measure._stat``
        turns it into Cramér's V / Tschuprow's T exactly as the scalar
        ``compute_association`` would, so results match the per-pair path.
        """
        codes_a, ka = self._codes(X, version_a)
        codes_b, kb = self._codes(X, version_b)
        chi2, n_obs = pairwise_chi2(codes_a, ka, codes_b, kb)
        return self.measure._stat(chi2, n_obs, ka, kb)  # type: ignore

    def _codes(self, X: pd.DataFrame, version: str) -> tuple:
        """Memoized integer codes + cardinality for a column (factorized once)."""
        cache = getattr(self, "_codes_cache", None)
        if cache is None:
            cache = self._codes_cache = {}
        if version not in cache:
            cache[version] = factorize_column(X[version])
        return cache[version]

    def _validate(self, worst_correlation: float) -> bool:
        """Checks if the worst correlation of a feature is above specified threshold"""
        # dropping the feature if it was too correlated to a better feature
        valid = True
        if worst_correlation > self.threshold:
            valid = False

        return valid


class CramervFilter(QualitativeFilter):
    """Computes maximum Cramer's V between qualitative features of ``X``"""

    __name__ = "CramervFilter"

    @extend_docstring(QualitativeFilter.filter)
    def __init__(self, threshold: float = 1.0) -> None:
        super().__init__(threshold)
        self.measure = CramervMeasure(threshold)


class TschuprowtFilter(QualitativeFilter):
    """Computes maximum Tschuprow's T between qualitative features of ``X``"""

    __name__ = "TschuprowtFilter"

    @extend_docstring(QualitativeFilter.filter)
    def __init__(self, threshold: float = 1.0) -> None:
        super().__init__(threshold)
        self.measure = TschuprowtMeasure(threshold)
