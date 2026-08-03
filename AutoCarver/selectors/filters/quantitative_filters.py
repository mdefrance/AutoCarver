"""Filters based on association measures between Quantitative features."""

import numpy as np
import pandas as pd

from AutoCarver.features import BaseFeature, get_versions
from AutoCarver.selectors.filters.base_filters import BaseFilter
from AutoCarver.utils.extend_docstring import extend_docstring


class QuantitativeFilter(BaseFilter):
    """Computes max association between X and X (quantitative) excluding features
    that are correlated to a feature more associated with the target
    (defined by the ranks).
    """

    __name__ = "QuantitativeFilter"

    is_x_quantitative = True
    is_absolute = True

    @extend_docstring(BaseFilter.filter)
    def filter(self, X: pd.DataFrame, ranks: list[BaseFeature], n_best: int | None = None) -> list[BaseFeature]:
        # computing correlation between features
        X_corr = self._compute_correlation(X, ranks)

        def on_drop(feature: BaseFeature) -> None:
            X_corr.drop(feature.version, axis=0, inplace=True)
            X_corr.drop(feature.version, axis=1, inplace=True)

        # filtering too correlated features
        return self._filter_ranked(
            ranks,
            worst_correlation_fn=lambda feature: self._compute_worst_correlation(X_corr, feature),
            on_drop=on_drop,
            n_best=n_best,
        )

    def _compute_correlation(self, X: pd.DataFrame, rank: list[BaseFeature]) -> pd.DataFrame:
        """Computing correlation between features"""
        X_features = X[get_versions(rank)]

        # Spearman = Pearson on ranks. Ranking the whole block once and running a
        # single Pearson is ~10x faster than pandas' corr("spearman"), which
        # re-ranks every pair when NaNs are present. Under NaN this is a ~1e-5
        # approximation of pairwise-complete Spearman (exact when no NaN), which
        # is well within tolerance for a redundancy threshold.
        if self.measure == "spearman":
            X_corr = X_features.rank().corr()
        else:
            X_corr = X_features.corr(self.measure)

        # getting upper right part of the correlation matrix and removing autocorrelation
        return X_corr.where(np.triu(np.ones(X_corr.shape), k=1).astype(bool))

    def _compute_worst_correlation(self, X_corr: pd.DataFrame, feature: BaseFeature) -> tuple[str, float]:
        """Computes correlation with better features (filtering out X_corr)"""

        # correlation with more associated features
        corr_with_better_features = X_corr.loc[: feature.version, feature.version].fillna(0)

        # worst/maximum absolute correlation with better features
        correlation_with, worst_correlation = corr_with_better_features.agg(
            [lambda x: x.abs().idxmax(), lambda x: max(x.min(), x.max(), key=abs)]
        )

        # no better feature correlated (or itself): normalize like QualitativeFilter's "itself"
        if correlation_with == feature.version:
            correlation_with = "itself"

        return correlation_with, worst_correlation

    def _validate(self, worst_correlation: float) -> bool:
        """Checks if the worst correlation of a feature is above specified threshold"""
        # dropping the feature if it was too correlated to a better feature
        valid = True
        if abs(worst_correlation) > self.threshold:
            valid = False

        return valid


class SpearmanFilter(QuantitativeFilter):
    """Computes maximum Spearman's rho between quantitative features of ``X``"""

    __name__ = "SpearmanFilter"

    @extend_docstring(QuantitativeFilter.__init__)
    def __init__(self, threshold: float = 1.0) -> None:
        super().__init__(threshold)
        self.measure = "spearman"


class PearsonFilter(QuantitativeFilter):
    """Computes maximum Pearson's r between quantitative features of ``X``"""

    __name__ = "PearsonFilter"

    @extend_docstring(QuantitativeFilter.__init__)
    def __init__(self, threshold: float = 1.0) -> None:
        super().__init__(threshold)
        self.measure = "pearson"
