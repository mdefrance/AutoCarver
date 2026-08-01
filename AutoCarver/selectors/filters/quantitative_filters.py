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

        # filtering too correlated features
        return self._filter_correlated_features(X_corr, ranks, n_best)

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

    def _filter_correlated_features(
        self, X_corr: pd.DataFrame, ranks: list[BaseFeature], n_best: int | None = None
    ) -> list[BaseFeature]:
        """filtering out features too correlated with a better ranked feature"""

        # iterating over each feature by target association order
        filtered: list[BaseFeature] = []
        for feature in ranks:
            # maximum correlation with a better feature
            correlation_with, worst_correlation = self._compute_worst_correlation(X_corr, feature)

            # checking for too much correlation
            valid = self._validate(feature, worst_correlation, correlation_with)

            # dropping feature if it was too correlated
            if not valid:
                X_corr.drop(feature.version, axis=0, inplace=True)
                X_corr.drop(feature.version, axis=1, inplace=True)

            # keeping feature
            else:
                filtered += [feature]

                # once n_best are kept the rest rank past the cutoff -> never
                # selected (mirrors QualitativeFilter so both types stop alike)
                if n_best is not None and len(filtered) >= n_best:
                    break

        return filtered

    def _compute_worst_correlation(self, X_corr: pd.DataFrame, feature: BaseFeature) -> tuple[str, float]:
        """Computes correlation with better features (filtering out X_corr)"""

        # correlation with more associated features
        corr_with_better_features = X_corr.loc[: feature.version, feature.version].fillna(0)

        # worst/maximum absolute correlation with better features
        return corr_with_better_features.agg([lambda x: x.abs().idxmax(), lambda x: max(x.min(), x.max(), key=abs)])

    def _validate(self, feature: BaseFeature, worst_correlation: float, correlation_with: str) -> bool:
        """Checks if the worst correlation of a feature is above specified threshold"""
        # dropping the feature if it was too correlated to a better feature
        valid = True
        if abs(worst_correlation) > self.threshold:
            valid = False

        # update feature accordingly (update stats)
        self._update_feature(
            feature,
            worst_correlation,
            valid,
            info={"correlation_with": (correlation_with if correlation_with != feature.version else "itself")},
        )

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
