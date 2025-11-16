""" Filters based on association measures between Quantitative features.
"""

from numpy import ones, triu
from pandas import DataFrame

from ...features import BaseFeature, get_versions
from ...utils.extend_docstring import extend_docstring
from .base_filters import BaseFilter

# from statsmodels.stats.outliers_influence import variance_inflation_factor


# TODO
# def vif_filter(X: DataFrame, ranks: DataFrame, **params) -> dict[str, Any]:
#     """Computes Variance Inflation Factor (multicolinearity)

#     Parameters
#     ----------
#     thresh_vif, float: default inf
#       Maximum VIF between features
#     """

#     # accessing the prefered order
#     prefered_order = ranks.index

#     # initiating list association per feature
#     associations = []

#     # list of dropped features
#     dropped = []

#     # iterating over each column
#     for i, feature in enumerate(prefered_order):
#         # identifying remaining more associated features
#         better_features = [f for f in prefered_order[: i + 1] if f not in dropped]

#         X_vif = X[better_features]  # keeping only better features
#         X_vif = X_vif.dropna(axis=0)  # dropping NaNs for OLS

#         # computation of VIF
#         vif = nan
#         if len(better_features) > 1 and len(X_vif) > 0:
#             vif = variance_inflation_factor(X_vif.values, len(better_features) - 1)

#         # dropping the feature if it was too correlated to a better feature
#         if vif > params.get("thresh_vif", inf) and notna(vif):
#             dropped += [feature]

#         # kept feature: updating associations with this feature
#         else:
#             associations += [{"feature": feature, "vif_filter": vif}]

#     # formatting ouput to DataFrame
#     associations = DataFrame(associations).set_index("feature")

#     # applying filter on association
#     associations = ranks.join(associations, how="right")

#     return associations


class QuantitativeFilter(BaseFilter):
    """Computes max association between X and X (quantitative) excluding features
    that are correlated to a feature more associated with the target
    (defined by the ranks).
    """

    __name__ = "QuantitativeFilter"

    is_x_quantitative = True
    is_absolute = True

    @extend_docstring(BaseFilter.filter)
    def filter(self, X: DataFrame, ranks: list[BaseFeature]) -> list[BaseFeature]:
        # computing correlation between features
        X_corr = self._compute_correlation(X, ranks)

        # filtering too correlated features
        return self._filter_correlated_features(X_corr, ranks)

    def _compute_correlation(self, X: DataFrame, rank: list[BaseFeature]) -> DataFrame:
        """Computing correlation between features"""
        # absolute correlation between features
        X_corr = X[get_versions(rank)].corr(self.measure)

        # getting upper right part of the correlation matrix and removing autocorrelation
        return X_corr.where(triu(ones(X_corr.shape), k=1).astype(bool))

    def _filter_correlated_features(
        self, X_corr: DataFrame, ranks: list[BaseFeature]
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

        return filtered

    def _compute_worst_correlation(
        self, X_corr: DataFrame, feature: BaseFeature
    ) -> tuple[str, float]:
        """Computes correlation with better features (filtering out X_corr)"""

        # correlation with more associated features
        corr_with_better_features = X_corr.loc[: feature.version, feature.version].fillna(0)

        # worst/maximum absolute correlation with better features
        return corr_with_better_features.agg(
            [lambda x: x.abs().idxmax(), lambda x: max(x.min(), x.max(), key=abs)]
        )

    def _validate(
        self, feature: BaseFeature, worst_correlation: float, correlation_with: str
    ) -> bool:
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
            info={
                "correlation_with": (
                    correlation_with if correlation_with != feature.version else "itself"
                )
            },
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
