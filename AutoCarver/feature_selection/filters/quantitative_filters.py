""" Filters based on association measures between Quantitative features.
"""

from typing import Any

from numpy import inf, nan, ones, triu
from pandas import DataFrame, notna
from statsmodels.stats.outliers_influence import variance_inflation_factor


def spearman_filter(
    X: DataFrame, ranks: DataFrame, thresh_corr: float = 1, **params
) -> dict[str, Any]:
    """Computes maximum Spearman's rho between X and X (quantitative).
    Features too correlated to a feature more associated with the target
    are excluded (according to provided ``ranks``).

    Parameters
    ----------
    X : DataFrame
        Contains columns named after ``ranks``'s index (feature names)
    ranks : DataFrame
        Ranked features as index of the association table
    thresh_corr : float, optional
        Maximum Spearman's rho bewteen features, by default ``1``

    Returns
    -------
    dict[str, Any]
        Maximum Spearman's rho with a better features
    """

    # applying quantitative filter with spearman correlation
    return quantitative_filter(X, ranks, "spearman", thresh_corr, **params)


def pearson_filter(
    X: DataFrame, ranks: DataFrame, thresh_corr: float = 1, **params
) -> dict[str, Any]:
    """Computes maximum Pearson's r between X and X (quantitative).
    Features too correlated to a feature more associated with the target
    are excluded (according to provided ``ranks``).

    Parameters
    ----------
    X : DataFrame
        Contains columns named after ``ranks``'s index (feature names)
    ranks : DataFrame
        Ranked features as index of the association table
    thresh_corr : float, optional
        Maximum Pearson's r bewteen features, by default ``1``

    Returns
    -------
    dict[str, Any]
        Maximum Pearson's r with a better feature
    """

    # applying quantitative filter with spearman correlation
    return quantitative_filter(X, ranks, "pearson", thresh_corr, **params)


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


def quantitative_filter(
    X: DataFrame, ranks: DataFrame, corr_measure: str, thresh_corr: float = 1, **params
) -> dict[str, Any]:
    """Computes max association between X and X (quantitative) excluding features
    that are correlated to a feature more associated with the target
    (defined by the ranks).

    Parameters
    ----------
    thresh_corr, float: default 1.
        Maximum association between features
    """

    # accessing the prefered order
    prefered_order = ranks.index

    # computing correlation between features
    X_corr = X[prefered_order].corr(corr_measure).abs()
    X_corr = X_corr.where(triu(ones(X_corr.shape), k=1).astype(bool))

    # initiating list of maximum association per feature
    associations = []

    # iterating over each feature by target association order
    for feature in prefered_order:
        # correlation with features more associated to the target
        corr_with_better_features = X_corr.loc[:feature, feature]

        # maximum correlation with a better feature
        corr_with, worst_corr = corr_with_better_features.agg(["idxmax", "max"])

        # dropping the feature if it was too correlated to a better feature
        if worst_corr > thresh_corr:
            X_corr = X_corr.drop(feature, axis=0).drop(feature, axis=1)

        # kept feature: updating associations with this feature
        else:
            associations += [
                {
                    "feature": feature,
                    f"{corr_measure}_filter": worst_corr,
                    f"{corr_measure}_with": corr_with,
                }
            ]

    # formatting ouput to DataFrame
    associations = DataFrame(associations).set_index("feature")

    # applying filter on association
    associations = ranks.join(associations, how="right")

    return associations
