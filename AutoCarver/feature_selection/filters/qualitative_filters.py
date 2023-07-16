""" Filters based on association measures between Qualitative features.
"""

from typing import Any, Callable

from pandas import DataFrame

from ..measures import cramerv_measure, tschuprowt_measure


def cramerv_filter(X: DataFrame, ranks: DataFrame, **params) -> dict[str, Any]:
    """Computes max Cramer's V between X and X (qualitative) excluding features
    that are correlated to a feature more associated with the target
    (defined by the ``ranks``).

    Parameters
    ----------
    thresh_corr, float: default 1.
        Maximum association between features
    """

    # applying quantitative filter with Cramer's V correlation
    return qualitative_filter(X, ranks, cramerv_measure, **params)


def tschuprowt_filter(X: DataFrame, ranks: DataFrame, **params) -> dict[str, Any]:
    """Computes max Tschuprow's T between X and X (qualitative) excluding features that are
    correlated to a feature more associated with the target (defined by the ``ranks``).

    Parameters
    ----------
    thresh_corr, float: default 1.
        Maximum association between features
    """

    # applying quantitative filter with Tschuprow's T correlation
    return qualitative_filter(X, ranks, tschuprowt_measure, **params)


def qualitative_filter(
    X: DataFrame, ranks: DataFrame, corr_measure: Callable, **params
) -> dict[str, Any]:
    """Computes max association between X and X (qualitative) excluding features
    that are correlated to a feature more associated with the target
    (defined by the ranks).

    Parameters
    ----------
    thresh_corr, float: default 1.
        Maximum association between features
    """

    # accessing the prefered order
    prefered_order = ranks.index

    # initiating list of maximum association per feature
    associations = []

    # iterating over each feature by target association order
    for feature in prefered_order:
        # computing correlation with better features anf filtering out ranks
        ranks, worst_corr = qualitative_worst_corr(X, feature, ranks, corr_measure, **params)

        # updating associations
        if worst_corr:
            associations += [worst_corr]

    # formatting ouput to DataFrame
    associations = DataFrame(associations).set_index("feature")

    # applying filter on association
    associations = ranks.join(associations, how="right")

    return associations


def qualitative_worst_corr(
    X: DataFrame,
    feature: str,
    ranks: DataFrame,
    corr_measure: Callable,
    **params,
):
    """Computes maximum association between a feature and features
    more associated to the target (according to ranks)
    """

    # measure name
    measure_name = corr_measure.__name__
    measure = measure_name.replace("_measure", "")

    # initiating worst correlation
    worst_corr = {"feature": feature}

    # features more associated with target
    better_features = list(ranks.loc[:feature].index)[:-1]

    # iterating over each better feature
    for better_feature in better_features:
        # computing association with better feature
        _, association = corr_measure(
            True,
            {f"{measure}_with": better_feature},
            X[feature],
            X[better_feature],
            **params,
        )

        # updating association if it's greater than previous better features
        if association.get(measure_name) > worst_corr.get(measure_name, 0):
            # renaming association measure as filter
            association[f"{measure}_filter"] = association.pop(measure_name)

            # removing temporary measures
            association = {k: v for k, v in association.items() if "_measure" not in k}

            # updating worst known association
            worst_corr.update(association)

        # stopping measurements if association is greater than threshold
        if association.get(f"{measure}_filter") > params.get("thresh_corr", 1):
            ranks = ranks.drop(feature, axis=0)  # removing feature from ranks

            return ranks, None

    return ranks, worst_corr
