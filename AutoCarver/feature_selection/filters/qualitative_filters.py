""" Filters based on association measures between Qualitative features.
"""

from typing import Any, Callable

from pandas import DataFrame

from ..measures import cramerv_measure, make_measure, tschuprowt_measure


def cramerv_filter(
    X: DataFrame, ranks: DataFrame, thresh_corr: float = 1, **kwargs
) -> dict[str, Any]:
    """Computes maximum Cramer's V between ``X`` and ``X`` (qualitative).
    Features too correlated to a feature more associated with the target
    are excluded (according to provided ``ranks``).

    Parameters
    ----------
    X : DataFrame
        Contains columns named after ``ranks``'s index (feature names)
    ranks : DataFrame
        Ranked features as index of the association table
    thresh_corr : float, optional
        Maximum Cramér's V bewteen features, by default ``1``

    Returns
    -------
    dict[str, Any]
        Maximum Cramér's V with a better feature
    """

    # applying quantitative filter with Cramer's V correlation
    return qualitative_filter(X, ranks, cramerv_measure, thresh_corr, **kwargs)


def tschuprowt_filter(
    X: DataFrame, ranks: DataFrame, thresh_corr: float = 1, **kwargs
) -> dict[str, Any]:
    """Computes max Tschuprow's T between X and X (qualitative).
    Features too correlated to a feature more associated with the target
    are excluded (according to provided ``ranks``).

    Parameters
    ----------
    X : DataFrame
        Contains columns named after ``ranks``'s index (feature names)
    ranks : DataFrame
        Ranked features as index of the association table
    thresh_corr : float, optional
        Maximum Tschuprow's T bewteen features, by default ``1``

    Returns
    -------
    dict[str, Any]
        Maximum Tschuprow's T with a better feature
    """

    # applying quantitative filter with Tschuprow's T correlation
    return qualitative_filter(X, ranks, tschuprowt_measure, thresh_corr, **kwargs)


def qualitative_filter(
    X: DataFrame, ranks: DataFrame, corr_measure: Callable, thresh_corr: float = 1, **kwargs
) -> dict[str, Any]:
    """Computes max association between X and X (qualitative) excluding features
    that are correlated to a feature more associated with the target
    (defined by the ranks).
    """

    # accessing the prefered order
    prefered_order = ranks.index

    # initiating list of maximum association per feature
    associations = []

    # iterating over each feature by target association order
    for feature in prefered_order:
        # computing correlation with better features anf filtering out ranks
        ranks, worst_corr = qualitative_worst_corr(
            X, feature, ranks, corr_measure, thresh_corr, **kwargs
        )

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
    thresh_corr: float,
    **kwargs,
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
        _, association = make_measure(
            corr_measure,
            True,
            {f"{measure}_with": better_feature},
            X[feature],
            X[better_feature],
            **kwargs,
        )

        # updating association if it's greater than previous better features
        if association.get(measure_name) > worst_corr.get(measure_name, 0):
            # renaming association measure as filter
            association[f"{measure}_filter"] = association.pop(measure_name)

            # removing temporary measures
            association = {k: v for k, v in association.items() if "_statistic" not in k}

            # updating worst known association
            worst_corr.update(association)

        # stopping measurements if association is greater than threshold
        if association.get(f"{measure}_filter") > thresh_corr:
            ranks = ranks.drop(feature, axis=0)  # removing feature from ranks

            return ranks, None

    return ranks, worst_corr
