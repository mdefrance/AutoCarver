""" Filters based on association measure between features and a binary target.
"""

from typing import Any

from pandas import DataFrame


def thresh_filter(X: DataFrame, ranks: DataFrame, **params) -> dict[str, Any]:
    """Filters out missing association measure (did not pass a threshold)"""

    # drops rows with nans
    associations = ranks.dropna(axis=0)

    return associations


def measure_filter(X: DataFrame, ranks: DataFrame, **params) -> dict[str, Any]:
    """Filters out specified measure's lower ranks than threshold

    Parameters
    ----------
    thresh_measure, float: default 0.
        Minimum association between target and features
        To be used with: `association_filter`
    name_measure, str
        Measure to be used for minimum association filtering
        To be used with: `association_filter`
    """

    associations = ranks.copy()

    # drops rows with nans
    if "name_measure" in params:
        associations = ranks[ranks[params.get("name_measure")] > params.get("thresh_measure", 0.0)]

    return associations
