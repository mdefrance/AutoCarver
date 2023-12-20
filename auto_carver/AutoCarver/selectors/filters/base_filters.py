""" Filters based on association measure between features and a binary target.
"""

from typing import Any

from pandas import DataFrame


def thresh_filter(X: DataFrame, ranks: DataFrame, **kwargs) -> dict[str, Any]:
    """Filters out missing association measure (did not pass a threshold)"""

    # drops rows with nans
    associations = ranks.dropna(axis=0)

    return associations
