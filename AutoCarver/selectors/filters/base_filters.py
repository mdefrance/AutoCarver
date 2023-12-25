""" Filters based on association measure between features and a binary target.
"""

from typing import Any

from pandas import DataFrame


def thresh_filter(X: DataFrame, ranks: DataFrame, **kwargs) -> dict[str, Any]:
    """Filters out missing association measure (did not pass a threshold)"""
    _, _ = X, kwargs  # unused attributes

    # drops rows with nans
    return ranks.dropna(axis=0)
