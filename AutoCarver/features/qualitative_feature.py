""" Defines a categorical feature"""

from pandas import DataFrame, Series
from typing import Any
from pandas import unique, notna

from .grouped_list import GroupedList
from .base_feature import BaseFeature


class CategoricalFeature(BaseFeature):
    def __init__(self, name: str, **kwargs: dict) -> None:
        super().__init__(name, **kwargs)

        self.dtype = "categorical"

    def __repr__(self):
        return f"CategoricalFeature('{self.name}')"

    def fit(self, X: DataFrame, y: Series = None) -> None:
        # initiating feature with its unique non-nan values
        self.values = GroupedList(nan_unique(X[self.name]))

        # adding NANS
        if any(X[self.name].isna()):
            self.values.append(self.nan)
            self.has_nan = True

        super().fit(X, y)

    def update(self, values: GroupedList) -> None:
        """updates values and labels for each value of the feature"""
        # updating feature's values
        super().update(values)

        # for qualitative feature -> by default, labels are values
        super().update_labels()


def nan_unique(x: Series) -> list[Any]:
    """Unique non-NaN values.

    Parameters
    ----------
    x : Series
        Values to be deduplicated.

    Returns
    -------
    list[Any]
        List of unique non-nan values
    """

    # unique values
    uniques = unique(x)

    # filtering out nans
    uniques = [u for u in uniques if notna(u)]

    return uniques
