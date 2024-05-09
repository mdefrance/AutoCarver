""" Defines a categorical feature"""

from typing import Any

from pandas import DataFrame, Series, notna, unique

from .base_feature import BaseFeature
from .grouped_list import GroupedList


class CategoricalFeature(BaseFeature):
    __name__ = "Categorical"

    def __init__(self, name: str, **kwargs: dict) -> None:
        super().__init__(name, **kwargs)
        self.is_categorical = True

        # TODO adding stats
        self.n_unique = None
        self.mode = None
        self.pct_mode = None

    def __repr__(self):
        return f"{self.__name__}('{self.name}')"

    def fit(self, X: DataFrame, y: Series = None) -> None:
        # checking that feature is not ordinal (already set values)
        if self.values is None:
            # initiating feature with its unique non-nan values
            self.values = GroupedList(nan_unique(X[self.name]))

        # adding NANS
        if any(X[self.name].isna()):
            self.values.append(self.nan)
            self.has_nan = True

        super().fit(X, y)

    def update(
        self,
        values: GroupedList,
        convert_labels: bool = False,
        sorted_values: bool = False,
        replace: bool = False,
    ) -> None:
        """updates values and labels for each value of the feature"""
        # updating feature's values
        super().update(values, convert_labels, sorted_values, replace)

        # for qualitative feature -> by default, labels are values
        super().update_labels()
        # TODO make better labels


class OrdinalFeature(CategoricalFeature):
    __name__ = "Ordinal"

    def __init__(self, name: str, values: list[str], **kwargs: dict) -> None:
        super().__init__(name, **kwargs)
        self.is_ordinal = True
        self.is_categorical = False

        # checking for values
        if len(values) == 0:
            raise ValueError(
                " - [Features] Please make sure to provide all ordered values for "
                f"{self.__name__}('{self.name}')"
            )

        # setting values and labels
        super().update(GroupedList(values))

    def fit(self, X: DataFrame, y: Series = None) -> None:
        """TODO: fit stats"""
        # checking for feature's unique non-nan values
        unique_values = nan_unique(X[self.name])

        # unexpected values for this feature
        unexpected = [val for val in unique_values if not self.values.contains(val)]
        if len(unexpected) > 0:
            raise ValueError(
                " - [Features] Please make sure to provide all ordered values for "
                f"{self.__name__}('{self.name}'). Unexpected values: {str(list(unexpected))}."
                "Make sure to use StringDiscretizer to convert float/int to str."
            )

        super().fit(X, y)


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
