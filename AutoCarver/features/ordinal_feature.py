""" Defines an ordinal feature"""

from pandas import DataFrame, Series

from .grouped_list import GroupedList
from .categorical_feature import nan_unique
from .base_feature import BaseFeature


class OrdinalFeature(BaseFeature):
    def __init__(self, name: str, values: list[str], **kwargs: dict) -> None:
        super().__init__(name, **kwargs)

        self.dtype = "ordinal"

        # checking for values
        if len(values) == 0:
            raise ValueError(
                " - [Features] Please make sure to provide all ordered values for "
                f"OrdinalFeature('{self.name}')"
            )

        # setting values and labels
        super().update(GroupedList(values))
        super().update_labels(values)

        # adding stats
        self.n_unique = None
        self.mode = None
        self.pct_mode = None

    def __repr__(self):
        return f"OrdinalFeature('{self.name}')"

    def fit(self, X: DataFrame, y: Series = None) -> None:
        """TODO: fit stats"""
        # checking for feature's unique non-nan values
        unique_values = nan_unique(X[self.name])

        # unexpected values for this feature
        unexpected = [val for val in unique_values if not self.values.contains(val)]
        if len(unexpected) > 0:
            raise ValueError(
                " - [Features] Please make sure to provide all ordered values for "
                f"OrdinalFeature('{self.name}'). Unexpected values: {str(list(unexpected))}"
            )

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
