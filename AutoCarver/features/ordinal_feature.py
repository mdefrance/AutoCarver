""" Defines an ordinal feature"""

from pandas import DataFrame, Series

from .grouped_list import GroupedList
from .categorical_feature import nan_unique
from .base_feature import BaseFeature


class OrdinalFeature(BaseFeature):
    def __init__(self, name: str, values: list[str], **kwargs: dict) -> None:
        super().__init__(name, **kwargs)

        self.dtype = "ordinal"

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
        assert len(unexpected) == 0, (
            " - [OrdinalFeature] Unexpected value for feature '{self.name}'! Values: "
            f"{str(list(unexpected))}. Make sure to set order accordingly when defining feature."
        )

        # adding NANS
        if any(X[self.name].isna()):
            self.values.append(self.str_nan)
            self.has_nan = True

        super().fit(X, y)

    def update(self, values: GroupedList) -> None:
        """updates values and labels for each value of the feature"""
        # updating feature's values
        super().update(values)

        # for qualitative feature -> by default, labels are values
        super().update_labels()
