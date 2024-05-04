""" Defines an ordinal feature"""

from pandas import DataFrame, Series

from ..config import STR_DEFAULT, STR_NAN
from ..discretizers import GroupedList
from ..discretizers.utils.base_discretizers import nan_unique
from .base_feature import BaseFeature


class OrdinalFeature(BaseFeature):
    def __init__(
        self,
        name: str,
        order: list[str],
        output_dtype: str,
        str_nan: str = STR_NAN,
        str_default: str = STR_DEFAULT,
    ) -> None:
        super().__init__(name, output_dtype, str_nan, str_default)

        self.dtype = "ordinal"
        self.order = GroupedList(order)

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
        super().update(values)

        # updating labels accordingly
        self.update_labels()

    def update_labels(self, labels: list[str] = None) -> None:
        """updates label for each value of the feature"""
        # for qualitative feature -> by default, labels are values
        labels = self.values[:]

        # requested float output (AutoCarver) -> converting to integers
        if self.output_dtype == "float":
            labels = [n for n, _ in enumerate(labels)]

        # building label per value
        super().update_labels(labels)
