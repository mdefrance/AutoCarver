""" Defines a categorical feature"""

from pandas import DataFrame, Series

from ..config import STR_DEFAULT, STR_NAN
from ..discretizers import GroupedList
from ..discretizers.utils.base_discretizers import nan_unique
from .base_feature import BaseFeature


class CategoricalFeature(BaseFeature):
    def __init__(
        self,
        name: str,
        str_nan: str = STR_NAN,
        str_default: str = STR_DEFAULT,
        **kwargs,
    ) -> None:
        super().__init__(name, str_nan, str_default)

        self.dtype = "categorical"

    def __repr__(self):
        return f"CategoricalFeature('{self.name}')"

    def fit(self, X: DataFrame, y: Series = None) -> None:
        # initiating feature with its unique non-nan values
        self.values = GroupedList(nan_unique(X[self.name]))

        # adding NANS
        if any(X[self.name].isna()):
            self.values.append(self.str_nan)
            self.has_nan = True

        super().fit(X, y)

    def update(self, values: GroupedList, output_dtype: str = "str") -> None:
        """updates values and labels for each value of the feature"""
        # updating feature's values
        super().update(values)

        # for qualitative feature -> by default, labels are values
        super().update_labels(values, output_dtype=output_dtype)
