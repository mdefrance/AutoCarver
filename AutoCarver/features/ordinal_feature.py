""" Defines an ordinal feature"""

from pandas import DataFrame, Series

from .base_feature import BaseFeature
from ..config import STR_DEFAULT, STR_NAN
from ..discretizers import GroupedList


class OrdinalFeature(BaseFeature):
    def __init__(
        self,
        name: str,
        order: list[str],
        input_dtype: str,
        output_dtype: str,
        str_nan: str = STR_NAN,
        str_default: str = STR_DEFAULT,
    ) -> None:
        super().__init__(name, input_dtype, output_dtype, str_nan, str_default)

        self.type = "ordinal"
        self.order = GroupedList(order)

    def fit(self, X: DataFrame, y: Series = None) -> None:
        _ = y  # unused attributes

        # adding NANS
        if any(X[self.name].isna()):
            if not self.order.contains(self.str_nan):
                self.order.append(self.str_nan)
