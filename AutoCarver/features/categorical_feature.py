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
        output_dtype: str,
        str_nan: str = STR_NAN,
        str_default: str = STR_DEFAULT,
    ) -> None:
        super().__init__(name, str_nan, str_default)

        self.dtype = "categorical"

    def fit(self, X: DataFrame, y: Series = None) -> None:
        # initiating feature with its unique non-nan values
        self.order = GroupedList(nan_unique(X[self.name]))

        # adding NANS
        if any(X[self.name].isna()):
            self.order.append(self.str_nan)
            self.has_nan = True

        super().fit(X, y)

    def update(self, order: GroupedList) -> None:
        super().update(order)

        # TODO: update labels
        self.labels.update({})
