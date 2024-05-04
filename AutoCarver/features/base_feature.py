""" TODO: initiate features from dataset
TODO: add labels
TODO add casted features?
"""

from pandas import DataFrame, Series

from ..config import STR_DEFAULT, STR_NAN
from ..discretizers import GroupedList


class BaseFeature:
    def __init__(
        self,
        name: str,
        output_dtype: str,
        str_nan: str = STR_NAN,
        str_default: str = STR_DEFAULT,
    ) -> None:
        self.name = name
        self.output_dtype = output_dtype

        self.str_nan = str_nan
        self.has_nan = False
        self.str_default = str_default
        self.has_default = False

        self.is_fitted = False

        self.dtype = None
        self.order: list[str] = GroupedList()

        # self.distribution = GroupedList()

    def fit(self, X: DataFrame, y: Series = None) -> None:  # pylint: disable=W0222
        _, _ = X, y  # unused attributes

        self.is_fitted = True  # fitted feature
