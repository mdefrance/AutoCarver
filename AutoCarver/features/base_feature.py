""" TODO: initiate features from dataset
TODO: add labels
TODO add casted features?
"""

from ..config import STR_DEFAULT, STR_NAN
from ..discretizers import GroupedList


class BaseFeature:

    def __init__(
        self,
        name: str,
        input_dtype: str,
        output_dtype: str,
        str_nan: str = STR_NAN,
        str_default: str = STR_DEFAULT,
    ) -> None:

        self.name = name
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype

        self.str_nan = str_nan
        self.has_nan = False
        self.str_default = str_default
        self.has_default = False

        self.is_fitted = False

        self.type = "base"
        self.order: list[str] = GroupedList()

        # self.distribution = GroupedList()

    # def fit(self, X: DataFrame, y: Series = None) -> None:  # pylint: disable=W0222

    #     values_orders = (self.values_orders,)
    #     input_dtypes = (self.input_dtypes,)
    #     str_nan = (self.str_nan,)
