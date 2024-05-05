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
        str_nan: str = STR_NAN,
        str_default: str = STR_DEFAULT,
    ) -> None:
        self.name = name

        self.str_nan = str_nan
        self.has_nan = False
        self.str_default = str_default
        self.has_default = False

        self.is_fitted = False

        self.dtype = None
        self.values = GroupedList()
        self.label_per_value: dict[str, str] = {}

    def fit(self, X: DataFrame, y: Series = None) -> None:  # pylint: disable=W0222
        _, _ = X, y  # unused attributes

        self.is_fitted = True  # fitted feature

    def update(self, values: GroupedList) -> None:
        self.values = values

    def update_labels(self, labels: GroupedList, output_dtype: str = "str") -> None:
        """updates label for each value of the feature TODO: take output_dtype as input"""

        # requested float output (AutoCarver) -> converting to integers
        if output_dtype == "float":
            labels = [n for n, _ in enumerate(labels)]

        # updating label per value
        for group_of_values, label in zip(self.values, labels):
            for value in self.values.get(group_of_values):
                self.label_per_value.update({value: label})
