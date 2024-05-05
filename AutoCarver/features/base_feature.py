""" TODO: initiate features from dataset
TODO: add labels
TODO add casted features?
"""

from pandas import DataFrame, Series

from ..config import DEFAULT, NAN
from .grouped_list import GroupedList


class BaseFeature:
    def __init__(self, name: str, **kwargs: dict) -> None:
        self.name = name

        # whether or not feature has some NaNs
        self.has_nan = None
        self.str_nan = kwargs.get("nan", NAN)

        # whether or not feature has some default values
        self.has_default = None
        self.str_default = kwargs.get("default", DEFAULT)

        # whether or not nans must be removed
        self.dropna = None

        # whether or not feature has been fitted
        self.is_fitted = False

        # feature values, type and labels
        self.values = GroupedList()
        self.dtype = None
        self.label_per_value: dict[str, str] = {}

    def fit(self, X: DataFrame, y: Series = None) -> None:  # pylint: disable=W0222
        _, _ = X, y  # unused attributes

        self.is_fitted = True  # fitted feature

    def update(self, values: GroupedList) -> None:
        """updates values for each value of the feature"""
        self.values = values

    def update_labels(self, labels: GroupedList = None, output_dtype: str = "str") -> None:
        """updates label for each value of the feature"""

        # initiating labels for qualitative features
        if labels is None:
            labels = self.values

        # requested float output (AutoCarver) -> converting to integers
        if output_dtype == "float":
            labels = [n for n, _ in enumerate(labels)]

        # updating label per value
        for group_of_values, label in zip(self.values, labels):
            for value in self.values.get(group_of_values):
                self.label_per_value.update({value: label})
