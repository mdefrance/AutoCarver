""" TODO: initiate features from dataset
TODO: add labels
TODO add casted features?
"""

from pandas import DataFrame, Series

from ..config import DEFAULT, NAN
from .grouped_list import GroupedList


class BaseFeature:
    __name__ = "Feature"

    def __init__(self, name: str, **kwargs: dict) -> None:
        self.name = name

        # whether or not feature has some NaNs
        self.has_nan = None
        self.nan = kwargs.get("nan", NAN)

        # whether or not feature has some default values
        self.has_default = None
        self.default = kwargs.get("default", DEFAULT)

        # whether or not nans must be removed
        self.dropna = None

        # whether or not feature has been fitted
        self.is_fitted = False

        # feature values, type and labels
        self.dtype = None
        self.values = None  # current values
        self.labels = None  # current labels
        self.label_per_value: dict[str, str] = {}  # current label for all existing values
        self.value_per_label: dict[str, str] = {}  # a value for each current label

    def fit(self, X: DataFrame, y: Series = None) -> None:  # pylint: disable=W0222
        _, _ = X, y  # unused attributes
        self.is_fitted = True  # feature is fitted

    def __repr__(self):
        return f"{self.__name__}('{self.name}')"

    def update(self, values: GroupedList, convert_labels: bool = False) -> None:
        """updates values for each value of the feature"""
        # values are not labels
        if not convert_labels:
            # checking for GroupedList
            if not isinstance(values, GroupedList):
                raise ValueError(f" - [{self.__name__}] Wrong input, expected GroupedList object.")

            # copying values
            self.values = values

        # values are labels -> converting them back to values
        else:
            # iterating over each grouped values
            for kept_value, grouped_values in values.content.items():
                # converting labels to values
                kept_value = self.value_per_label.get(kept_value)
                grouped_values = [self.value_per_label.get(value) for value in grouped_values]

                # updating values
                self.values.group_list(grouped_values, kept_value)

    def update_labels(self, labels: GroupedList = None, output_dtype: str = "str") -> None:
        """updates label for each value of the feature"""

        # initiating labels for qualitative features
        if labels is None:
            labels = self.values

        # requested float output (AutoCarver) -> converting to integers
        if output_dtype == "float":
            labels = [n for n, _ in enumerate(labels)]

        # saving updated labels
        self.labels = labels

        # updating label_per_value nand value_per_label
        for value, label in zip(self.values, labels):
            for grouped_value in self.values.get(value):
                self.label_per_value.update({grouped_value: label})
            self.value_per_label.update({label: value})
