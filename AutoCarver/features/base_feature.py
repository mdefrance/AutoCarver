""" TODO: initiate features from dataset
TODO: add labels
TODO add casted features?
"""

from typing import Any


from pandas import DataFrame, Series

from ..config import DEFAULT, NAN
from .grouped_list import GroupedList


class BaseFeature:
    """TODO add transform that checks for new unexpected values"""

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
        self.dropna = False

        # whether or not feature has been fitted
        self.is_fitted = False

        # feature values, type and labels
        self.values = None  # current values
        self.labels = None  # current labels
        self.label_per_value: dict[str, str] = {}  # current label for all existing values
        self.value_per_label: dict[str, str] = {}  # a value for each current label

        # initating feature dtypes
        self.is_ordinal = False
        self.is_categorical = False
        self.is_qualitative = False
        self.is_quantitative = False

    def fit(self, X: DataFrame, y: Series = None) -> None:  # pylint: disable=W0222
        _, _ = X, y  # unused attributes

        # looking for NANS
        if any(X[self.name].isna()):
            self.has_nan = True

        self.is_fitted = True  # feature is fitted

    def group_list(self, to_discard: list[Any], to_keep: Any) -> None:
        """wrapper of GroupedList: groups a list of values into a kept value"""

        values = GroupedList(self.values)
        values.group_list(to_discard, to_keep)
        self.update(values, replace=True)

    def check_values(self, X: DataFrame) -> None:
        """checks for unexpected values from unique values in DataFrame"""

        if any(X[self.name].isna()) and not self.has_nan:
            raise ValueError(f" - [Features] Unexpected NaN for {self.__name__}('{self.name}').")

    def __repr__(self):
        return f"{self.__name__}('{self.name}')"

    def update(
        self,
        values: GroupedList,
        convert_labels: bool = False,
        sorted_values: bool = False,
        replace: bool = False,
    ) -> None:
        """updates values for each value of the feature"""
        # values are the same but sorted
        if sorted_values:
            self.values = self.values.sort_by(values)

        # checking for GroupedList
        elif not isinstance(values, GroupedList):
            raise ValueError(f" - [{self.__name__}] Wrong input, expected GroupedList object.")

        # replacing existing values
        elif replace:
            self.values = values

        # values are not labels
        elif not convert_labels:
            # initiating values
            if self.values is None:
                self.values = values

            # updating existing values
            else:
                # iterating over each grouped values
                for kept_value, grouped_values in values.content.items():
                    # updating values
                    self.values.group_list(grouped_values, kept_value)

        # values are labels -> converting them back to values
        else:
            # iterating over each grouped values
            for kept_label, grouped_labels in values.content.items():
                # converting labels to values
                kept_value = self.value_per_label.get(kept_label, kept_label)
                grouped_values = [
                    self.value_per_label.get(label, label) for label in grouped_labels
                ]

                # choosing which value to keep for quantitative features
                if self.is_quantitative:
                    which_to_keep = [value for value in grouped_values if value != self.nan]
                    # keeping the largest value amongst the discarded (otherwise not grouped)
                    if len(which_to_keep) > 0:
                        kept_value = max(which_to_keep)

                # choosing which value to keep for qualitative features
                else:
                    # getting group of kept_value
                    kept_value = self.values.get_group(kept_value)
                    # keeping only values not already grouped with kept_value
                    grouped_values = [
                        self.values.get_group(value)
                        for value in grouped_values
                        if self.values.get_group(value) != kept_value
                    ]
                    # deduplicating
                    grouped_values = [
                        value
                        for num, value in enumerate(grouped_values)
                        if value not in grouped_values[num + 1 :]
                    ]

                # updating values if any to group
                if len(grouped_values) > 0:
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

    def set_has_default(self, has_default: bool = True) -> None:
        """adds default to the feature"""
        # copying values
        values = GroupedList(self.values)

        # setting default
        if has_default:
            # adding to the values
            values.append(self.default)
            self.has_default = True

            # updating labels
            self.update(values, replace=True)

    def set_dropna(self, dropna: bool = True) -> None:
        """Activates or deactivates feature's dropna mode"""
        # activating dropna mode
        if dropna:
            # setting dropna
            self.dropna = dropna

            # adding nan to the values only if they were found
            if self.has_nan:
                values = GroupedList(self.values)
                values.append(self.nan)

                # updating values
                self.update(values, replace=True)

        # deactivating dropna mode
        else:
            # setting dropna
            self.dropna = dropna

            # checking for values merged with nans
            if len(self.values.get(self.nan)) > 1:
                raise RuntimeError(
                    "Can not set feature dropna=False has values were grouped with nans."
                )

            # dropping nans from values
            self.values.remove(self.nan)
