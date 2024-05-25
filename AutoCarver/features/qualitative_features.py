""" Defines a categorical feature"""

from numpy import floating, integer
from pandas import DataFrame, Series, notna, unique

from .utils.base_feature import BaseFeature
from .utils.grouped_list import GroupedList


class CategoricalFeature(BaseFeature):
    __name__ = "Categorical"

    def __init__(self, name: str, **kwargs: dict) -> None:
        super().__init__(name, **kwargs)
        self.is_categorical = True
        self.is_qualitative = True

        # TODO adding stats
        self.n_unique = None
        self.mode = None
        self.pct_mode = None

    def __repr__(self):
        return f"{self.__name__}('{self.name}')"

    def fit(self, X: DataFrame, y: Series = None) -> None:
        """TODO fit stats"""

        # checking for feature's unique non-nan values
        sorted_unique_values = nan_unique(X[self.name], sort=True)

        # checking that feature is not ordinal (already set values)
        if self.values is None:

            # initiating feature with its unique non-nan values
            self.update(GroupedList(sorted_unique_values))

        # checking that raw order has not been set
        if len(self.raw_order) == 0:
            # saving up number ordering for labeling
            self.raw_order = [self.values.get_group(value) for value in sorted_unique_values]

        super().fit(X, y)

        # checking for unexpected values
        self.check_values(X)

    def check_values(self, X: DataFrame) -> None:
        """checks for unexpected values from unique values in DataFrame"""

        # computing unique labels in dataframe
        unique_labels = unique(X[self.name])

        # converting to labels
        unique_values = [self.value_per_label.get(label, label) for label in unique_labels]

        # unexpected values for this feature
        unexpected = [
            value
            for value in unique_values
            if not self.values.contains(value) and notna(value) and value != self.nan
        ]
        if len(unexpected) > 0:
            # feature does not have a default value
            if not self.has_default:
                raise ValueError(f" - [{self}] Unexpected values: {str(list(unexpected))}")

            # feature has default value:
            # adding unexpected value to list of known values
            for unexpected_value in unexpected:
                self.values.append(unexpected_value)

            # adding unexpected to default
            default_group = self.values.get_group(self.default)
            self.group_list(unexpected, default_group)

        super().check_values(X)

    def get_labels(self) -> GroupedList:
        """gives labels per values"""

        # iterating over each value and there contente
        labels = []
        for group, content in self.get_content().items():

            # ordering content as per original ordering (removes DEFAULT and NAN)
            ordered_content = [
                value
                for value in self.raw_order
                if value in content
                # removing nan
                and value != self.nan
                # removing floats
                and not isinstance(value, floating) and not isinstance(value, float)
                # removing ints
                and not isinstance(value, integer) and not isinstance(value, int)
            ]

            # building label from ordered content
            if len(ordered_content) == 0:
                label = group
            elif len(ordered_content) == 1:
                label = ordered_content[0]
            else:
                # list label for categorical feature
                label = ", ".join(ordered_content)
                if len(label) > self.max_n_chars:
                    label = label[: self.max_n_chars] + "..."

                # ordered label for ordinal features
                if self.is_ordinal:
                    label = f"{ordered_content[0]} to {ordered_content[-1]}"

            # adding nans
            if self.nan in content and label != self.nan:  # and self.nan not in label:
                label += f", {self.nan}"

            # saving label
            labels += [label]

        return GroupedList(labels)

    def get_summary(self):
        """returns summary of feature's values' content"""
        # iterating over each value
        summary = []
        for group, values in self.get_content().items():
            # getting group label
            group_label = self.label_per_value.get(group)

            # Qualtiative features: filtering out numbers
            values = [value for value in values if isinstance(value, str)]

            # if there is only one value converting to str
            if len(values) == 1:
                values = values[0]

            # adding group summary
            summary += [{"feature": str(self), "label": group_label, "content": values}]

        return summary


class OrdinalFeature(CategoricalFeature):
    __name__ = "Ordinal"

    def __init__(self, name: str, values: list[str], **kwargs: dict) -> None:
        super().__init__(name, **kwargs)
        self.is_ordinal = True
        self.is_categorical = False

        # checking for values
        if values is None or len(values) == 0:
            raise ValueError(f" - [{self}] Please make sure to provide values.")

        # checking for nan ordering
        if self.nan in values:
            raise ValueError(
                f" - [{self}] Ordering for '{self.nan}' can't be set by user, only fitted on data."
            )

        # checking for str values
        if not all(isinstance(value, str) for value in values):
            raise ValueError(f" - [{self}] Please make sure to provide str values.")

        # saving up raw ordering for labeling
        self.raw_order = kwargs.get("raw_order", values[:])

        # setting values and labels
        super().update(GroupedList(values))


def nan_unique(x: Series, sort: bool = False) -> list[str]:
    """Unique non-NaN values.

    Parameters
    ----------
    x : Series
        Values to be deduplicated.
    sorted : boolean, optionnal
        Whether or not to sort unique by appearance.

    Returns
    -------
    list[str]
        List of unique non-nan values
    """

    # unique values not sorted
    if sort:
        uniques = unique(x)

    # sorting unique values
    else:
        uniques = list(x.value_counts(sort=True, ascending=False).index)

    # filtering out nans
    uniques = [value for value in uniques if notna(value)]

    return uniques
