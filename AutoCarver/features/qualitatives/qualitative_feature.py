""" Defines a categorical feature"""

from abc import abstractmethod

from numpy import floating, integer
from pandas import DataFrame, Series, notna, unique

from ..utils.base_feature import BaseFeature
from ..utils.grouped_list import GroupedList


class QualitativeFeature(BaseFeature):
    """Defines a qualitative feature"""

    __name__ = "Qualitative"
    is_qualitative = True

    # def _update_value_per_label(self, raw_labels: list[str]) -> None:
    #     """updates value per label and label per value"""
    #     self.value_per_label = {}
    #     for value, label, raw_label in zip(self.values, self._labels, raw_labels):
    #         # updating label_per_value
    #         for grouped_value in self.values.get(value):
    #             self.label_per_value.update({grouped_value: label})

    #         # updating value_per_label
    #         self.value_per_label.update({label: raw_label})

    def fit(self, X: DataFrame, y: Series = None) -> None:
        """TODO fit stats"""

        # checking for feature's unique non-nan values
        sorted_unique_values = nan_unique(X[self.version], sort=True)

        # checking that feature is not ordinal (already set values)
        if self.values is None:
            # initiating feature with its unique non-nan values
            self.update(GroupedList(sorted_unique_values))

        # checking that raw order has not been set (also useful when loading from json)
        if len(self.raw_order) == 0:
            # saving up number ordering for labeling
            self.raw_order = [self.values.get_group(value) for value in sorted_unique_values]

        # fitting BaseFeature
        super().fit(X, y)

        # class-specific checking for unexpected values
        self.check_values(X)

    def check_values(self, X: DataFrame) -> None:
        """checks for unexpected values from unique values in DataFrame"""

        # computing unique labels in dataframe
        unique_labels = unique(X[self.version])

        # converting to labels
        unique_values = unique_labels[:]
        if not self.ordinal_encoding:
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
                raise ValueError(f"[{self}] Unexpected values: {str(list(unexpected))}")

            # feature has default value:
            # adding unexpected value to list of known values
            for unexpected_value in unexpected:
                self.values.append(unexpected_value)

            # adding unexpected to default
            default_group = self.values.get_group(self.default)
            self.group(unexpected, default_group)

        super().check_values(X)

    def make_labels(self) -> GroupedList:
        """gives labels per values"""

        # iterating over each value and there content
        labels = []
        for group, content in self.content.items():
            # formatting label
            labels += [self._format_modalities(group, content)]

        return GroupedList(labels)

    def _make_summary(self):
        """returns summary of feature's values' content"""
        # iterating over each value
        summary = []
        for group, values in self.content.items():
            # getting group label
            group_label = self.label_per_value.get(group)

            # Qualtiative features: filtering out numbers
            values = [value for value in values if isinstance(value, str)]

            # if there is only one value converting to str
            if len(values) == 1:
                values = values[0]

            # adding group summary
            summary += [{"feature": str(self), "label": group_label, "content": values}]

        # adding statistics and history
        return self._add_statistics_to_summary(summary)

    def _specific_update(self, values: GroupedList, convert_labels: bool = False) -> None:
        """update content of values specifically per feature type"""

        # no values have been set
        if not convert_labels and self.values is None:
            self.values = values

        # values are not labels
        elif not convert_labels:
            # updating: iterating over each grouped values
            for kept_value, grouped_values in values.content.items():
                self.values.group(grouped_values, kept_value)

        # values are labels -> converting them back to values
        else:
            # iterating over each grouped values
            for kept_label, grouped_labels in values.content.items():
                # converting labels to values
                kept_value = self.value_per_label.get(kept_label)
                grouped_values = [self.value_per_label.get(label) for label in grouped_labels]

                # checking that kept values exists
                if kept_label not in self.value_per_label:
                    raise AttributeError(
                        f"{self} no {kept_label}, in value_per_label: {self.value_per_label}"
                    )

                # checking that grouped values exists
                for grouped_value, grouped_label in zip(grouped_values, grouped_labels):
                    if grouped_value is None:
                        print(
                            f"{self} no {grouped_label}, in value_per_label: {self.value_per_label}"
                        )

                # choosing which value to keep: getting group of kept_value
                kept_value = self.values.get_group(kept_value)
                # TODO force kept_value to != self.nan like in quantitative feature?

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
                    # if ordinal_encoding, converting values to unique values
                    if self.ordinal_encoding:
                        r_value_per_label = {
                            v: self.values[k] for k, v in self.value_per_label.items()
                        }
                        grouped_values = [r_value_per_label[value] for value in grouped_values]

                    # grouping values
                    self.values.group(grouped_values, kept_value)

                # updating statistics
                self._update_statistics_value(kept_label, kept_value)

    @abstractmethod
    def _specific_formatting(self, ordered_content: list[str]) -> str:
        """specific label formatting"""

    def _format_modalities(self, group: str, content: list[str]) -> list[str]:
        """Formats a list of float quantiles into a list of boundaries.

        Rounds quantiles to the closest power of 1000.

        Parameters
        ----------
        a_list : list[float]
            Sorted list of quantiles to convert into string

        Returns
        -------
        list[str]
            List of boundaries per quantile
        """

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
        # removing duplicates
        ordered_content = list(dict.fromkeys(ordered_content))

        # building label from ordered content
        if len(ordered_content) == 0:
            label = group
        elif len(ordered_content) == 1:
            label = ordered_content[0]
        else:
            label = self._specific_formatting(ordered_content)

        # adding nans
        if self.nan in content and label != self.nan:  # and self.nan not in label:
            label += f", {self.nan}"

        return label


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


def get_qualitative_features(features: list[BaseFeature]) -> list[QualitativeFeature]:
    """returns qualitative features amongst provided features"""
    return [feature for feature in features if feature.is_qualitative]
