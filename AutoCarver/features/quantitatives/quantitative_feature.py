""" Defines a continuous feature"""

from numpy import diff, floor, inf, isfinite, log10, nan  # pylint: disable=E0611
from pandas import isna

from ..utils.base_feature import BaseFeature
from ..utils.grouped_list import GroupedList


class QuantitativeFeature(BaseFeature):
    """Defines a quantitative feature"""

    __name__ = "Quantitative"
    is_quantitative = True

    @BaseFeature.has_default.setter
    def has_default(self, value: bool) -> None:
        """does nothing for quantitative features: no default value possible"""
        _ = value

    def _specific_update(self, values: GroupedList, convert_labels: bool = False) -> None:
        """update content of values specifically per feature type"""

        # no values have been set
        if not convert_labels and self.values is None:
            # checking that inf is amongst values
            if values[-1] != inf:
                raise ValueError(f"[{self}] Must provide values with values[-1] == numpy.inf")
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

                # choosing which value to keep
                which_to_keep = [value for value in grouped_values if value != self.nan]

                # keeping the largest value amongst the discarded
                if len(which_to_keep) > 0:
                    kept_value = max(which_to_keep)

                # updating values if any to group
                if len(grouped_values) > 0:
                    # if ordinal_encoding, converting values to unique values
                    if self.ordinal_encoding:
                        r_value_per_label = {
                            v: self.values[k] for k, v in self.value_per_label.items()
                        }
                        grouped_values = [r_value_per_label[value] for value in grouped_values]
                        kept_value = r_value_per_label[kept_value]
                    self.values.group(grouped_values, kept_value)

                # updating statistics
                self._update_statistics_value(kept_label, kept_value)

    def make_labels(self) -> GroupedList:
        """gives labels per quantile (values for continuous features)

        Parameters
        ----------
        values : GroupedList
            feature's values (quantiles in the case of continuous ones)
        nan : str
            default string value for nan

        Returns
        -------
        list[str]
            list of labels per quantile
        """
        # filtering out nan and inf for formatting
        quantiles = [val for val in self.values if val != self.nan and isfinite(val)]

        # converting quantiles in string
        labels = format_quantiles(quantiles)

        # converting to grouped list
        labels = GroupedList(labels)

        # add NaNs if there are any (not grouped)
        if self.nan in self.values:
            labels.append(self.nan)

        # TODO add NaNs if there are any (grouped)
        # elif self.nan in self.values.values:

        return labels

    def _make_summary(self):
        """returns summary of feature's values' content"""
        # getting feature's labels
        labels = self.make_labels()

        # iterating over each value
        summary = []
        for num, (group, values) in enumerate(self.content.items()):
            # getting group label
            group_label = self.label_per_value.get(group)

            # Quantitative features: getting labels
            values = labels[num]

            # adding group summary
            summary += [{"feature": str(self), "label": group_label, "content": values}]

        # adding statistics and history
        return self._add_statistics_to_summary(summary)


def format_quantiles(a_list: list[float]) -> list[str]:
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
    # only one non nan quantile
    if len(a_list) == 0:
        order = ["-inf < x < inf"]

    # several quantiles
    else:
        # getting minimal number of decimals to differentiate labels
        decimals_needed = min_decimals_to_differentiate(a_list, min_decimals=1)

        # scientific formatting
        formatted_list = [f"{number:.{decimals_needed}e}" for number in a_list]

        # stripping whitespaces
        formatted_list = [string.strip() for string in formatted_list]

        # low and high bounds per quantiles
        upper_bounds = formatted_list + [nan]
        lower_bounds = [nan] + formatted_list
        order: list[str] = []
        for lower, upper in zip(lower_bounds, upper_bounds):
            if isna(lower):
                order += [f"x <= {upper}"]
            elif isna(upper):
                order += [f"{lower} < x"]
            else:
                order += [f"{lower} < x <= {upper}"]

    return order


def min_decimals_to_differentiate(sorted_numbers: list[float], min_decimals: int = 0) -> int:
    """computes number of decimals needed for printing"""

    # checking for values
    if len(sorted_numbers) <= 1:
        return min_decimals

    # Find the smallest difference between consecutive numbers
    smallest_diff = min(diff(sorted_numbers))

    # All numbers are identical
    if smallest_diff == 0:
        return min_decimals

    # Number of decimal places needed
    decimal_places = -int(floor(log10(smallest_diff)))

    # minimum of 0
    return max(decimal_places, min_decimals) + 1


def get_quantitative_features(features: list[BaseFeature]) -> list[QuantitativeFeature]:
    """returns quantitative features amongst provided features"""
    return [feature for feature in features if feature.is_quantitative]
