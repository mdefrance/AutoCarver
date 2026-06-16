"""Defines a continuous feature"""

import numpy as np
import pandas as pd

from AutoCarver.features.utils.base_feature import BaseFeature
from AutoCarver.features.utils.grouped_list import GroupedList


class QuantitativeFeature(BaseFeature):
    """Defines a quantitative feature"""

    __name__ = "Quantitative"
    is_quantitative = True

    @BaseFeature.has_default.setter
    def has_default(self, value: bool) -> None:
        """No-op: quantitative features cannot have a default value."""
        _ = value

    def _check_empty_values(self, values: GroupedList) -> None:
        """checks that inf is amongst values"""
        if values[-1] != np.inf:
            raise ValueError(f"[{self}] Must provide values with values[-1] == numpy.inf")

    def _resolve_grouping(
        self, kept_label: str | float, grouped_values: list, r_value_per_label: dict
    ) -> tuple[list, str | float]:
        """selects the kept value and finalizes grouped values"""

        # choosing which value to keep
        kept_value = self.value_per_label[kept_label]

        # keeping the largest value amongst the discarded
        which_to_keep = [value for value in grouped_values if value != self.nan]
        if len(which_to_keep) > 0:
            kept_value = max(which_to_keep)

        # if ordinal_encoding, converting values to unique values
        if len(grouped_values) > 0 and self.ordinal_encoding:
            grouped_values = [r_value_per_label[value] for value in grouped_values]
            kept_value = r_value_per_label[kept_value]

        return grouped_values, kept_value

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
        quantiles = [val for val in self.values if val != self.nan and np.isfinite(val)]

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
        order = ["(-inf, inf)"]

    # several quantiles
    else:
        # getting minimal number of decimals to differentiate labels
        decimals_needed = min_decimals_to_differentiate(a_list, min_decimals=1)

        # scientific formatting
        formatted_list = [f"{number:.{decimals_needed}e}" for number in a_list]

        # stripping whitespaces
        formatted_list = [string.strip() for string in formatted_list]

        # low and high bounds per quantiles
        upper_bounds = formatted_list + [np.nan]
        lower_bounds = [np.nan] + formatted_list
        order: list[str] = []
        for lower, upper in zip(lower_bounds, upper_bounds):
            if pd.isna(lower):
                order += [f"(-inf, {upper}]"]
            elif pd.isna(upper):
                order += [f"({lower}, inf)"]
            else:
                order += [f"({lower}, {upper}]"]

    return order


def min_decimals_to_differentiate(sorted_numbers: list[float], min_decimals: int = 0) -> int:
    """computes number of decimals needed for printing"""

    # checking for values
    if len(sorted_numbers) <= 1:
        return min_decimals

    # Find the smallest difference between consecutive numbers
    smallest_diff = min(np.diff(sorted_numbers))

    # All numbers are identical
    if smallest_diff == 0:
        return min_decimals

    # Theoretical lower bound from the value gap.
    decimal_places = -int(np.floor(np.log10(smallest_diff)))
    decimals = max(decimal_places, min_decimals) + 1

    # Banker's rounding in ``f"{x:.Ne}"`` can still collapse adjacent values to
    # the same string (e.g. -118.04 and -118.05 both round to "-1.180e+02" at
    # 3 decimals). When that happens, ``GroupedList(format_quantiles(...))``
    # silently dedupes labels, leaving ``len(labels) < len(self.values)`` and
    # the trailing ``np.inf`` leader without an entry in ``label_per_value``
    # — which surfaces downstream as ``KeyError: inf`` in
    # ``transform_quantitative_feature``. Bump decimals until every formatted
    # string is distinct.
    while len({f"{n:.{decimals}e}" for n in sorted_numbers}) < len(sorted_numbers):
        decimals += 1
        if decimals > 17:  # double precision exhausted
            break

    return decimals


def get_quantitative_features(features: list[BaseFeature]) -> list[QuantitativeFeature]:
    """returns quantitative features amongst provided features"""
    return [feature for feature in features if isinstance(feature, QuantitativeFeature)]
