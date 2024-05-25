""" Defines a continuous feature"""

from numpy import isfinite, nan, diff, floor, log10
from pandas import isna

from .utils.base_feature import BaseFeature
from .utils.grouped_list import GroupedList


class QuantitativeFeature(BaseFeature):
    __name__ = "Quantitative"

    def __init__(self, name: str, **kwargs: dict) -> None:
        super().__init__(name, **kwargs)
        self.is_quantitative = True

    def __repr__(self):
        return f"QuantitativeFeature('{self.name}')"

    def get_labels(self) -> GroupedList:
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

        # add NaNs if there are any
        if self.nan in self.values:
            labels.append(self.nan)

        return labels


class DatetimeFeature(BaseFeature):
    """TODO"""

    def __init__(self, name: str, reference_date: str, **kwargs: dict) -> None:
        super().__init__(name, **kwargs)
        self.is_quantitative = True
        self.reference_date = reference_date  # date of reference to compare with

    def __repr__(self):
        return f"DatetimeFeature('{self.name}')"


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
        decimals_needed = min_decimals_to_differentiate(a_list)
        decimals_needed = max(decimals_needed, 1)

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


def min_decimals_to_differentiate(sorted_numbers: list[float]) -> int:
    """computes number of decimals needed for printing"""

    # checking for values
    if len(sorted_numbers) <= 1:
        return 0

    # Find the smallest difference between consecutive numbers
    smallest_diff = min(diff(sorted_numbers))

    # All numbers are identical
    if smallest_diff == 0:
        return 0

    # Number of decimal places needed
    decimal_places = -int(floor(log10(smallest_diff)))
    return decimal_places
