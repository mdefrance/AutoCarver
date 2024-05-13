""" Defines a continuous feature"""

from numpy import isfinite, nan
from pandas import isna

from .utils.base_feature import BaseFeature
from .utils.grouped_list import GroupedList


class QuantitativeFeature(BaseFeature):
    def __init__(self, name: str, **kwargs: dict) -> None:
        super().__init__(name, **kwargs)
        self.is_quantitative = True

    def __repr__(self):
        return f"QuantitativeFeature('{self.name}')"

    def update(
        self,
        values: GroupedList,
        convert_labels: bool = False,
        sorted_values: bool = False,
        replace: bool = False,
        output_dtype: str = "str",
    ) -> None:
        """updates values and labels for each value of the feature"""
        # updating feature's values
        super().update(
            values,
            convert_labels=convert_labels,
            sorted_values=sorted_values,
            replace=replace,
            output_dtype=output_dtype,
        )

        # updating feature's labels
        self.update_labels(output_dtype=output_dtype)

    def update_labels(
        self,
        labels: GroupedList = None,
        output_dtype: str = "str",
    ) -> None:
        """updates label for each value of the feature"""

        # for quantitative features -> labels per quantile (removes nan)
        labels = GroupedList(get_labels(self.values, self.nan))

        # add NaNs if there are any
        if self.nan in self.values:
            labels.append(self.nan)

        # building label per value
        super().update_labels(labels, output_dtype=output_dtype)


class DatetimeFeature(BaseFeature):
    """TODO"""

    def __init__(self, name: str, reference_date: str, **kwargs: dict) -> None:
        super().__init__(name, **kwargs)
        self.is_quantitative = True
        self.reference_date = reference_date  # date of reference to compare with

    def __repr__(self):
        return f"DatetimeFeature('{self.name}')"

    def update(
        self,
        values: GroupedList,
        convert_labels: bool = False,
        sorted_values: bool = False,
        replace: bool = False,
        output_dtype: str = "str",
    ) -> None:
        """updates values and labels for each value of the feature"""
        # updating feature's values
        super().update(values, convert_labels, sorted_values, replace)

        # updating feature's labels
        self.update_labels(output_dtype)

    def update_labels(
        self,
        labels: GroupedList = None,
        output_dtype: str = "str",
    ) -> None:
        """updates label for each value of the feature"""

        # for quantitative features -> labels per quantile (removes nan)
        labels = GroupedList(get_labels(self.values, self.nan))

        # add NaNs if there are any
        if self.nan in self.values:
            labels.append(self.nan)

        # building label per value
        super().update_labels(labels, output_dtype=output_dtype)


def get_labels(values: GroupedList, str_nan: str) -> list[str]:
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
    quantiles = [val for val in values if val != str_nan and isfinite(val)]

    # converting quantiles in string
    labels = format_quantiles(quantiles)

    return labels


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
        # scientific formatting
        formatted_list = [f"{number:.3e}" for number in a_list]

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
