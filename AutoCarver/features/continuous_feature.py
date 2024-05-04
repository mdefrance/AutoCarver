""" Defines a continuous feature"""

from numpy import nan, isfinite
from pandas import DataFrame, Series, isna

from ..config import STR_DEFAULT, STR_NAN
from ..discretizers import GroupedList
from .base_feature import BaseFeature


class QuantitativeFeature(BaseFeature):
    def __init__(
        self,
        name: str,
        output_dtype: str,
        str_nan: str = STR_NAN,
        str_default: str = STR_DEFAULT,
        **kwargs,
    ) -> None:
        super().__init__(name, output_dtype, str_nan, str_default)

        self.dtype = "continuous"

    def __repr__(self):
        return f"QuantitativeFeature('{self.name}')"

    def fit(self, X: DataFrame, y: Series = None) -> None:
        # adding NANS
        if any(X[self.name].isna()):
            self.has_nan = True

        super().fit(X, y)

    def update(self, values: GroupedList) -> None:
        super().update(values)

        # updating labels accordingly
        self.update_labels()

    def update_labels(self, labels: list[str] = None) -> None:
        """updates label for each value of the feature"""

        # case 0: quantitative feature -> labels per quantile (removes str_nan)
        labels = get_labels(self.values, self.str_nan)

        # add NaNs if there are any
        if self.str_nan in self.values:
            labels += [self.str_nan]

        # requested float output (AutoCarver) -> converting to integers
        if self.output_dtype == "float":
            labels = [n for n, _ in enumerate(labels)]

        # building label per value
        super().update_labels(labels)


def get_labels(quantiles: list[float], str_nan: str) -> list[str]:
    """_summary_

    Parameters
    ----------
    feature : str
        _description_
    order : GroupedList
        _description_
    str_nan : str
        _description_

    Returns
    -------
    list[str]
        _description_
    """
    # filtering out nan and inf for formatting
    quantiles = [val for val in quantiles if val != str_nan and isfinite(val)]

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
