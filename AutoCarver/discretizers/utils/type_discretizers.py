"""Base tools to convert values into specific types.
"""

from pandas import DataFrame, Series

from .base_discretizers import BaseDiscretizer, extend_docstring, nan_unique
from .multiprocessing import apply_async_function

from ...features import GroupedList, BaseFeature


class StringDiscretizer(BaseDiscretizer):
    """Converts specified columns of a DataFrame into strings.
    First step of a Qualitative discretization pipeline.

    * Keeps NaN inplace
    * Converts floats of int to int
    """

    __name__ = "StringDiscretizer"

    @extend_docstring(BaseDiscretizer.__init__)
    def __init__(
        self,
        features: list[BaseFeature],
        **kwargs: dict,
    ) -> None:
        """
        Parameters
        ----------
        features : list[str]
            List of column names of qualitative features to be converted as string
        """
        # Initiating BaseDiscretizer
        super().__init__(features=features, **kwargs)

    @extend_docstring(BaseDiscretizer.fit)
    def fit(self, X: DataFrame, y: Series = None) -> None:  # pylint: disable=W0222
        self.verbose()  # verbose if requested

        # checking for binary target and copying X
        x_copy = self._prepare_data(X, y)  # X[self.features].fillna(self.str_nan)

        # transforming all features
        all_orders = apply_async_function(fit_feature, self.features, self.n_jobs, x_copy)

        # updating features accordingly
        self.features.update(all_orders)
        self.features.fit(x_copy, y)

        # discretizing features based on each feature's values_order
        super().fit(X, y)

        return self


def fit_feature(feature: BaseFeature, df_feature: Series):
    """fits one feature"""

    # unique feature values
    unique_values = nan_unique(df_feature)
    values_order = GroupedList(unique_values)

    # formatting values
    for value in unique_values:
        # case 0: the value is an integer
        if isinstance(value, float) and float.is_integer(value):
            str_value = str(int(value))  # converting value to string
        # case 1: converting to string
        else:
            str_value = str(value)

        # checking for string values already in the order
        if str_value not in values_order:
            values_order.append(str_value)  # adding string value to the order
            values_order.group(value, str_value)  # grouping integer value into the string value

    # adding str_nan
    if any(df_feature.isna()):
        values_order.append(feature.str_nan)

    return feature.name, values_order
