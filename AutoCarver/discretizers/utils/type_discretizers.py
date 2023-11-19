"""Base tools to convert values into specific types.
"""

from pandas import DataFrame, Series

from .base_discretizers import BaseDiscretizer, extend_docstring, nan_unique
from .grouped_list import GroupedList


class StringDiscretizer(BaseDiscretizer):
    """Converts specified columns of a DataFrame into strings.
    First step of a Qualitative discretization pipeline.

    * Keeps NaN inplace
    * Converts floats of int to int
    """

    @extend_docstring(BaseDiscretizer.__init__)
    def __init__(
        self,
        qualitative_features: list[str],
        *,
        values_orders: dict[str, GroupedList] = None,
        copy: bool = False,
        verbose: bool = False,
        **kwargs: dict,
    ) -> None:
        """
        Parameters
        ----------
        qualitative_features : list[str]
            List of column names of qualitative features (non-ordinal) to be discretized
        """
        # Initiating BaseDiscretizer
        super().__init__(
            features=qualitative_features,
            values_orders=values_orders,
            input_dtypes="str",
            output_dtype="str",
            str_nan=kwargs.get("str_nan", "__NAN__"),
            copy=copy,
            verbose=verbose,
        )

    @extend_docstring(BaseDiscretizer.fit)
    def fit(self, X: DataFrame, y: Series = None) -> None:
        if self.verbose:  # verbose if requested
            print(f" - [StringDiscretizer] Fit {str(self.features)}")

        # checking for binary target and copying X
        x_copy = self._prepare_data(X, y)  # X[self.features].fillna(self.str_nan)

        # converting each feature's value
        for feature in self.features:
            # unique feature values
            unique_values = nan_unique(x_copy[feature])
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
                    values_order.group(
                        value, str_value
                    )  # grouping integer value into the string value

            # adding str_nan
            if any(x_copy[feature].isna()):
                values_order.append(self.str_nan)

            # updating values_orders accordingly
            # case 0: non-ordinal features, updating as is (no order provided)
            if feature not in self.values_orders:
                self.values_orders.update({feature: values_order})
            # case 1: ordinal features, adding to content dict (order provided)
            else:
                # currently known order (only with strings)
                known_order = self.values_orders[feature]
                known_order.update(values_order.content)
                self.values_orders.update({feature: known_order})

        # discretizing features based on each feature's values_order
        super().fit(X, y)

        return self
