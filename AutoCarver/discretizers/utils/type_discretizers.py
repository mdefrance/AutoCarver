"""Base tools to convert values into specific types.
"""

from typing import List

from pandas import DataFrame, Series

from .base_discretizers import GroupedList, GroupedListDiscretizer, nan_unique


class StringDiscretizer(GroupedListDiscretizer):
    """Converts specified columns of a DataFrame into str

    - Keeps NaN inplace
    - Converts floats of int to int
    """

    def __init__(
        self,
        features: List[str],
    ) -> None:
        """_summary_

        Parameters
        ----------
        features : List[str]
            _description_
        copy : bool, optional
            _description_, by default False
        verbose : bool, optional
            _description_, by default False
        """

        self.features = features[:]
        self.values_orders = {}

    def fit(self, X: DataFrame, y: Series = None) -> None:
        """_summary_

        Parameters
        ----------
        X : DataFrame
            _description_
        y : Series, optional
            _description_, by default None
        """
        # converting each feature's value
        for feature in self.features:
            # unique feature values
            unique_values = nan_unique(X[feature])
            values_order = GroupedList(unique_values)

            # formatting values
            for value in unique_values:
                # case 0: the value is an integer
                if isinstance(value, float) and float.is_integer(value):
                    str_value = str(int(value))  # converting value to string
                # case 1: converting to string
                else:
                    str_value = str(value)

                # updating order
                values_order.append(str_value)  # adding string value to the order
                values_order.group(value, str_value)  # grouping integer value into the string value

            # saving feature's values
            self.values_orders.update({feature: values_order})

        # discretizing features based on each feature's values_order
        super().__init__(
            features=self.features,
            values_orders=self.values_orders,
            input_dtypes="str",
            output_dtype="str",
        )
        super().fit(X, y)

        return self
