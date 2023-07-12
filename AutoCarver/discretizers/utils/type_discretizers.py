"""Base tools to convert values into specific types.
"""

from numpy import float32
from pandas import DataFrame, Series

from .base_discretizers import GroupedList, GroupedListDiscretizer, nan_unique


class StringDiscretizer(GroupedListDiscretizer):
    """Converts specified columns of a DataFrame into str.
    First step of a Qualitative Discretization pipe.

    - Keeps NaN inplace
    - Converts floats of int to int
    """

    def __init__(
        self,
        features: list[str],
        *,
        values_orders: dict[str, GroupedList] = None,
    ) -> None:
        """_summary_

        Parameters
        ----------
        features : list[str]
            _description_
        values_orders : dict[str, GroupedList], optional
            _description_, by default None
        """             

        self.features = features[:]
        if values_orders is None:
            values_orders = {}
        self.values_orders = {feature: GroupedList(value) for feature, value in values_orders.items()}

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

            # updating values_orders accordingly
            # case 0: non-ordinal features, updating as is (no order provided)
            if feature not in self.values_orders:
                self.values_orders.update({feature: values_order})
            # case 1: ordinal features, adding to contained dict (order provide)
            else:
                # currently known order (only with strings)
                known_order = self.values_orders[feature]  
                known_order.update(values_order.contained)
                self.values_orders.update({feature: known_order})

        # discretizing features based on each feature's values_order
        super().__init__(
            features=self.features,
            values_orders=self.values_orders,
            input_dtypes="str",
        )
        super().fit(X, y)

        return self


class FloatDiscretizer(GroupedListDiscretizer):
    """TODO: Converts specified columns of a DataFrame into ordered floats.
    Last step of a Discretization pipe.

    - Keeps NaN inplace
    """

    def __init__(
        self,
        features: list[str],
        values_orders: dict[str, GroupedList],
    ) -> None:
        """_summary_

        Parameters
        ----------
        features : list[str]
            _description_
        values_orders : dict[str, GroupedList]
            _description_
        """        

        self.features = features[:]
        self.values_orders = {feature: GroupedList(value) for feature, value in values_orders.items()}

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
            # feature's values
            values_order = GroupedList(feature)

            for n, value in enumerate(values_order):
                # adding n in order
                # float32(n)
                # not possible
                pass


            # saving feature's values
            self.values_orders.update({feature: values_order})

        # discretizing features based on each feature's values_order
        super().__init__(
            features=self.features,
            values_orders=self.values_orders,
            input_dtypes="str",
        )
        super().fit(X, y)

        return self
