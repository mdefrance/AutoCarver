"""Tools to build simple buckets out of Qualitative features
for a binary classification model.
"""

from typing import Any, Dict, List

from .base_discretizers import (
    GroupedList,
    GroupedListDiscretizer,
    nunique,
    target_rate,
    value_counts,
    nan_unique,
    check_new_values,
    check_missing_values,
)
from numpy import nan, select
from pandas import DataFrame, Series, notna, unique


class DefaultDiscretizer(GroupedListDiscretizer):
    """Groups a qualitative features' values less frequent than min_freq into a default_value7
    
    Only use for qualitative non-ordinal features
    """

    def __init__(
        self,
        features: List[str],
        min_freq: float,
        *,
        values_orders: Dict[str, GroupedList] = None,
        default_value: str = "__OTHER__",
        str_nan: str = "__NAN__",
        copy: bool = False,
        verbose: bool = False,
    ) -> None:
        """_summary_

        Parameters
        ----------
        features : List[str]
            List of column names to be discretized
        min_freq : float
            Minimum frequency per modality. Less frequent modalities are grouped in the closest value of the order.
        values_orders : Dict[str, GroupedList], optional
            Dict of column names (keys) and modalities' associated order (values), by default None
        default_value : str, optional
            _description_, by default "__OTHER__"
        str_nan : str, optional
            _description_, by default "__NAN__"
        copy : bool, optional
            _description_, by default False
        verbose : bool, optional
            _description_, by default False
        """
        self.features = features[:]
        self.min_freq = min_freq
        if values_orders is None:
            values_orders = {}
        self.values_orders = {k: GroupedList(v) for k, v in values_orders.items()}

        self.default_value = default_value[:]
        self.str_nan = str_nan[:]

        self.copy = copy
        self.verbose = verbose

        # dict of features and corresponding kept modalities
        self.to_keep: Dict[str, Any] = {}

    def prepare_data(self, X: DataFrame, y: Series) -> DataFrame:
        """Called during fit step

        Parameters
        ----------
        X : DataFrame
            _description_
        y : Series
            _description_

        Returns
        -------
        DataFrame
            _description_
        """
        # checks and initilizes values_orders
        for feature in self.features:
            # initiatind features missing from values_orders
            if feature not in self.values_orders:
                self.values_orders.update({feature: GroupedList(nan_unique(X[feature]))})

        # checking that all unique values in X are in values_orders
        check_new_values(X, self.features, self.values_orders)
        # checking that all unique values in values_orders are in X
        check_missing_values(X, self.features, self.values_orders)

        # adding NANS
        for feature in self.features:
            if any(X[feature].isna()):
                self.values_orders[feature].append(self.str_nan)
        
        # filling up NaNs
        Xc = X[self.features].fillna(self.str_nan)

        # checking for binary target
        y_values = unique(y)
        assert (0 in y_values) & (
            1 in y_values
        ), "y must be a binary Series (int or float, not object)"
        assert len(y_values) == 2, "y must be a binary Series (int or float, not object)"

        return Xc


    def fit(self, X: DataFrame, y: Series) -> None:
        """Learns modalities that are common enough (greater than min_freq)

        Parameters
        ----------
        X : DataFrame
            _description_
        y : Series
            _description_
        """
        # checking data before bucketization
        Xc = self.prepare_data(X, y)

        # computing frequencies of each modality
        frequencies = Xc.apply(value_counts, normalize=True, axis=0)
        
        for feature in self.features:
            # sorting orders based on target rates
            order = self.values_orders[feature]
                                  
            # checking for rare values
            rare_values = [val for val, freq in frequencies[feature].items() if freq < self.min_freq]
            if any(rare_values):
                 # adding default value to the order
                order.append(self.default_value)

                # grouping rare values in default value
                order.group_list(rare_values, self.default_value)
                Xc.loc[Xc[feature].isin(rare_values), feature] = self.default_value 

                # updating values_orders
                self.values_orders.update({feature: order})

        # computing target rate per modality for ordering
        target_rates = Xc.apply(target_rate, y=y, ascending=True, axis=0)
        
        for feature in self.features:
            # sorting orders based on target rates
            order = self.values_orders[feature]

            # updating values_orders
            self.values_orders.update({feature: order.sort_by(list(target_rates[feature]))})

        # discretizing features based on each feature's values_order
        super().__init__(
            self.values_orders,
            copy=self.copy,
            output='str',
            str_nan=self.str_nan,
        )
        super().fit(Xc, y)

        return self
    
class ChainedDiscretizer(GroupedListDiscretizer):
    """Chained Discretization based on a list of GroupedList."""

    def __init__(
        self,
        features: List[str],
        min_freq: float,
        chained_orders: List[GroupedList],
        *,
        remove_unknown: bool = True,
        str_nan: str = "__NAN__",
        copy: bool = False,
        verbose: bool = False,
    ) -> None:
        """Initializes a ChainedDiscretizer

        Parameters
        ----------
        features : List[str]
            Columns to be bucketized
        min_freq : float
            Minimum frequency per modality
        chained_orders : List[GroupedList]
            List of modality orders
        remove_unknown : bool, optional
            Whether or not to remove unknown values. If true, they are grouped
            into the value of`str_nan` otherwise it throws an error,
            by default True
        str_nan : str, optional
            Value used to replace nan. If set, same value should be used across
            all Discretizers, by default "__NAN__"
        copy : bool, optional
            Whether or not to copy the dataset, by default False
        verbose : bool, optional
            Whether or not to print information during fit, by default False
        """
        self.min_freq = min_freq
        self.features = features[:]
        self.chained_orders = [GroupedList(order) for order in chained_orders]
        self.copy = copy
        self.verbose = verbose

        # parameters to handle missing/unknown values
        self.remove_unknown = remove_unknown
        self.str_nan = str_nan

        # initiating features' values orders to all possible values
        self.known_values = list(set(v for o in self.chained_orders for v in o.values()))
        self.values_orders = {
            f: GroupedList(self.known_values[:] + [self.str_nan]) for f in self.features
        }

    def fit(self, X: DataFrame, y: Series = None) -> None:
        """_summary_

        Parameters
        ----------
        X : DataFrame
            _description_
        y : Series, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """
        # filling nans
        Xc = X[self.features].fillna(self.str_nan)

        # iterating over each feature
        for n, feature in enumerate(self.features):
            # verbose if requested
            if self.verbose:
                print(f" - [ChainedDiscretizer] Fit {feature} ({n+1}/{len(self.features)})")

            # computing frequencies of each modality
            frequencies = Xc[feature].value_counts(normalize=True)
            values, frequencies = frequencies.index, frequencies.values

            # checking for unknown values (values to present in an order of self.chained_orders)
            missing = [
                value for value in values if notna(value) and (value not in self.known_values)
            ]

            # converting unknown values to NaN
            if self.remove_unknown & (len(missing) > 0):
                # alerting user
                print(
                    f"Order for feature '{feature}' was not provided for values:  {missing}, these values will be converted to '{self.str_nan}' (policy remove_unknown=True)"
                )

                # adding missing valyes to the order
                order = self.values_orders.get(feature)
                order.update({self.str_nan: missing + order.get(self.str_nan)})

            # alerting user
            else:
                assert (
                    not len(missing) > 0
                ), f"Order for feature '{feature}' needs to be provided for values: {missing}, otherwise set remove_unknown=True"

            # iterating over each specified orders
            for order in self.chained_orders:
                # values that are frequent enough
                to_keep = list(values[frequencies >= self.min_freq])

                # all values from the order
                values_order = [o for v in order for o in order.get(v)]

                # values from the order to group (not frequent enough or absent)
                to_discard = [value for value in values_order if value not in to_keep]

                # values to group into discarded values
                value_to_group = [order.get_group(value) for value in to_discard]

                # values of the series to input
                df_to_input = [
                    Xc[feature] == discarded for discarded in to_discard
                ]  # identifying observation to input

                # inputing non frequent values
                Xc[feature] = select(df_to_input, value_to_group, default=Xc[feature])

                # historizing in the feature's order
                for discarded, kept in zip(to_discard, value_to_group):
                    self.values_orders.get(feature).group(discarded, kept)

                # updating frequencies of each modality for the next ordering
                frequencies = (
                    Xc[feature]
                    .value_counts(dropna=False, normalize=True)
                    .drop(nan, errors="ignore")
                )  # dropping nans to keep them anyways
                values, frequencies = frequencies.index, frequencies.values

        super().__init__(
            self.values_orders,
            str_nan=self.str_nan,
            copy=self.copy,
            output=str,
        )
        super().fit(X, y)

        return self


