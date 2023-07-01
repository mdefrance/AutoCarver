"""Tools to build simple buckets out of Qualitative features
for a binary classification model.
"""

from typing import Any, Dict, List

from numpy import argmin, inf, nan, select
from pandas import DataFrame, Series, isna, notna
from sklearn.base import BaseEstimator, TransformerMixin

from BaseDiscretizers import GroupedList, GroupedListDiscretizer, value_counts, target_rate, nunique


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
        self.known_values = list(set([v for o in self.chained_orders for v in o.values()]))
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


class DefaultDiscretizer(BaseEstimator, TransformerMixin):
    """Groups a qualitative features' values less frequent than min_freq into a default_value"""
    def __init__(
        self,
        features: List[str],
        min_freq: float,
        *,
        values_orders: Dict[str, Any] = {},
        default_value: str = "__OTHER__",
        str_nan: str = "__NAN__",
        copy: bool = False,
        verbose: bool = False,
    ) -> None:
        """_summary_

        Parameters
        ----------
        features : List[str]
            _description_
        min_freq : float
            _description_
        values_orders : Dict[str, Any], optional
            _description_, by default {}
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
        self.values_orders = {k: GroupedList(v) for k, v in values_orders.items()}
        self.default_value = default_value[:]
        self.str_nan = str_nan[:]
        self.copy = copy
        self.verbose = verbose

    def fit(self, X: DataFrame, y: Series) -> None:
        """_summary_

        Parameters
        ----------
        X : DataFrame
            _description_
        y : Series
            _description_
        """
        # filling up NaNs
        Xc = X[self.features].fillna(self.str_nan)

        # computing frequencies of each modality
        frequencies = Xc.apply(value_counts, normalize=True, axis=0)

        # computing ordered target rate per modality for ordering
        target_rates = Xc.apply(target_rate, y=y, ascending=True, axis=0)

        # attributing orders to each feature
        self.values_orders.update({f: GroupedList(list(target_rates[f])) for f in self.features})

        # number of unique modality per feature
        nuniques = Xc.apply(nunique, axis=0)

        # identifying modalities which are the most common
        self.to_keep: Dict[str, Any] = {}  # dict of features and corresponding kept modalities

        # iterating over each feature
        for feature in self.features:
            # checking for binary features
            if nuniques[feature] > 2:
                kept = [val for val, freq in frequencies[feature].items() if freq >= self.min_freq]

            # keeping all modalities of binary features
            else:
                kept = [val for val, freq in frequencies[feature].items()]

            self.to_keep.update({feature: kept})

        # grouping rare modalities
        for n, feature in enumerate(self.features):
            # printing verbose
            if self.verbose:
                print(f" - [DefaultDiscretizer] Fit {feature} ({n+1}/{len(self.features)})")

            # identifying values to discard (rare modalities)
            to_discard = [
                value for value in self.values_orders[feature] if value not in self.to_keep[feature]
            ]

            # discarding rare modalities
            if len(to_discard) > 0:
                # adding default_value to possible values
                order = self.values_orders[feature]
                order.append(self.default_value)

                # grouping rare modalities into default_value
                order.group_list(to_discard, self.default_value)

                # computing target rate for default value and reordering values according to feature's target rate
                default_target_rate = y.loc[
                    X[feature].isin(order.get(self.default_value))
                ].mean()  # computing target rate for default value
                order_target_rate = [
                    target_rates.get(feature).get(value)
                    for value in order
                    if value != self.default_value
                ]
                default_position = next(
                    n
                    for n, trate in enumerate(order_target_rate + [inf])
                    if trate > default_target_rate
                )

                # updating the modalities' order
                new_order = (
                    order[:-1][:default_position]
                    + [self.default_value]
                    + order[:-1][default_position:]
                )  # getting rid of default value already in order
                order = order.sort_by(new_order)
                self.values_orders.update({feature: order})

        return self

    def transform(self, X: DataFrame, y: Series = None) -> DataFrame:
        """_summary_

        Parameters
        ----------
        X : DataFrame
            _description_
        y : Series, optional
            _description_, by default None

        Returns
        -------
        DataFrame
            _description_
        """
        # copying dataset if requested
        Xc = X
        if self.copy:
            Xc = X.copy()

        # filling up NaNs
        Xc[self.features] = Xc[self.features].fillna(self.str_nan)

        # grouping values inside groups of modalities
        for n, feature in enumerate(self.features):
            # verbose if requested
            if self.verbose:
                print(f" - [DefaultDiscretizer] Transform {feature} ({n+1}/{len(self.features)})")

            # grouping modalities
            Xc[feature] = select(
                [~Xc[feature].isin(self.to_keep[feature])],
                [self.default_value],
                default=Xc[feature],
            )

        return Xc


