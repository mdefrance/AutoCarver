"""Tools to build simple buckets out of Qualitative features
for a binary classification model.
"""

from typing import Any, Dict, List

from .base_discretizers import (
    GroupedList,
    GroupedListDiscretizer,
    target_rate,
    value_counts,
    nan_unique,
    check_new_values,
    check_missing_values,
    get_quantiles_aliases,
    get_aliases_quantiles,
)
from numpy import nan, select, argmin
from pandas import DataFrame, Series, notna, unique, isna


class DefaultDiscretizer(GroupedListDiscretizer):
    """Groups a qualitative features' values less frequent than min_freq into a str_default string
    
    Only use for qualitative non-ordinal features
    """

    def __init__(
        self,
        features: List[str],
        min_freq: float,
        *,
        values_orders: Dict[str, GroupedList] = None,
        str_default: str = "__OTHER__",
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
        str_default : str, optional
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

        self.str_default = str_default[:]
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
        
        for n, feature in enumerate(self.features):
            if self.verbose:  # verbose if requested
                print(f" - [DefaultDiscretizer] Fit {feature} ({n+1}/{len(self.features)})")
            # sorting orders based on target rates
            order = self.values_orders[feature]
                                  
            # checking for rare values
            rare_values = [val for val, freq in frequencies[feature].items() if freq < self.min_freq]
            if any(rare_values):
                 # adding default value to the order
                order.append(self.str_default)

                # grouping rare values in default value
                order.group_list(rare_values, self.str_default)
                Xc.loc[Xc[feature].isin(rare_values), feature] = self.str_default 

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
            self.features,
            self.values_orders,
            copy=self.copy,
            input_dtype='str',
            output_dtype='str',
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
            self.features,
            self.values_orders,
            str_nan=self.str_nan,
            copy=self.copy,
            input_dtype='str',
            output_dtype='str',
        )
        super().fit(X, y)

        return self


class OrdinalDiscretizer(GroupedListDiscretizer):
    """Discretizes ordered qualitative features into groups more frequent than min_freq.
    NaNs are left untouched.

    Modality is choosen amongst the preceding and following values in the provided order.
    The choosen modality is:
    - the closest in target rate
    - or the one with the lowest frequency
    """

    def __init__(
        self,
        features: List[str],
        values_orders: Dict[str, Any],
        min_freq: float,
        *,
        str_nan: str = '__NAN__',
        input_dtype: str = 'str',
        copy: bool = False,
        verbose: bool = False,
    ):
        """Initializes a OrdinalDiscretizer

        Parameters
        ----------
        features : List[str]
            List of column names to be discretized
        values_orders : Dict[str, Any]
            Dict of column names (keys) and modalities' associated order (values)
        min_freq : float
            Minimum frequency per modality. Less frequent modalities are grouped in the closest value of the order.
        str_nan : str, optional
            _description_, by default None
        input_dtype : type, optional
            Type of the input columns:
            - if 'float' uses transform_quantitative
            - if 'str' uses transform_qualitative,
            by default 'str'
        copy : bool, optional
            _description_, by default False
        verbose : bool, optional
            _description_, by default False
        """

        self.features = features[:]
        self.min_freq = min_freq
        self.values_orders = {k: GroupedList(v) for k, v in values_orders.items()}
        self.copy = copy
        self.verbose = verbose
        self.str_nan = str_nan
        self.input_dtype = input_dtype

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
        # adding NANS
        for feature in self.features:
            if any(X[feature].isna()):
                self.values_orders[feature].append(self.str_nan)

        # removing NaNs if any already imputed
        x_copy = X
        if self.str_nan:
            x_copy = x_copy.replace(self.str_nan, nan)

        # checking for binary target
        y_values = unique(y)
        assert (0 in y_values) & (
            1 in y_values
        ), "y must be a binary Series (int or float, not object)"
        assert len(y_values) == 2, "y must be a binary Series (int or float, not object)"

        return x_copy

    def fit(self, X: DataFrame, y: Series) -> None:
        """Learns common modalities on the training set

        Parameters
        ----------
        X : DataFrame
            Training set that contains columns named after `values_orders` keys.
        y : Series
            Model target.
        """
        # verbose
        if self.verbose:
            print(f" - [OrdinalDiscretizer] Fit {', '.join(self.features)}")

        # checking values orders
        x_copy = self.prepare_data(X, y)

        # copying values_orders without nans
        known_orders = {
            feature: GroupedList([value for value in self.values_orders[feature] if value != self.str_nan])
            for feature in self.features
        }

        # for quantitative features getting aliases per quantile
        if self.input_dtype == 'float':
            # getting group "name" per quantile
            quantiles_aliases = get_quantiles_aliases(self.features, self.values_orders, self.str_nan)
            # getting group "name" per quantile
            aliases_quantiles = get_aliases_quantiles(self.features, self.values_orders, self.str_nan)
            # applying alliases to known orders
            known_orders.update({
                feature: GroupedList([quantiles_aliases[feature][quantile] for quantile in known_orders[feature]])
                for feature in self.features
            })

        # grouping rare modalities for each feature
        common_modalities = (
            x_copy[self.features]
            .apply(
                find_common_modalities,
                y=y,
                min_freq=self.min_freq,
                values_orders=known_orders,
                axis=0,
                result_type="reduce",
            )
        )

        # updating feature orders (that keeps NaNs and quantiles)
        for feature in self.features:
            # initial complete ordering with NAN and quantiles
            order = self.values_orders[feature]

            # checking for grouped modalities
            grouped_modalities = common_modalities[feature].contained

            # grouping the raw quantile values
            for kept, discarded in grouped_modalities.items():

                # for qualitative features grouping as is
                # for quantitative features getting quantile per alias
                if self.input_dtype == 'float':
                    # getting raw quantiles to be grouped
                    discarded = [aliases_quantiles[feature][discard] for discard in discarded]
                    # keeping the largest value amongst the discarded (otherwise they wont be grouped)
                    kept = max(discarded)

                # grouping quantiles
                order.group_list(discarded, kept)
            
            # updating ordering
            self.values_orders.update({feature: order})

        # discretizing features based on each feature's values_order
        super().__init__(
            self.features,
            self.values_orders,
            copy=self.copy,
            str_nan=self.str_nan,
            input_dtype=self.input_dtype,
            output_dtype='str',
        )
        super().fit(x_copy, y)

        return self


def find_common_modalities(
    df_feature: Series,
    y: Series,
    min_freq: float,
    values_orders: Dict[str, GroupedList],
    len_df: int = None,
) -> Dict[str, Any]:
    """Découpage en modalités de fréquence minimal (Cas des variables ordonnées).

    Fonction récursive : on prend l'échantillon et on cherche des valeurs sur-représentées
    Si la valeur existe on la met dans une classe et on cherche à gauche et à droite de celle-ci, d'autres variables sur-représentées
    Une fois il n'y a plus de valeurs qui soient sur-représentées,
    on fait un découpage classique avec qcut en multipliant le nombre de classes souhaité par le pourcentage de valeurs restantes sur le total

    Parameters
    ----------
    df_feature : Series
        _description_
    y : Series
        _description_
    min_freq : float
        _description_
    order : GroupedList
        _description_
    len_df : int, optional
        _description_, by default None

    Returns
    -------
    Dict[str, Any]
        _description_
    """
    # getting feature's order
    order = values_orders[df_feature.name]

    # initialisation de la taille totale du dataframe
    if len_df is None:
        len_df = len(df_feature)

    # case 1: there are missing values
    if any(isna(df_feature)):
        return find_common_modalities(
            df_feature[notna(df_feature)],
            y[notna(df_feature)],
            min_freq,
            values_orders,
            len_df,
        )

    # case 2: no missing values
    # computing frequencies and target rate of each modality
    init_frequencies = df_feature.value_counts(dropna=False, normalize=False) / len_df
    init_target_rates = y.groupby(df_feature).sum().reindex(order)

    # per-modality/value frequencies, filling missing by 0
    frequencies = init_frequencies.reindex(order, fill_value=0)

    # case 2.1: there are underrepresented modalities/values
    while any(frequencies < min_freq) & (len(frequencies) > 1):

        # updating per-group target rate per modality/value
        target_rates = series_groupy_order(init_target_rates, order) / frequencies / len_df

        # identifying the underrepresented value
        discarded_idx = argmin(frequencies)
        discarded_value = order[discarded_idx]

        # choosing amongst previous and next modality (by volume and target rate)
        kept_value = find_closest_modality(
            discarded_idx,
            order,
            list(frequencies),
            list(target_rates),
            min_freq,
        )

        # removing the value from the initial ordering
        order.group(discarded_value, kept_value)

        # removing discarded_value from frequencies
        frequencies = series_groupy_order(init_frequencies, order).fillna(0)

    # case 2.2 : no underrepresented value
    return order

def series_groupy_order(series: Series, order: GroupedList) -> Series:
    """Groups a series according to groups specified in the order

    Parameters
    ----------
    series : Series
        Values to group
    order : GroupedList
        Groups of values

    Returns
    -------
    Series
        Grouped Series
    """
    grouped_series = series.groupby(
            list(map(order.get_group, series.index)), dropna=False
    ).sum().reindex(order)

    return grouped_series


def find_closest_modality(idx: int, order: GroupedList, frequencies: List[float], target_rates: Series, min_freq: float) -> Any:
    """HELPER Finds the closest modality in terms of frequency and target rate

    Parameters
    ----------
    idx : int
        _description_
    freq_target : Series
        _description_
    min_freq : float
        _description_

    Returns
    -------
    int
        _description_
    """
    # case 1: lowest ranked modality
    if idx == 0:
        idx_closest_modality = 1

    # case 2: highest ranked modality
    elif idx == len(order) - 1:
        idx_closest_modality = len(order) - 2

    # case 3: modality ranked in the middle
    else:
        # previous modality's volume and target rate
        previous_freq, previous_target = frequencies[idx - 1], target_rates[idx - 1]

        # current modality's volume and target rate
        current_target = target_rates[idx]

        # next modality's volume and target rate
        next_freq, next_target = frequencies[idx + 1], target_rates[idx + 1]

        # identifying closest modality in terms of frequency
        least_frequent = idx - 1
        if next_freq < previous_freq:
            least_frequent = idx + 1

        # identifying closest modality in terms of target rate
        closest_target = idx - 1
        if abs(previous_target - current_target) >= abs(next_target - current_target):
            closest_target = idx + 1
        
        # case 3.1: grouping with the closest target rate
        idx_closest_modality = closest_target

        # case 3.2: (only) one modality isn't common, the least frequent is choosen
        if (previous_freq < min_freq < next_freq) or (next_freq < min_freq < previous_freq):
            idx_closest_modality = least_frequent

        # case 3.3: current modality is missing (no target rate), the least frequent is choosen
        if isna(current_target):
            idx_closest_modality = least_frequent

    # finding the closest value
    closest_value = order[idx_closest_modality]

    return closest_value
