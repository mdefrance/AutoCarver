"""Tools to build simple buckets out of Qualitative features
for a binary classification model.
"""

from typing import Any, Dict, List, Union
from AutoCarver.discretizers.utils.base_discretizers import GroupedList

from numpy import argmin, nan, select
from pandas import DataFrame, Series, isna, notna, unique

from .base_discretizers import (
    GroupedList,
    GroupedListDiscretizer,
    check_missing_values,
    check_new_values,
    convert_to_labels,
    convert_to_values,
    nan_unique,
    target_rate,
    value_counts,
)
from .type_discretizers import StringDiscretizer

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
        # Initiating GroupedListDiscretizer
        super().__init__(
            features=features,
            values_orders=values_orders,
            input_dtypes='str',
            output_dtype='str',
            str_nan=str_nan,
            copy=copy,
        )
        
        self.min_freq = min_freq
        self.str_default = str_default
        self.verbose = verbose

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
        # checking for binary target
        x_copy = super().prepare_data(X, y)

        # checks and initilizes values_orders
        for feature in self.features:
            # initiating features missing from values_orders
            if feature not in self.values_orders:
                self.values_orders.update({feature: GroupedList(nan_unique(x_copy[feature]))})

        # checking that all unique values in X are in values_orders
        check_new_values(x_copy, self.features, self.values_orders)
        # checking that all unique values in values_orders are in X
        check_missing_values(x_copy, self.features, self.values_orders)

        # adding NANS
        for feature in self.features:
            if any(x_copy[feature].isna()):
                self.values_orders[feature].append(self.str_nan)

        # filling up NaNs
        x_copy = x_copy[self.features].fillna(self.str_nan)

        return x_copy

    def fit(self, X: DataFrame, y: Series) -> None:
        """Learns modalities that are common enough (greater than min_freq)

        Parameters
        ----------
        X : DataFrame
            _description_
        y : Series
            _description_
        """
        # copying dataframe and checking data before bucketization
        x_copy = self.prepare_data(X, y)
        
        if self.verbose:  # verbose if requested
            print(f" - [DefaultDiscretizer] Fit {', '.join(self.features)}")

        # computing frequencies of each modality
        frequencies = x_copy.apply(value_counts, normalize=True, axis=0)

        for feature in self.features:
            # sorting orders based on target rates
            order = self.values_orders[feature]

            # checking for rare values
            values_to_group = [
                val
                for val, freq in frequencies[feature].items()
                if freq < self.min_freq and val != self.str_nan
            ]
            if any(values_to_group):
                # adding default value to the order
                order.append(self.str_default)

                # grouping rare values in default value
                order.group_list(values_to_group, self.str_default)
                x_copy.loc[x_copy[feature].isin(values_to_group), feature] = self.str_default

                # updating values_orders
                self.values_orders.update({feature: order})

        # computing target rate per modality for ordering
        target_rates = x_copy.apply(target_rate, y=y, ascending=True, axis=0)

        # sorting orders based on target rates
        for feature in self.features:
            order = self.values_orders[feature]

            # new ordering according to target rate
            new_order = list(target_rates[feature])

            # leaving NaNs at the end of the list
            if self.str_nan in new_order:
                new_order.remove(self.str_nan)
                new_order += [self.str_nan]

            # sorting order updating values_orders
            self.values_orders.update({feature: order.sort_by(new_order)})

        # discretizing features based on each feature's values_order
        super().fit(X, y)

        return self

class BaseQualitativeDiscretizer(GroupedListDiscretizer):
    """TODO: to implement to mutualize prepare_data and remove_feature across qualitative discretizers"""

    def __init__(
        self,
        features: List[str],
        values_orders: Dict[str, GroupedList],
        *,
        copy: bool = False,
        input_dtypes: Union[str, Dict[str, str]] = None,
        str_nan: str = None,
        verbose: bool = False
    ) -> None:
        super().__init__(features, values_orders, copy=copy, input_dtypes=input_dtypes, str_nan=str_nan, verbose=verbose)

class ChainedDiscretizer(GroupedListDiscretizer):
    """Chained Discretization based on a list of GroupedList."""

    def __init__(
        self,
        features: list[str],
        min_freq: float,
        chained_orders: list[GroupedList],
        *,
        values_orders: dict[str, GroupedList] = None,
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
        # Initiating GroupedListDiscretizer
        super().__init__(
            features=features,
            values_orders=values_orders,
            input_dtypes='str',
            output_dtype='str',
            str_nan=str_nan,
            copy=copy,
        )

        self.min_freq = min_freq

        self.chained_orders = [GroupedList(values) for values in chained_orders]

        # parameters to handle missing/unknown values
        self.remove_unknown = remove_unknown

        # known_values: all ordered values describe in each level of the chained_orders
        # starting off with first level 
        known_values = self.chained_orders[0].values()
        # adding each level
        for next_level in self.chained_orders[1:]:
            # highest value per group of the level
            highest_ranking_value = {group: [value for value in values if value!=group][-1] for group, values in next_level.contained.items()}
            
            # adding next_level group to the order
            for group, highest_value in highest_ranking_value.items():
                highest_index = known_values.index(highest_value)
                known_values = known_values[:highest_index + 1] + [group] + known_values[highest_index + 1:]
        self.known_values = known_values

        # adding known_values to each feature's order
        for feature in self.features:
            # checking for already known values of the feature
            if feature in self.values_orders:
                order = self.values_orders[feature]
            # no known values for the feature
            else:
                order = GroupedList([])
            # checking that all values from the order are in known_values
            for value in order:
                assert value in self.known_values, f"Value {value} from feature {feature} provided in values_orders is missing from levels of chained_orders. Add value to a level of chained_orders or adapt values_orders."
            # adding known values if missing from the order
            for value in self.known_values:
                if value not in order.values():
                    order.append(value)
            order = order.sort_by(self.known_values)
            self.values_orders.update({feature: order})

        self.verbose = verbose

    def prepare_data(self, X: DataFrame, y: Series = None) -> DataFrame:
        """Prepares the data for bucketization, checks column types.
        Converts non-string columns into strings.

        Parameters
        ----------
        X : DataFrame
            Dataset to be bucketized
        y : Series
            Model target, by default None

        Returns
        -------
        DataFrame
            Formatted X for bucketization
        """
        # copying dataframe
        x_copy = X.copy()

        # checking for ids (unique value per row)
        frequencies = x_copy[self.features].apply(
            lambda u: u.value_counts(normalize=True, dropna=False).drop(nan, errors='ignore').max(), axis=0
        )
        # for each feature, checking that at least one value is more frequent than min_freq
        for feature in self.features:
            if frequencies[feature] < self.min_freq:
                print(f"For feature '{feature}', the largest modality has {frequencies[feature]:2.2%} observations which is lower than {self.min_freq:2.2%}. This feature will not be Discretized. Consider decreasing parameter min_freq or removing this feature.")
                self.remove_feature(feature)

        # checking for columns containing floats or integers even with filled nans
        dtypes = x_copy[self.features].fillna(self.str_nan).applymap(type).apply(unique)
        not_object = dtypes.apply(lambda u: any(typ != str for typ in u))

        # non qualitative features detected
        if any(not_object):
            features_to_convert = list(not_object.index[not_object])
            if self.verbose:
                unexpected_dtypes = [typ for dtyp in dtypes[not_object] for typ in dtyp if typ != str]
                print(
                    f"""Non-string features: {str(features_to_convert)}. Trying to convert them using type_discretizers.StringDiscretizer, otherwise convert them manually. Unexpected data types: {str(list(unexpected_dtypes))}."""
                )

            # converting specified features into qualitative features
            stringer = StringDiscretizer(features=features_to_convert, values_orders=self.values_orders)
            x_copy = stringer.fit_transform(x_copy)

            # updating values_orders accordingly
            self.values_orders.update(stringer.values_orders)

        # all known values for features
        known_values = {feature: values.values() for feature, values in self.values_orders.items()}

        # checking that all unique values in X are in values_orders
        check_new_values(x_copy, self.features, known_values)

        # filling nans
        x_copy = x_copy.fillna(self.str_nan)

        return x_copy
    
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
        x_copy = self.prepare_data(X, y)

        # iterating over each feature
        for n, feature in enumerate(self.features):
            if self.verbose:  # verbose if requested
                print(f" - [ChainedDiscretizer] Fit {feature} ({n+1}/{len(self.features)})")

            # computing frequencies of each modality
            frequencies = x_copy[feature].value_counts(normalize=True)
            values, frequencies = frequencies.index, frequencies.values

            # adding NaNs to the order if any
            order = self.values_orders[feature]
            if self.str_nan in values:
                order.append(self.str_nan)

            # checking for unknown values (missing from known_values)
            missing = [value for value in values if value not in self.known_values and value != self.str_nan]

            # converting unknown values to NaN
            if self.remove_unknown & (len(missing) > 0):
                # alerting user
                print(
                    f"Order for feature '{feature}' was not provided for values:  {str(missing)}, these values will be converted to '{self.str_nan}' (policy remove_unknown=True)"
                )

                # adding missing values to the order
                order.update({self.str_nan: missing + order.get(self.str_nan)})

            # alerting user
            else:
                assert (
                    not len(missing) > 0
                ), f"Order for feature '{feature}' needs to be provided for values: {str(missing)}, otherwise set remove_unknown=True"

            # iterating over each specified orders
            for level_order in self.chained_orders:  # TODO replace all of this with labels_per_orders
                # values that are frequent enough
                to_keep = list(values[frequencies >= self.min_freq])

                # values from the order to group (not frequent enough or absent)
                values_to_group = [value for value in level_order.values() if value not in to_keep]

                # values to group into discarded values
                groups_value = [level_order.get_group(value) for value in values_to_group]

                # values of the feature to input (needed for next levels of the order)
                df_to_input = [x_copy[feature] == discarded for discarded in values_to_group]

                # inputing non frequent values
                x_copy[feature] = select(df_to_input, groups_value, default=x_copy[feature])

                # historizing in the feature's order
                for discarded, kept in zip(values_to_group, groups_value):
                    self.values_orders.get(feature).group(discarded, kept)

                # updating frequencies of each modality for the next ordering
                frequencies = x_copy[feature].value_counts(normalize=True)
                values, frequencies = frequencies.index, frequencies.values

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
        features: list[str],
        values_orders: dict[str, GroupedList],
        min_freq: float,
        *,
        str_nan: str = "__NAN__",
        input_dtypes: Union[str, dict[str, str]] = "str",
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
        input_dtypes : Union[str, Dict[str, str]], optional
            String of type to be considered for all features or
            Dict of column names and associated types:
            - if 'float' uses transform_quantitative
            - if 'str' uses transform_qualitative,
            default 'str'
        copy : bool, optional
            _description_, by default False
        verbose : bool, optional
            _description_, by default False
        """
        # Initiating GroupedListDiscretizer
        super().__init__(
            features=features,
            values_orders=values_orders,
            input_dtypes=input_dtypes,
            output_dtype='str',
            str_nan=str_nan,
            copy=copy,
        )

        self.min_freq = min_freq
        self.verbose = verbose

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
        if self.verbose:  # verbose if requested
            print(f" - [OrdinalDiscretizer] Fit {', '.join(self.features)}")

        # checking values orders
        x_copy = self.prepare_data(X, y)

        # converting potential quantiles into there labels
        known_orders = convert_to_labels(
            features=self.features,
            quantitative_features=self.quantitative_features,
            values_orders=self.values_orders,
            str_nan=self.str_nan,
            dropna=True,
        )

        # grouping rare modalities for each feature
        common_modalities = x_copy[self.features].apply(
            find_common_modalities,
            y=y,
            min_freq=self.min_freq,
            values_orders=known_orders,
            axis=0,
            result_type="reduce",
        )

        # converting potential labels into there respective values (quantiles)
        self.values_orders.update(
            convert_to_values(
                features=self.features,
                quantitative_features=self.quantitative_features,
                values_orders=self.values_orders,
                label_orders=common_modalities,
                str_nan=self.str_nan,
            )
        )

        # discretizing features based on each feature's values_order
        super().fit(x_copy, y)

        return self


def find_common_modalities(
    df_feature: Series,
    y: Series,
    min_freq: float,
    values_orders: dict[str, GroupedList],
    len_df: int = None,
) -> dict[str, Any]:
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
    grouped_series = (
        series.groupby(list(map(order.get_group, series.index)), dropna=False).sum().reindex(order)
    )

    return grouped_series


def find_closest_modality(
    idx: int, order: GroupedList, frequencies: List[float], target_rates: Series, min_freq: float
) -> Any:
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
