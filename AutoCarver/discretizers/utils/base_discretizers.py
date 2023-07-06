"""Base tools to build simple buckets out of Quantitative and Qualitative features
for a binary classification model.
"""

from typing import Any, Dict, List, Union
from warnings import warn

from numpy import argmin, float32, nan, select, sort, ndarray, inf, array
from pandas import DataFrame, Series, isna, notna, unique
from sklearn.base import BaseEstimator, TransformerMixin


def nan_unique(x: Series) -> List[Any]:
    """Unique non-NaN values.

    Parameters
    ----------
    x : Series
        Values to be deduplicated.

    Returns
    -------
    List[Any]
        List of unique non-nan values
    """

    # unique values
    uniques = unique(x)

    # filtering out nans
    uniques = [u for u in uniques if notna(u)]

    return uniques

def applied_to_dict_list(applied: Union[DataFrame, Series]) -> Dict[str, List[Any]]:
    """Converts a DataFrame or a List in a Dict of lists

    Parameters
    ----------
    applied : Union[DataFrame, Series]
        Result of pandas.DataFrame.apply

    Returns
    -------
    Dict[List[Any]]
        Dict of lists of rows values
    """
    # TODO: use this function whenever apply is used

    # case when it's a Series
    converted = applied.to_dict()

    # case when it's a DataFrame
    if isinstance(applied, DataFrame):
        converted = applied.to_dict(orient='list')
    
    return converted



def check_new_values(X: DataFrame, features: List[str], known_values: Dict[str, List[Any]]) -> None:
    """Checks for new, unexpected values, in X 

    Parameters
    ----------
    X : DataFrame
        New DataFrame (at transform time)
    features : List[str]
        List of column names
    known_values : Dict[str, List[Any]]
        Dict of known values per column name
    """
    # unique non-nan values in new dataframe 
    uniques = X[features].apply(
        nan_unique,
        axis=0,
        result_type="expand",
    )
    uniques = applied_to_dict_list(uniques)

    # checking for unexpected values for each feature
    for feature in features:
        unexpected = [val for val in uniques[feature] if val not in known_values[feature]]
        assert (
            len(unexpected) == 0
        ), f"Unexpected value! The ordering for values: {', '.join(unexpected)} of feature '{feature}' was not provided. There might be new values in your test/dev set. Consider taking a bigger test/dev set or dropping the column {feature}."

def check_missing_values(X: DataFrame, features: List[str], known_values: Dict[str, List[Any]]) -> None:
    """Checks for missing values from X, (unexpected values in values_orders) 

    Parameters
    ----------
    X : DataFrame
        New DataFrame (at transform time)
    features : List[str]
        List of column names
    known_values : Dict[str, List[Any]]
        Dict of known values per column name
    """
    # unique non-nan values in new dataframe 
    uniques = X[features].apply(
        nan_unique,
        axis=0,
        result_type="expand",
    )
    uniques = applied_to_dict_list(uniques)

    # checking for unexpected values for each feature
    for feature in features:
        unexpected = [val for val in known_values[feature] if val not in uniques[feature]]
        assert (
            len(unexpected) == 0
        ), f"Unexpected value! The ordering for values: {', '.join(unexpected)} of feature '{feature}' was provided but there are not matching value in provided X. You should check 'values_orders['{feature}']' for unwanted values."


class GroupedList(list):
    """An ordered list that's extended with a dict."""

    def __init__(self, iterable: Any=()) -> None:
        """An ordered list that historizes its elements' merges.

        Parameters
        ----------
        iterable : Any, optional
            list, dict or GroupedList, by default ()
        """

        # case -1: iterable is an array
        if isinstance(iterable, ndarray):
            iterable = list(iterable)

        # case 0: iterable is the contained dict
        if isinstance(iterable, dict):
            # storing ordered keys of the dict
            keys = list(iterable)

            # storing the values contained per key
            self.contained = dict(iterable.items())

            # checking that all values are only present once
            all_values = [val for _, values in iterable.items() for val in values]
            assert len(list(set(all_values))) == len(all_values), "A value is present in several keys (groups)"
            
            # adding key to itself if it's not present in an other key
            keys_copy = keys[:]  # copying initial keys
            for key in keys_copy:
                # checking that the value is not comprised in an other key
                all_values = [val for iter_key, values in iterable.items() for val in values if key != iter_key]
                if key not in all_values:
                    # checking that key is missing from its values
                    if key not in iterable[key]:
                        self.contained.update({key: self.contained[key] + [key]})
                # the key already is in another key (and its values are empty)
                # the key as already been grouped
                else:
                    self.contained.pop(key)
                    keys.remove(key)

            # initiating the list with those keys
            super().__init__(keys)

        # case 1: copying a GroupedList
        elif hasattr(iterable, "contained"):
            # initiating the list with the provided list of keys
            super().__init__(iterable)

            # copying values associated to keys
            self.contained = dict(iterable.contained.items())

        # case 2: initiating GroupedList from a list
        elif isinstance(iterable, list):
            # initiating the list with the provided list of keys
            super().__init__(iterable)

            # initiating the values with the provided list of keys
            self.contained = {v: [v] for v in iterable}

    def get(self, key: Any) -> List[Any]:
        """List of values contained in key

        Parameters
        ----------
        key : Any
            Group.

        Returns
        -------
        List[Any]
            Values contained in key
        """

        # default to fing an element
        found = self.contained.get(key)

        return found

    def group(self, discarded: Any, kept: Any) -> None:
        """Groups the discarded value with the kept value

        Parameters
        ----------
        discarded : Any
            Value to be grouped into the key `to_keep`.
        kept : Any
            Key value in which to group `discarded`.
        """

        # checking that those values are distinct
        if not is_equal(discarded, kept):
            # checking that those values exist in the list
            assert discarded in self, f"{discarded} not in list"
            assert kept in self, f"{kept} not in list"

            # accessing values contained in each value
            contained_discarded = self.contained.get(discarded)
            contained_kept = self.contained.get(kept)

            # updating contained dict
            self.contained.update({kept: contained_discarded + contained_kept, discarded: []})

            # removing discarded from the list
            self.remove(discarded)

    def group_list(self, to_discard: List[Any], to_keep: Any) -> None:
        """Groups elements to_discard into values to_keep

        Parameters
        ----------
        to_discard : List[Any]
            Values to be grouped into the key `to_keep`.
        to_keep : Any
            Key value in which to group `to_discard` values.
        """

        for discarded, kept in zip(to_discard, [to_keep] * len(to_discard)):
            self.group(discarded, kept)

    def append(self, new_value: Any) -> None:
        """Appends a new_value to the GroupedList

        Parameters
        ----------
        new_value : Any
            New key to be added.
        """

        self += [new_value]
        self.contained.update({new_value: [new_value]})

    def update(self, new_value: Dict[Any, List[Any]]) -> None:
        """Updates the GroupedList via a dict"

        Parameters
        ----------
        new_value : Dict[Any, List[Any]]
            Dict of key, values to updated `contained` dict
        """

        # adding keys to the order if they are new values
        self += [key for key, _ in new_value.items() if key not in self]

        # updating contained according to new_value
        self.contained.update(new_value)

    def sort(self):
        """Sorts the values of the list and dict (if any, NaNs are last).

        Returns
        -------
        GroupedList
            Sorted GroupedList
        """
        # str values
        keys_str = [key for key in self if isinstance(key, str)]

        # non-str values
        keys_float = [key for key in self if not isinstance(key, str)]

        # sorting and merging keys
        keys = list(sort(keys_str)) + list(sort(keys_float))

        # recreating an ordered GroupedList
        sorted = GroupedList({k: self.get(k) for k in keys})

        return sorted

    def sort_by(self, ordering: List[Any]) -> None:
        """Sorts the values of the list and dict according to `ordering`, if any, NaNs are the last.

        Parameters
        ----------
        ordering : List[Any]
            Order used for ordering of the list of keys.

        Returns
        -------
        GroupedList
            Sorted GroupedList
        """

        # checking that all values are given an order
        assert all(
            o in self for o in ordering
        ), f"Unknown values in ordering: {', '.join([str(v) for v in ordering if v not in self])}"
        assert all(
            s in ordering for s in self
        ), f"Missing value from ordering: {', '.join([str(v) for v in self if v not in ordering])}"

        # ordering the contained
        sorted = GroupedList({k: self.get(k) for k in ordering})

        return sorted

    def remove(self, value: Any) -> None:
        """Removes a value from the GroupedList

        Parameters
        ----------
        value : Any
            value to be removed
        """
        super().remove(value)
        self.contained.pop(value)

    def pop(self, idx: int) -> None:
        """Pop a value from the GroupedList by index

        Parameters
        ----------
        idx : int
            Index of the value to be popped out
        """
        value = self[idx]
        self.remove(value)

    def get_group(self, value: Any) -> Any:
        """Returns the key (group) containing the specified value

        Parameters
        ----------
        value : Any
            Value for which to find the group.

        Returns
        -------
        Any
            Corresponding key (group)
        """

        found = [
            key
            for key, values in self.contained.items()
            if any(is_equal(value, elt) for elt in values)
        ]

        if any(found):
            return found[0]

        return value

    def values(self) -> List[Any]:
        """All values contained in all groups

        Returns
        -------
        List[Any]
            List of all values in the GroupedList
        """

        known = [value for values in self.contained.values() for value in values]

        return known

    def contains(self, value: Any) -> bool:
        """Checks if a value is contained in any group, also matches NaNs.

        Parameters
        ----------
        value : Any
            Value to search for

        Returns
        -------
        bool
            Whether the value is in the GroupedList 
        """

        known_values = self.values()

        return any(is_equal(value, known) for known in known_values)

    def get_repr(self, char_limit: int = 6) -> List[str]:
        """Returns a representative list of strings of values of groups.

        Parameters
        ----------
        char_limit : int, optional
            Maximum number of character per string, by default 6

        Returns
        -------
        List[str]
            List of short str representation of the keys' values
        """

        # initiating list of group representation
        repr: List[str] = []

        # iterating over each group
        for group in self:
            # accessing group's values
            values = self.get(group)

            if len(values) == 0:  # case 0: there are no value in this group
                pass

            elif len(values) == 1:  # case 1: there is only one value in this group
                repr += [values[0]]

            elif len(values) == 2:  # case 2: two values in this group
                repr += [f"{values[1]}"[:char_limit] + " and " + f"{values[0]}"[:char_limit]]

            elif len(values) > 2:  # case 3: more than two values in this group
                repr += [f"{values[-1]}"[:char_limit] + " to " + f"{values[0]}"[:char_limit]]

        return repr


# TODO: add a summary
# TODO: output a json
# TODO: GroupedListDiscretizer for quantitative features?
# TODO: add a base discretizer that implements prepare_data (add reset_index ? -> add duplicate column check)
class GroupedListDiscretizer(BaseEstimator, TransformerMixin):
    """Discretizer that uses a dict of grouped values."""

    def __init__(
        self,
        features: List[str],
        values_orders: Dict[str, GroupedList],
        *,
        copy: bool = False,
        input_dtype: str = 'str',
        output_dtype: str = 'str',
        str_nan: str = None,
        verbose: bool = False,
    ) -> None:
        """Initiates a Discretizer by dict of GroupedList

        Parameters
        ----------
        features : List[str]
            List of column names to be discretized
        values_orders : Dict[str, GroupedList]
            Per feature ordering
        copy : bool, optional
            Whether or not to copy the input DataFrame, by default False
        input_dtype : type, optional
            Type of the input columns:
            - if 'float' uses transform_quantitative
            - if 'str' uses transform_qualitative,
            by default 'str'
        output_dtype : type, optional
            Type of the columns to be returned: 'float' or 'str', by default 'str'
        str_nan : str, optional
            Default values attributed to nan, by default None
        verbose : bool, optional
            If True, prints information, by default False
        """

        self.features = features[:]
        self.values_orders = {k: GroupedList(v) for k, v in values_orders.items()}
        self.copy = copy
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype
        self.verbose = verbose
        self.str_nan = str_nan

        self.known_values = {feature: self.values_orders[feature].values() for feature in self.features}

    def fit(self, X: DataFrame, y: Series = None) -> None:
        """Learns the known values for each feature

        Parameters
        ----------
        X : DataFrame
            Contains columns named after `values_orders` keys
        y : Series, optional
            Model target, by default None
        """

        return self

    def transform(self, X: DataFrame, y: Series = None) -> DataFrame:
        """Groups values of features (values_orders keys) according to
        there corresponding GroupedList (values_orders values) based on
        the `GroupedList.contained` dict.

        Parameters
        ----------
        X : DataFrame
            Contains columns named after `values_orders` keys
        y : Series, optional
            Model target, by default None

        Returns
        -------
        DataFrame
            _description_
        """

        # copying dataframes
        x_copy = X
        if self.copy:
            x_copy = X.copy()

        # checking for input type
        if self.input_dtype == 'float':  # quantitative features
            x_copy = self.transform_quantitative(x_copy, y)

        else:  # quantitative features
            x_copy = self.transform_qualitative(x_copy, y)

        return x_copy

    def transform_quantitative(self, X: DataFrame, y: Series) -> DataFrame:
        """Groups values of features (values_orders keys) according to
        there corresponding GroupedList (values_orders values) based on
        the `GroupedList.contained` dict.

        Parameters
        ----------
        X : DataFrame
            Contains columns named after `values_orders` keys
        y : Series, optional
            Model target, by default None

        Returns
        -------
        DataFrame
            _description_
        """
        # getting group "name" per quantile
        quantiles_aliases = get_quantiles_aliases(self.features, self.values_orders, self.str_nan)

        # dataset length
        x_len = X.shape[0]

        for n, feature in enumerate(self.features):
            if self.verbose:  # verbose if requested
                print(f" - [GroupedListDiscretizer] Transform Qualitative {feature} ({n+1}/{len(self.features)})")

            nans = isna(X[feature])  # keeping track of nans

            # quantiles ordering for the feature
            order = self.values_orders[feature]

            # filtering out nan if any
            quantiles = list(order)
            if self.str_nan in quantiles:
                quantiles = [val for val in quantiles if val != self.str_nan]

            # list of masks of values to replace with there respective group
            values_to_group = [X[feature] <= q for q in quantiles]

            # corressponding group for each value
            group_key = [[quantiles_aliases[feature][quantile]] * x_len for quantile in quantiles if quantile != self.str_nan] 
            
            # grouping values into groups
            X[feature] = select(values_to_group, group_key, default=X[feature])

            # converting nans
            if any(nans):
                assert self.str_nan in order, f"Unexpected value! Missing values found for feature '{feature}' at transform step but not during fit. There might be new values in your test/dev set. Consider taking a bigger test/dev set or dropping the column {feature}."
                X.loc[nans, feature] = self.str_nan

        return X

    def transform_qualitative(self, X: DataFrame, y: Series = None) -> DataFrame:
        """Groups values of features (values_orders keys) according to
        there corresponding GroupedList (values_orders values) based on
        the `GroupedList.contained` dict.

        Parameters
        ----------
        X : DataFrame
            Contains columns named after `values_orders` keys
        y : Series, optional
            Model target, by default None

        Returns
        -------
        DataFrame
            _description_
        """

        # filling up nans with specified value
        if self.str_nan:
            X[self.features] = X[self.features].fillna(self.str_nan)

        # checking that all unique values in X are in values_orders
        check_new_values(X, self.features, self.known_values)

        # iterating over each feature
        for n, feature in enumerate(self.features):
            if self.verbose:  # verbose if requested
                print(f" - [GroupedListDiscretizer] Transform Qualitative {feature} ({n+1}/{len(self.features)})")

            # Selected groups (keys)
            order = self.values_orders[feature]

            # converting order to float if requested
            output_order = GroupedList(order)  # copy of order
            if self.output_dtype == 'float':
                # creating GroupedList of ints
                output_order = GroupedList(list(range(len(order))))

                # adding values to each key
                for key in output_order:
                    output_order.update({key: order.get(order[key]) + [key]})

                # keeping track of the ordering
                self.values_orders.update({feature: output_order})

            # values associated to each key
            group_values = [output_order.get(key) for key in output_order]

            # identifying bucket's key per modality
            values_to_key = [X[feature].isin(discarded) for discarded in group_values]

            # case when there are no values
            if len(values_to_key) == 0 & len(values_to_key) == 0:
                pass

            # grouping modalities
            else:
                X[feature] = select(values_to_key, output_order, default=X[feature])

        # converting to float32
        if self.output_dtype == 'float':
            X[self.features] = X[self.features].astype(float32)

        return X

def min_value_counts(x: Series, dropna: bool = False, normalize: bool = True) -> float:
    """Minimum of modalities' frequencies.

    Parameters
    ----------
    x : Series
        _description_
    values_orders : Dict[str, GroupedList]
        _description_

    Returns
    -------
    float
        _description_
    """
    # modality frequency
    values = x.value_counts(dropna=dropna, normalize=normalize)

    # minimal frequency
    min_values = values.values.min()

    return min_values


def value_counts(x: Series, dropna: bool = False, normalize: bool = True) -> dict:
    """Counts the values of each modality of a series into a dictionnary

    Parameters
    ----------
    x : Series
        _description_
    dropna : bool, optional
        _description_, by default False
    normalize : bool, optional
        _description_, by default True

    Returns
    -------
    dict
        _description_
    """

    values = x.value_counts(dropna=dropna, normalize=normalize)

    return values.to_dict()


def target_rate(x: Series, y: Series, dropna: bool = True, ascending=True) -> dict:
    """Target y rate per modality of x into a dictionnary

    Parameters
    ----------
    x : Series
        _description_
    y : Series
        _description_
    dropna : bool, optional
        _description_, by default True
    ascending : bool, optional
        _description_, by default True

    Returns
    -------
    dict
        _description_
    """

    rates = y.groupby(x, dropna=dropna).mean().sort_values(ascending=ascending)

    return rates.to_dict()


def nunique(x: Series, dropna=False) -> int:
    """Computes number of unique modalities

    Parameters
    ----------
    x : Series
        _description_
    dropna : bool, optional
        _description_, by default False

    Returns
    -------
    int
        _description_
    """

    uniques = unique(x)
    n = len(uniques)

    # removing nans
    if dropna:
        if any(isna(uniques)):
            n -= 1

    return n


def is_equal(a: Any, b: Any) -> bool:
    """Checks if a and b are equal (NaN insensitive)

    Parameters
    ----------
    a : Any
        _description_
    b : Any
        _description_

    Returns
    -------
    bool
        _description_
    """

    # default equality
    equal = a == b

    # Case where a and b are NaNs
    if isna(a) and isna(b):
        equal = True

    return equal

def get_aliases(quantiles: List[float], str_nan: str) -> List[str]:
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
    List[str]
        _description_
    """
    # filtering out nan for formatting
    if str_nan in quantiles:
        quantiles = [val for val in quantiles if val != str_nan]
        
    # converting quantiles in string
    aliases = format_quantiles(quantiles[:-1])  # removing inf

    return aliases

def get_quantiles_aliases(features: List[str], values_orders: Dict[str, GroupedList], str_nan: str) -> Dict[str, GroupedList]:
    """Converts a values_orders of quantiles into a values_orders of string quantiles

    Parameters
    ----------
    features : List[str]
        _description_
    values_orders : Dict[str, GroupedList]
        _description_
    str_nan : str
        _description_

    Returns
    -------
    Dict[str, GroupedList]
        _description_
    """
    # applying quantiles formatting to orders of specified features
    quantiles_aliases = {}
    for feature in features:
        quantiles = list(values_orders[feature])
        aliases = get_aliases(quantiles, str_nan)
        # associates quantiles to their respective aliases
        quantiles_aliases.update({feature: {quantile: alias for quantile, alias in zip(quantiles, aliases)}})

    return quantiles_aliases

def get_aliases_quantiles(features: List[str], values_orders: Dict[str, GroupedList], str_nan: str) -> Dict[str, GroupedList]:
    """Converts a values_orders of quantiles into a values_orders of string quantiles

    Parameters
    ----------
    features : List[str]
        _description_
    values_orders : Dict[str, GroupedList]
        _description_
    str_nan : str
        _description_

    Returns
    -------
    Dict[str, GroupedList]
        _description_
    """
    # applying quantiles formatting to orders of specified features
    aliases_quantiles = {}
    for feature in features:
        quantiles = list(values_orders[feature])
        aliases = get_aliases(quantiles, str_nan)
        # associates quantiles to their respective aliases
        aliases_quantiles.update({feature: {alias: quantile for quantile, alias in zip(quantiles, aliases)}})

    return aliases_quantiles 

def format_quantiles(a_list: List[float]) -> List[str]:
    """Formats a list of float quantiles into a list of boundaries.

    Rounds quantiles to the closest power of 1000.

    Parameters
    ----------
    a_list : List[float]
        Sorted list of quantiles to convert into string

    Returns
    -------
    List[str]
        List of boundaries per quantile
    """

    # finding the closest power of thousands for each element
    closest_powers = [
        next((k for k in range(-3, 4) if abs(elt) / 100 // 1000 ** (k) < 10)) for elt in a_list
    ]

    # rounding elements to the closest power of thousands
    rounded_to_powers = [elt / 1000 ** (k) for elt, k in zip(a_list, closest_powers)]

    # computing optimal decimal per unique power of thousands
    optimal_decimals: Dict[str, int] = {}
    for power in unique(closest_powers):  # iterating over each power of thousands found
        # filtering on the specific power of thousands
        sub_array = array([elt for elt, p in zip(rounded_to_powers, closest_powers) if power == p])

        # number of distinct values
        n_uniques = sub_array.shape[0]

        # computing the first rounding decimal that allows for distinction of
        # each values when rounded, by default None (no rounding)
        optimal_decimal = next(
            (k for k in range(1, 10) if len(unique(sub_array.round(k))) == n_uniques),
            None,
        )

        # storing in the dict
        optimal_decimals.update({power: optimal_decimal})

    # rounding according to each optimal_decimal
    rounded_list: List[float] = []
    for elt, power in zip(rounded_to_powers, closest_powers):
        # rounding each element
        rounded = elt  # by default None (no rounding)
        optimal_decimal = optimal_decimals.get(power)
        if optimal_decimal:  # checking that the optimal decimal exists
            rounded = round(elt, optimal_decimal)

        # adding the rounded number to the list
        rounded_list += [rounded]

    # dict of suffixes per power of thousands
    suffixes = {-3: "n", -2: "mi", -1: "m", 0: "", 1: "K", 2: "M", 3: "B"}

    # adding the suffixes
    formatted_list = [
        f"{elt: 1.1f}{suffixes[power]}" for elt, power in zip(rounded_list, closest_powers)
    ]

    # keeping zeros
    formatted_list = [
        rounded if raw != 0 else f"{raw: 1.1f}" for rounded, raw in zip(formatted_list, a_list)
    ]
    
    # stripping whitespaces
    formatted_list = [string.strip() for string in formatted_list]

    # low and high bounds per quantiles
    upper_bounds = formatted_list + [nan]
    lower_bounds = [nan] + formatted_list
    order: List[str] = []
    for lower, upper in zip(lower_bounds, upper_bounds):
        if isna(lower):
            order += [f"x <= {upper}"]
        elif isna(upper):
            order += [f"{lower} < x"]
        else:
            order += [f"{lower} < x <= {upper}"]

    return order
