"""Base tools to build simple buckets out of Quantitative and Qualitative features
for a binary classification model.
"""

from typing import Any, Dict, List, Union
from warnings import warn

from numpy import argmin, float32, nan, select, sort
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


class GroupedList(list):
    """An ordered list that's extended with a dict."""

    def __init__(self, iterable: Any=()) -> None:
        """An ordered list that historizes its elements' merges.

        Parameters
        ----------
        iterable : Any, optional
            list, dict or GroupedList, by default ()
        """

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




# TODO: add a base discretizer that implements prepare_data (add reset_index ? -> add duplicate column check)
class GroupedListDiscretizer(BaseEstimator, TransformerMixin):
    """Discretizer that uses a dict of grouped values."""

    def __init__(
        self,
        values_orders: Dict[str, GroupedList],
        *,
        copy: bool = False,
        output: str = 'float',
        str_nan: str = None,
        verbose: bool = False,
    ) -> None:
        """Initiates a Discretizer by dict of GroupedList

        Parameters
        ----------
        values_orders : Dict[str, GroupedList]
            Per feature ordering
        copy : bool, optional
            Whether or not to copy the input DataFrame, by default False
        output : type, optional
            Type of the columns to be returned: 'float' or 'str', by default 'float'
        str_nan : str, optional
            Default values attributed to nan, by default None
        verbose : bool, optional
            If True, prints information, by default False
        """

        self.features = list(values_orders.keys())
        self.values_orders = {k: GroupedList(v) for k, v in values_orders.items()}
        self.copy = copy
        self.output = output
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
        Xc = X
        if self.copy:
            Xc = X.copy()

        # filling up nans with specified value
        if self.str_nan:
            Xc[self.features] = Xc[self.features].fillna(self.str_nan)

        # checking that all unique values in X are in values_orders
        check_new_values(Xc, self.features, self.known_values)

        # iterating over each feature
        for n, feature in enumerate(self.features):
            if self.verbose:  # verbose if requested
                print(
                    f" - [GroupedListDiscretizer] Transform {feature} ({n+1}/{len(self.features)})"
                )

            # Selected groups (keys)
            order = self.values_orders.get(feature)

            # converting order to float if requested
            output_order = GroupedList(order)
            if self.output == 'float':
                # creating GroupedList of ints
                output_order = GroupedList(list(range(len(order))))

                # adding values to each key
                for key in output_order:
                    output_order.update({key: order.get(order[key])+[key]})

                # keeping track ther ordering
                self.values_orders.update({feature: output_order})

            # values associated to each key
            group_values = [output_order.get(key) for key in output_order]

            # identifying bucket's key per modality
            values_to_key = [Xc[feature].isin(discarded) for discarded in group_values]

            # case when there are no values
            if len(values_to_key) == 0 & len(values_to_key) == 0:
                pass

            # grouping modalities
            else:
                Xc[feature] = select(values_to_key, output_order, default=Xc[feature])

        # converting to float32
        if self.output == 'float':
            Xc[self.features] = Xc[self.features].astype(float32)

        return Xc


def min_value_counts(x: Series, order: List[Any]) -> float:
    """Minimum of modalities' frequencies."""

    # modality frequency
    values = x.value_counts(dropna=False, normalize=True)

    # ignoring NaNs
    values = values.drop(nan, errors="ignore")

    # adding missing modalities
    values = values.reindex(order).fillna(0)

    # minimal frequency
    min_values = values.values.min()

    return min_values


def value_counts(x: Series, dropna: bool = False, normalize: bool = True) -> dict:
    """Counts the values of each modality of a series into a dictionnary"""

    values = x.value_counts(dropna=dropna, normalize=normalize)

    return values.to_dict()


def target_rate(x: Series, y: Series, dropna: bool = True, ascending=True) -> dict:
    """Target y rate per modality of x into a dictionnary"""

    rates = y.groupby(x, dropna=dropna).mean().sort_values(ascending=ascending)

    return rates.to_dict()


def nunique(x: Series, dropna=False) -> int:
    """Computes number of unique modalities"""

    uniques = unique(x)
    n = len(uniques)

    # removing nans
    if dropna:
        if any(isna(uniques)):
            n -= 1

    return n


def is_equal(a: Any, b: Any) -> bool:
    """checks if a and b are equal (NaN insensitive)"""

    # default equality
    equal = a == b

    # Case where a and b are NaNs
    if isna(a) and isna(b):
        equal = True

    return equal


class ClosestDiscretizer(GroupedListDiscretizer):
    """Discretizes ordered qualitative features into groups more frequent than min_freq.
    NaNs are left untouched.

    Modality is choosen amongst the preceding and following values in the provided order.
    The choosen modality is:
    - the closest in target rate
    - or the one with the lowest frequency
    """

    def __init__(
        self,
        values_orders: Dict[str, Any],
        min_freq: float,
        *,
        copy: bool = False,
        verbose: bool = False,
    ):
        """Initializes a ClosestDiscretizer

        Parameters
        ----------
        values_orders : Dict[str, Any]
            Dict of column names (keys) and modalities' associated order (values)
        min_freq : float
            Minimum frequency per modality. Less frequent modalities are grouped in the closest value of the order.
        default : str, optional
            _description_, by default "worst"
        copy : bool, optional
            _description_, by default False
        verbose : bool, optional
            _description_, by default False
        """

        self.features = list(values_orders.keys())
        self.min_freq = min_freq
        self.values_orders = {k: GroupedList(v) for k, v in values_orders.items()}
        self.copy = copy
        self.verbose = verbose

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
            print(f" - [ClosestDiscretizer] Fit {', '.join(self.features)}")

        # grouping rare modalities for each feature
        common_modalities = (
            X[self.features]
            .apply(
                lambda u: find_common_modalities(
                    u, y, self.min_freq, self.values_orders.get(u.name)
                ),
                axis=0,
                result_type="expand",
            )
        )
        print(common_modalities)
        print(applied_to_dict_list(common_modalities))
        print(common_modalities.T.to_dict())

        # updating the order per feature
        self.values_orders.update({f: common_modalities.get("order").get(f) for f in self.features})

        # discretizing features based on each feature's values_order
        super().__init__(
            self.values_orders,
            copy=self.copy,
            output='str',
        )
        super().fit(X, y)

        return self

# TODO: if there is no target rate, group with the less frequent previous or next modality
def find_common_modalities(
    df_feature: Series,
    y: Series,
    min_freq: float,
    order: GroupedList,
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


    # initialisation de la taille totale du dataframe
    if len_df is None:
        len_df = len(df_feature)

    # case 1: there are missing values
    if any(isna(df_feature)):
        return find_common_modalities(
            df_feature[notna(df_feature)],
            y[notna(df_feature)],
            min_freq,
            order,
            len_df,
        )

    # case 2: no missing values
    # computing frequencies and target rate of each modality
    init_frequencies = df_feature.value_counts(dropna=False, normalize=False) / len_df
    init_values = init_frequencies.index

    # per-modality/value frequencies, filling missing by 0
    frequencies = [
        init_frequencies[value] if value in init_values else 0 for value in order
    ]
    # underrepresented modalities/values
    underrepresented = [
        value for value, frequency in zip(order, frequencies) if frequency < min_freq
    ]

    # case 1: there are underrepresented modalities/values
    while any(underrepresented) & (len(frequencies) > 1):
        # target rate per modality/value
        rate_target = list(
            y.groupby(df_feature).sum().reindex(order) / frequencies / len_df
        )

        # identification de la valeur sous-représentée
        discarded_idx = argmin(frequencies)
        discarded_value = order[discarded_idx]

        # choosing amongst previous and next modality (by volume and target rate)
        kept_value = find_closest_modality(
            discarded_idx,
            order,
            frequencies,
            rate_target,
            min_freq,
        )

        # removing the value from the initial ordering
        order.group(discarded_value, kept_value)

        # removing discarded_value from frequencies
        del frequencies[discarded_idx]
        # removing discarded_value from underrepresented
        underrepresented = [
            value for value, frequency in zip(order, frequencies) if frequency < min_freq
        ]

    # case 2 : no underrepresented value
    return order


def find_closest_modality(idx: int, order: GroupedList, frequencies: List[float], rate_target: Series, min_freq: float) -> Any:
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
        previous_freq, previous_target = frequencies[idx - 1], rate_target[idx - 1]

        # current modality's volume and target rate
        current_target = rate_target[idx]

        # next modality's volume and target rate
        next_freq, next_target = frequencies[idx + 1], rate_target[idx + 1]

        # identifying closest modality in terms of frequency
        least_frequent = idx - 1
        if next_freq < previous_freq:
            least_frequent = idx + 1

        # identifying closest modality in terms of target rate
        closest_target = idx - 1
        if notna(current_target):  # checking that the modality exists in the dataset
            if abs(previous_target - current_target) >= abs(next_target - current_target):
                closest_target = idx + 1
        
        # case 1: grouping with the closest target rate
        idx_closest_modality = closest_target

        # case 2: one modality isn't common, the least frequent is choosen
        if min(previous_freq, next_freq) < min_freq:
            idx_closest_modality = least_frequent

    # finding the closest value
    closest_value = order[idx_closest_modality]

    return closest_value
