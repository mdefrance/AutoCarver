"""Base tools to build simple buckets out of Quantitative and Qualitative features
for a binary classification model.
"""

from typing import Any, Dict, List
from warnings import warn

from numpy import argmin, float32, nan, select, sort
from pandas import DataFrame, Series, isna, notna, unique
from sklearn.base import BaseEstimator, TransformerMixin


def nan_unique(x: Series):
    """Unique non-NaN values."""

    # unique values
    uniques = unique(x)

    # filtering out nans
    uniques = [u for u in uniques if notna(u)]

    return uniques


class GroupedList(list):
    """An ordered list that extends dict."""

    def __init__(self, iterable=()) -> None:
        """An ordered list that historizes its elements' merges."""

        # case 0: iterable is the contained dict
        if isinstance(iterable, dict):
            # TODO: check thaht keys are in list

            # récupération des valeurs de la liste (déjà ordonné)
            values = [key for key in iterable]

            # initialsiation de la liste
            super().__init__(values)

            # attribution des valeurs contenues
            self.contained = {k: v for k, v in iterable.items()}

            # adding key to itself if that's not the case
            for k in [k for k in values if k not in self.contained.get(k)]:
                self.contained.update({k: self.contained.get(k) + [k]})

        # case 1: copying a GroupedList
        elif hasattr(iterable, "contained"):
            # initialsiation de la liste
            super().__init__(iterable)

            # copie des groupes
            self.contained = {k: v for k, v in iterable.contained.items()}

        # case 2: initiating GroupedList from a list
        elif isinstance(iterable, list):
            # initialsiation de la liste
            super().__init__(iterable)

            # création des groupes
            self.contained = {v: [v] for v in iterable}

    def group_list(self, to_discard: List[Any], to_keep: Any) -> None:
        """Groups elements to_discard into values to_keep"""

        for discarded, kept in zip(to_discard, [to_keep] * len(to_discard)):
            self.group(discarded, kept)

    def group(self, discarded: Any, kept: Any) -> None:
        """Groups the discarded value with the kept value"""

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

        return self

    def append(self, new_value: Any) -> None:
        """Appends a new_value to the GroupedList"""

        self += [new_value]

        self.contained.update({new_value: [new_value]})

        return self

    def update(self, new_value: Dict[Any, List[Any]]) -> None:
        """Updates the GroupedList via a dict"""

        # adding keys to the order if they are new values
        for k in [c for c in new_value if c not in self]:
            self += new_value.keys()

        # updating contained accord to new_value
        self.contained.update(new_value)

        return self

    def sort(self) -> None:
        """Sorts the values of the list and dict (if any, NaNs are last)."""

        # str values
        keys_str = [key for key in self if isinstance(key, str)]

        # non-str values
        keys_float = [key for key in self if not isinstance(key, str)]

        # sorting and merging keys
        keys = list(sort(keys_str)) + list(sort(keys_float))

        # recreating an ordered GroupedList
        self = GroupedList({k: self.get(k) for k in keys})

        return self

    def sort_by(self, ordering: List[Any]) -> None:
        """Sorts the values of the list and dict, if any, NaNs are the last."""

        # checking that all values are given an order
        assert all(
            o in self for o in ordering
        ), f"Unknown values in ordering: {', '.join([str(v) for v in ordering if v not in self])}"
        assert all(
            s in ordering for s in self
        ), f"Missing value from ordering: {', '.join([str(v) for v in self if v not in ordering])}"

        # ordering the contained
        self = GroupedList({k: self.get(k) for k in ordering})

        return self

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

    def get(self, key: Any) -> List[Any]:
        """returns list of values contained in key"""

        # default to fing an element
        found = self.contained.get(key)

        # copying with dictionnaries (not working with numpy.nan)
        # if isna(key):
        # found = [value for dict_key, value in self.contained.items() if is_equal(dict_key, key)][0]

        return found

    def get_group(self, value: Any) -> Any:
        """returns the group containing the specified value"""

        found = [
            key
            for key, values in self.contained.items()
            if any(is_equal(value, elt) for elt in values)
        ]

        if any(found):
            return found[0]
        
        return value

    def values(self) -> List[Any]:
        """returns all values contained in each group"""

        known = [value for values in self.contained.values() for value in values]

        return known

    def contains(self, value: Any) -> bool:
        """checks if a value if contained in any group"""

        known_values = self.values()

        return any(is_equal(value, known) for known in known_values)

    def get_repr(self, char_limit: int = 10) -> List[str]:
        """Returns a representative list of strings of values of groups."""

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


class GroupedListDiscretizer(BaseEstimator, TransformerMixin):
    """Discretizer that uses a dict of grouped values."""

    def __init__(
        self,
        values_orders: Dict[str, GroupedList],
        *,
        copy: bool = False,
        output: type = float,
        str_nan: str = None,
        verbose: bool = False,
    ) -> None:
        """Initiates a Discretizer that uses a dict of GroupedList

        Parameters
        ----------
        values_orders : Dict[str, GroupedList]
            Per feature ordering
        copy : bool, optional
            Whether or not to copy the input DataFrame, by default False
        output : type, optional
            Type of the columns to be returned: float or str, by default float
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

    def fit(self, X: DataFrame, y: Series = None) -> None:
        """_summary_

        Parameters
        ----------
        X : DataFrame
            _description_
        y : Series, optional
            _description_, by default None
        """        
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

        # copying dataframes
        Xc = X
        if self.copy:
            Xc = X.copy()

        # filling up nans with specified value
        if self.str_nan:
            Xc[self.features] = Xc[self.features].fillna(self.str_nan)

        # iterating over each feature
        for n, feature in enumerate(self.features):
            # verbose if requested
            if self.verbose:
                print(
                    f" - [GroupedListDiscretizer] Transform {feature} ({n+1}/{len(self.features)})"
                )

            # bucketizing feature
            order = self.values_orders.get(feature)  # récupération des groupes
            to_discard = [
                order.get(group) for group in order
            ]  # identification des valeur à regrouper
            to_input = [
                Xc[feature].isin(discarded) for discarded in to_discard
            ]  # identifying main bucket value
            to_keep = [
                n if self.output == float else group for n, group in enumerate(order)
            ]  # récupération du groupe dans lequel regrouper

            # case when there are no values
            if len(to_input) == 0 & len(to_keep) == 0:
                pass

            # grouping modalities
            else:
                Xc[feature] = select(to_input, to_keep, default=Xc[feature])

        # converting to float
        if self.output == float:
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


class ClosestDiscretizer(BaseEstimator, TransformerMixin):
    """Discretizes ordered qualitative features into groups more frequent than min_freq"""
    def __init__(
        self,
        values_orders: Dict[str, Any],
        min_freq: float,
        *,
        default: str = "worst",
        copy: bool = False,
        verbose: bool = False,
    ):
        """_summary_

        Parameters
        ----------
        values_orders : Dict[str, Any]
            _description_
        min_freq : float
            _description_
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
        self.default = default[:]
        self.default_values: Dict[str, Any] = {}
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
            .T.to_dict()
        )

        # updating the order per feature
        self.values_orders.update({f: common_modalities.get("order").get(f) for f in self.features})

        # defining the default value based on the strategy
        self.default_values = {f: common_modalities.get(self.default).get(f) for f in self.features}

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

        # iterating over each feature
        for n, feature in enumerate(self.features):
            # printing verbose if requested
            if self.verbose:
                print(f" - [ClosestDiscretizer] Transform {feature} ({n+1}/{len(self.features)})")

            # accessing feature's modalities' order
            order = self.values_orders.get(feature)

            # imputation des valeurs inconnues le cas échéant
            unknowns = [
                value
                for value in Xc[feature].unique()
                if not any(is_equal(value, known) for known in order.values())
            ]
            unknowns = [value for value in unknowns if notna(value)]  # suppression des NaNs
            if any(unknowns):
                to_input = [
                    Xc[feature] == unknown for unknown in unknowns
                ]  # identification des valeurs à regrouper
                Xc[feature] = select(
                    to_input,
                    [self.default_values.get(feature)],
                    default=Xc[feature],
                )  # regroupement des valeurs
                warn(f"Unknown modalities provided for feature '{feature}': {unknowns}")

            # grouping values inside groups of modalities
            to_discard = [
                order.get(group) for group in order
            ]  # identification des valeur à regrouper
            to_input = [
                Xc[feature].isin(discarded) for discarded in to_discard
            ]  # identification des valeurs à regrouper
            Xc[feature] = select(to_input, order, default=Xc[feature])  # regroupement des valeurs

        return Xc


def find_common_modalities(
    df_feature: Series,
    y: Series,
    min_freq: float,
    order: GroupedList,
    len_df: int = None,
) -> Dict[str, Any]:
    """[Qualitative] Découpage en modalités de fréquence minimal (Cas des variables ordonnées).

    Fonction récursive : on prend l'échantillon et on cherche des valeurs sur-représentées
    Si la valeur existe on la met dans une classe et on cherche à gauche et à droite de celle-ci, d'autres variables sur-représentées
    Une fois il n'y a plus de valeurs qui soient sur-représentées,
    on fait un découpage classique avec qcut en multipliant le nombre de classes souhaité par le pourcentage de valeurs restantes sur le total

    """

    # initialisation de la taille totale du dataframe
    if len_df is None:
        len_df = len(df_feature)

    # conversion en GroupedList si ce n'est pas le cas
    order = GroupedList(order)

    # cas 1 : il y a des valeurs manquantes NaN
    if any(isna(df_feature)):
        return find_common_modalities(
            df_feature[notna(df_feature)],
            y[notna(df_feature)],
            min_freq,
            order,
            len_df,
        )

    # cas 2 : il n'y a que des valeurs dans le dataframe (pas de NaN)
    else:
        # computing frequencies and target rate of each modality
        init_frequencies = df_feature.value_counts(dropna=False, normalize=False) / len_df
        init_values, init_frequencies = (
            init_frequencies.index,
            init_frequencies.values,
        )

        # ordering
        frequencies = [
            init_frequencies[init_values == value][0] if any(init_values == value) else 0
            for value in order
        ]  # sort selon l'ordre des modalités
        values = [
            init_values[init_values == value][0] if any(init_values == value) else value
            for value in order
        ]  # sort selon l'ordre des modalités
        rate_target = (
            y.groupby(df_feature).sum().reindex(order) / frequencies / len_df
        )  # target rate per modality
        underrepresented = [
            value for value, frequency in zip(values, frequencies) if frequency < min_freq
        ]  # valeur peu fréquentes

        # cas 1 : il existe une valeur sous-représentée
        while any(underrepresented) & (len(frequencies) > 1):
            # identification de la valeur sous-représentée
            discarded_idx = argmin(frequencies)
            discarded = values[discarded_idx]

            # identification de la modalité la plus proche (volume et taux de défaut)
            kept = find_closest_modality(
                discarded,
                discarded_idx,
                list(zip(order, frequencies, rate_target)),
                min_freq,
            )

            # removing the value from the initial ordering
            order.group(discarded, kept)

            # ordering
            frequencies = [
                init_frequencies[init_values == value][0] if any(init_values == value) else 0
                for value in order
            ]  # sort selon l'ordre des modalités
            values = [
                init_values[init_values == value][0] if any(init_values == value) else value
                for value in order
            ]  # sort selon l'ordre des modalités
            rate_target = (
                y.groupby(df_feature).sum().reindex(order) / frequencies / len_df
            )  # target rate per modality
            underrepresented = [
                value for value, frequency in zip(values, frequencies) if frequency < min_freq
            ]  # valeur peu fréquentes

        worst, best = rate_target.idxmin(), rate_target.idxmax()

        # cas 2 : il n'existe pas de valeur sous-représentée
        return {"order": order, "worst": worst, "best": best}


def find_closest_modality(value, idx: int, freq_target: Series, min_freq: float) -> int:
    """[Qualitative] HELPER Finds the closest modality in terms of frequency and target rate"""

    # case 1: lowest modality
    if idx == 0:
        closest_modality = 1

    # case 2: biggest modality
    elif idx == len(freq_target) - 1:
        closest_modality = len(freq_target) - 2

    # case 3: not the lowwest nor the biggest modality
    else:
        # previous modality's volume and target rate
        _, previous_volume, previous_target = freq_target[idx - 1]

        # current modality's volume and target rate
        _, _, target = freq_target[idx]

        # next modality's volume and target rate
        _, next_volume, next_target = freq_target[idx + 1]

        # regroupement par volumétrie (préféré s'il n'y a pas beaucoup de volume)
        # case 1: la modalité suivante est inférieure à min_freq
        if next_volume < min_freq <= previous_volume:
            closest_modality = idx + 1

        # case 2: la modalité précédante est inférieure à min_freq
        elif previous_volume < min_freq <= next_volume:
            closest_modality = idx - 1

        # case 3: les deux modalités sont inférieures à min_freq
        elif (previous_volume < min_freq) & (next_volume < min_freq):
            # cas 1: la modalité précédante est inférieure à la modalité suivante
            if previous_volume < next_volume:
                closest_modality = idx - 1

            # cas 2: la modalité suivante est inférieure à la modalité précédante
            elif next_volume < previous_volume:
                closest_modality = idx + 1

            # cas 3: les deux modalités ont la même fréquence
            else:
                # cas1: la modalité précédante est plus prorche en taux de cible
                if abs(previous_target - target) <= abs(next_target - target):
                    closest_modality = idx - 1

                # cas2: la modalité précédente est plus proche en taux de cible
                else:
                    closest_modality = idx + 1

        # case 4: les deux modalités sont supérieures à min_freq
        else:
            # cas1: la modalité précédante est plus prorche en taux de cible
            if abs(previous_target - target) <= abs(next_target - target):
                closest_modality = idx - 1

            # cas2: la modalité précédante est plus prorche en taux de cible
            else:
                closest_modality = idx + 1

    # récupération de la valeur associée
    closest_value = freq_target[closest_modality][0]

    return closest_value