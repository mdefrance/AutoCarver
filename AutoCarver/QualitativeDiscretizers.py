from typing import Any, Dict, List

from numpy import (
    argmin,
    inf,
    nan,
    select,
)
from pandas import DataFrame, Series, isna, notna
from sklearn.base import BaseEstimator, TransformerMixin

from .BaseDiscretizers import GroupedList




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
        self.known_values = list(
            set([v for o in self.chained_orders for v in o.values()])
        )
        self.values_orders = {
            f: GroupedList(self.known_values[:] + [self.str_nan])
            for f in self.features
        }

    def fit(self, X: DataFrame, y: Series = None) -> None:
        # filling nans
        Xc = X[self.features].fillna(self.str_nan)

        # iterating over each feature
        for n, feature in enumerate(self.features):
            # verbose if requested
            if self.verbose:
                print(
                    f" - [ChainedDiscretizer] Fit {feature} ({n+1}/{len(self.features)})"
                )

            # computing frequencies of each modality
            frequencies = Xc[feature].value_counts(normalize=True)
            values, frequencies = frequencies.index, frequencies.values

            # checking for unknown values (values to present in an order of self.chained_orders)
            missing = [
                value
                for value in values
                if notna(value) and (value not in self.known_values)
            ]

            # converting unknown values to NaN
            if self.remove_unknown & (len(missing) > 0):
                # alerting user
                print(
                    f"Order for feature '{feature}' was not provided for values: {missing}, these values will be converted to '{self.str_nan}' (policy remove_unknown=True)"
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
                to_discard = [
                    value for value in values_order if value not in to_keep
                ]

                # values to group into discarded values
                value_to_group = [
                    order.get_group(value) for value in to_discard
                ]

                # values of the series to input
                df_to_input = [
                    Xc[feature] == discarded for discarded in to_discard
                ]  # identifying observation to input

                # inputing non frequent values
                Xc[feature] = select(
                    df_to_input, value_to_group, default=Xc[feature]
                )

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
        """Groups a qualitative features' values less frequent than min_freq into a default_value"""

        self.features = features[:]
        self.min_freq = min_freq
        self.values_orders = {
            k: GroupedList(v) for k, v in values_orders.items()
        }
        self.default_value = default_value[:]
        self.str_nan = str_nan[:]
        self.copy = copy
        self.verbose = verbose

    def fit(self, X: DataFrame, y: Series) -> None:
        # filling up NaNs
        Xc = X[self.features].fillna(self.str_nan)

        # computing frequencies of each modality
        frequencies = Xc.apply(value_counts, normalize=True, axis=0)

        # computing ordered target rate per modality for ordering
        target_rates = Xc.apply(target_rate, y=y, ascending=True, axis=0)

        # attributing orders to each feature
        self.values_orders.update(
            {f: GroupedList(list(target_rates[f])) for f in self.features}
        )

        # number of unique modality per feature
        nuniques = Xc.apply(nunique, axis=0)

        # identifying modalities which are the most common
        self.to_keep: Dict[
            str, Any
        ] = {}  # dict of features and corresponding kept modalities

        # iterating over each feature
        for feature in self.features:
            # checking for binary features
            if nuniques[feature] > 2:
                kept = [
                    val
                    for val, freq in frequencies[feature].items()
                    if freq >= self.min_freq
                ]

            # keeping all modalities of binary features
            else:
                kept = [val for val, freq in frequencies[feature].items()]

            self.to_keep.update({feature: kept})

        # grouping rare modalities
        for n, feature in enumerate(self.features):
            # printing verbose
            if self.verbose:
                print(
                    f" - [DefaultDiscretizer] Fit {feature} ({n+1}/{len(self.features)})"
                )

            # identifying values to discard (rare modalities)
            to_discard = [
                value
                for value in self.values_orders[feature]
                if value not in self.to_keep[feature]
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
                print(
                    f" - [DefaultDiscretizer] Transform {feature} ({n+1}/{len(self.features)})"
                )

            # grouping modalities
            Xc[feature] = select(
                [~Xc[feature].isin(self.to_keep[feature])],
                [self.default_value],
                default=Xc[feature],
            )

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
        init_frequencies = (
            df_feature.value_counts(dropna=False, normalize=False) / len_df
        )
        init_values, init_frequencies = (
            init_frequencies.index,
            init_frequencies.values,
        )

        # ordering
        frequencies = [
            init_frequencies[init_values == value][0]
            if any(init_values == value)
            else 0
            for value in order
        ]  # sort selon l'ordre des modalités
        values = [
            init_values[init_values == value][0]
            if any(init_values == value)
            else value
            for value in order
        ]  # sort selon l'ordre des modalités
        target_rate = (
            y.groupby(df_feature).sum().reindex(order) / frequencies / len_df
        )  # target rate per modality
        underrepresented = [
            value
            for value, frequency in zip(values, frequencies)
            if frequency < min_freq
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
                list(zip(order, frequencies, target_rate)),
                min_freq,
            )

            # removing the value from the initial ordering
            order.group(discarded, kept)

            # ordering
            frequencies = [
                init_frequencies[init_values == value][0]
                if any(init_values == value)
                else 0
                for value in order
            ]  # sort selon l'ordre des modalités
            values = [
                init_values[init_values == value][0]
                if any(init_values == value)
                else value
                for value in order
            ]  # sort selon l'ordre des modalités
            target_rate = (
                y.groupby(df_feature).sum().reindex(order)
                / frequencies
                / len_df
            )  # target rate per modality
            underrepresented = [
                value
                for value, frequency in zip(values, frequencies)
                if frequency < min_freq
            ]  # valeur peu fréquentes

        worst, best = target_rate.idxmin(), target_rate.idxmax()

        # cas 2 : il n'existe pas de valeur sous-représentée
        return {"order": order, "worst": worst, "best": best}


def find_closest_modality(
    value, idx: int, freq_target: Series, min_freq: float
) -> int:
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
        previous_value, previous_volume, previous_target = freq_target[idx - 1]

        # current modality's volume and target rate
        _, volume, target = freq_target[idx]

        # next modality's volume and target rate
        next_value, next_volume, next_target = freq_target[idx + 1]

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

