from typing import Any, Dict, List

from numpy import (
    array,
    inf,
    linspace,
    nan,
    quantile,
    select,
)
from pandas import DataFrame, Series, isna, notna, unique
from sklearn.base import BaseEstimator, TransformerMixin

from .BaseDiscretizers import GroupedList




class QuantileDiscretizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        features: List[str],
        q: int,
        *,
        values_orders: Dict[str, Any] = {},
        copy: bool = False,
        verbose: bool = False,
    ) -> None:
        """Discretizes quantitative features into groups of q quantiles"""

        self.features = features[:]
        self.q = q
        self.values_orders = {
            k: GroupedList(v) for k, v in values_orders.items()
        }
        self.quantiles: Dict[str, Any] = {}
        self.copy = copy
        self.verbose = verbose

    def fit(self, X: DataFrame, y: Series = None) -> None:
        # computing quantiles for the feature
        self.quantiles = X[self.features].apply(
            find_quantiles, q=self.q, axis=0
        )

        # case when only one feature is discretized
        if len(self.features) == 1:
            self.quantiles = {
                self.features[0]: list(
                    self.quantiles.get(self.features[0]).values
                )
            }

        # building string of values to be displayed
        values: List[str] = []
        for n, feature in enumerate(self.features):
            # verbose
            if self.verbose:
                print(
                    f" - [QuantileDiscretizer] Fit {feature} ({n+1}/{len(self.features)})"
                )

            # quantiles as strings
            feature_quantiles = self.quantiles.get(feature)
            str_values = ["<= " + q for q in format_list(feature_quantiles)]

            # case when there is only one value
            if len(feature_quantiles) == 0:
                str_values = ["non-nan"]

            # last quantile is between the last value and inf
            else:
                str_values = str_values + [
                    str_values[-1].replace("<= ", ">  ")
                ]
            values += [str_values]

        # adding inf for the last quantiles
        self.quantiles = {f: q + [inf] for f, q in self.quantiles.items()}

        # creating the values orders based on quantiles
        self.values_orders.update(
            {
                feature: GroupedList(str_values)
                for feature, str_values in zip(self.features, values)
            }
        )

        return self

    def transform(self, X: DataFrame, y: Series = None) -> DataFrame:
        # copying dataset if requested
        Xc = X
        if self.copy:
            Xc = X.copy()

        # iterating over each feature
        for n, feature in enumerate(self.features):
            # verbose if requested
            if self.verbose:
                print(
                    f" - [QuantileDiscretizer] Transform {feature} ({n+1}/{len(self.features)})"
                )

            nans = isna(Xc[feature])  # keeping track of nans

            # grouping values inside quantiles
            to_input = [
                Xc[feature] <= q for q in self.quantiles.get(feature)
            ]  # values that will be imputed
            values = [
                [v] * len(X) for v in self.values_orders.get(feature)
            ]  # new values to imput
            Xc[feature] = select(
                to_input, values, default=Xc[feature]
            )  # grouping modalities

            # adding back nans
            if any(nans):
                Xc.loc[nans, feature] = nan

        return Xc


def find_quantiles(
    df_feature: Series,
    q: int,
    len_df: int = None,
    quantiles: List[float] = None,
) -> List[float]:
    """[Quantitative] Découpage en quantile de la feature.

    Fonction récursive : on prend l'échantillon et on cherche des valeurs sur-représentées
    Si la valeur existe on la met dans une classe et on cherche à gauche et à droite de celle-ci, d'autres variables sur-représentées
    Une fois il n'y a plus de valeurs qui soient sur-représentées,
    on fait un découpage classique avec qcut en multipliant le nombre de classes souhaité par le pourcentage de valeurs restantes sur le total

    """

    # initialisation de la taille total du dataframe
    if len_df is None:
        len_df = len(df_feature)

    # initialisation de la liste des quantiles
    if quantiles is None:
        quantiles = []

    # calcul du nombre d'occurences de chaque valeur
    frequencies = (
        df_feature.value_counts(dropna=False, normalize=False).drop(
            nan, errors="ignore"
        )
        / len_df
    )  # dropping nans to keep them anyways
    values, frequencies = array(frequencies.index), array(frequencies.values)

    # cas 1 : il n'y a pas d'observation dans le dataframe
    if len(df_feature) == 0:
        return quantiles

    # cas 2 : il y a des valeurs manquantes NaN
    elif any(isna(df_feature)):
        return find_quantiles(
            df_feature[notna(df_feature)],
            q,
            len_df=len_df,
            quantiles=quantiles,
        )

    # cas 2 : il n'y a que des valeurs dans le dataframe (pas de NaN)
    else:
        # cas 1 : il existe une valeur sur-représentée
        if any(frequencies > 1 / q):
            # identification de la valeur sur-représentée
            frequent_value = values[frequencies.argmax()]

            # ajout de la valeur fréquente à la liste des quantiles
            quantiles += [frequent_value]

            # calcul des quantiles pour les parties inférieures et supérieures
            quantiles_inf = find_quantiles(
                df_feature[df_feature < frequent_value], q, len_df=len_df
            )
            quantiles_sup = find_quantiles(
                df_feature[df_feature > frequent_value], q, len_df=len_df
            )

            return quantiles_inf + quantiles + quantiles_sup

        # cas 2 : il n'existe pas de valeur sur-représentée
        else:
            # nouveau nombre de quantile en prenant en compte les classes déjà constituées
            new_q = max(round(len(df_feature) / len_df * q), 1)

            # calcul des quantiles sur le dataframe
            if new_q > 1:
                quantiles += list(
                    quantile(
                        df_feature.values,
                        linspace(0, 1, new_q + 1)[1:-1],
                        interpolation="lower",
                    )
                )

            # case when there are no enough observations to compute quantiles
            else:
                quantiles += [max(df_feature.values)]

            return quantiles


def format_list(a_list: List[float]) -> List[str]:
    """Formats a list of floats to a list of unique rounded strings of floats"""

    # finding the closest power of thousands for each element
    closest_powers = [
        next((k for k in range(-3, 4) if abs(elt) / 100 // 1000 ** (k) < 10))
        for elt in a_list
    ]

    # rounding elements to the closest power of thousands
    rounded_to_powers = [
        elt / 1000 ** (k) for elt, k in zip(a_list, closest_powers)
    ]

    # computing optimal decimal per unique power of thousands
    optimal_decimals: Dict[str, int] = {}
    for power in unique(
        closest_powers
    ):  # iterating over each power of thousands found
        # filtering on the specific power of thousands
        sub_array = array(
            [
                elt
                for elt, p in zip(rounded_to_powers, closest_powers)
                if power == p
            ]
        )

        # number of distinct values
        n_uniques = sub_array.shape[0]

        # computing the first rounding decimal that allows for distinction of
        # each values when rounded, by default None (no rounding)
        optimal_decimal = next(
            (
                k
                for k in range(1, 10)
                if len(unique(sub_array.round(k))) == n_uniques
            ),
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
        f"{elt: 3.3f}{suffixes[power]}"
        for elt, power in zip(rounded_list, closest_powers)
    ]

    # keeping zeros
    formatted_list = [
        rounded if raw != 0 else f"{raw: 3.3f}"
        for rounded, raw in zip(formatted_list, a_list)
    ]

    return formatted_list
