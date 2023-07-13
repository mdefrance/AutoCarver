"""Tools to build simple buckets out of Quantitative features
for a binary classification model.
"""

from typing import Any, Dict, List

from numpy import array, inf, linspace, nan, quantile
from pandas import DataFrame, Series, isna, notna

from .base_discretizers import GroupedList, GroupedListDiscretizer, applied_to_dict_list


class QuantileDiscretizer(GroupedListDiscretizer):
    """Builds per-feature buckets of quantiles"""

    def __init__(
        self,
        features: list[str],
        min_freq: float,
        *,
        values_orders: dict[str, Any] = None,
        str_nan: str = "__NAN__",
        copy: bool = False,
        verbose: bool = False,
    ) -> None:
        """Discretizes quantitative features into groups of q quantiles

        Parameters
        ----------
        features : List[str]
            _description_
        min_freq : float
            _description_
        values_orders : Dict[str, Any], optional
            _description_, by default None
        str_nan : str, optional
            _description_, by default '__NAN__'
        copy : bool, optional
            _description_, by default False
        verbose : bool, optional
            _description_, by default False
        """
        # Initiating GroupedListDiscretizer
        super().__init__(
            features=features,
            values_orders=values_orders,
            input_dtypes='float',
            output_dtype='str',
            str_nan=str_nan,
            copy=copy,
        )

        self.min_freq = min_freq
        self.q = round(1 / min_freq)  # number of quantiles
        self.verbose = verbose

    def fit(self, X: DataFrame, y: Series = None) -> None:
        """_summary_

        Parameters
        ----------
        X : DataFrame
            _description_
        y : Series, optional
            _description_, by default None
        """
        if self.verbose:  # verbose if requested
            print(f" - [QuantileDiscretizer] Fit {', '.join(self.features)}")

        # computing quantiles for the feature
        quantiles = applied_to_dict_list(X[self.features].apply(find_quantiles, q=self.q, axis=0))

        # storing ordering
        for feature in self.features:
            # Converting to a groupedlist
            order = GroupedList(quantiles[feature] + [inf])

            # adding NANs if ther are any
            if any(isna(X[feature])):
                order.append(self.str_nan)

            # storing into the values_orders
            self.values_orders.update({feature: order})

        # discretizing features based on each feature's values_order
        super().fit(X, y)

        return self


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
        df_feature.value_counts(dropna=False, normalize=False).drop(nan, errors="ignore") / len_df
    )  # dropping nans to keep them anyways
    values, frequencies = array(frequencies.index), array(frequencies.values)

    # case 1: no observation, all values have been attributed there corresponding modality
    if len(df_feature) == 0:
        return quantiles

    # case 2: there are missing values
    if any(isna(df_feature)):
        return find_quantiles(
            df_feature[notna(df_feature)],
            q,
            len_df=len_df,
            quantiles=quantiles,
        )

    # case 3 : there are no missing values
    # case 3.1 : there is an over-populated value
    if any(frequencies > 1 / q):
        # identification de la valeur sur-représentée
        frequent_value = values[frequencies.argmax()]

        # ajout de la valeur fréquente à la liste des quantiles
        quantiles += [frequent_value]

        # calcul des quantiles pour les parties inférieures et supérieures
        quantiles_inf = find_quantiles(df_feature[df_feature < frequent_value], q, len_df=len_df)
        quantiles_sup = find_quantiles(df_feature[df_feature > frequent_value], q, len_df=len_df)

        return quantiles_inf + quantiles + quantiles_sup

    # case 3.2 : there is no over-populated value
    # nouveau nombre de quantile en prenant en compte les classes déjà constituées
    new_q = max(round(len(df_feature) / len_df * q), 1)

    # calcul des quantiles sur le dataframe
    if new_q > 1:
        quantiles += list(
            quantile(
                df_feature.values,
                linspace(0, 1, new_q + 1)[1:-1],
                method="lower",
            )
        )

    # case when there are no enough observations to compute quantiles
    else:
        quantiles += [max(df_feature.values)]

    return quantiles
