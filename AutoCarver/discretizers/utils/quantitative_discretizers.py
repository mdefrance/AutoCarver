"""Tools to build simple buckets out of Quantitative features
for a binary classification model.
"""

from typing import Any

from numpy import array, inf, linspace, nan, quantile
from pandas import DataFrame, Series, isna, notna

from .base_discretizers import BaseDiscretizer, applied_to_dict_list
from .grouped_list import GroupedList


class QuantileDiscretizer(BaseDiscretizer):
    """Automatic discretizing of continuous and discrete features, building simple groups of quantiles of values.

    Quantile discretization creates a lot of modalities (for example: 100 modalities for ``min_freq=0.01``).
    Set ``min_freq`` with caution.

    The number of quantiles depends on overrepresented modalities and nans:

    * Values more frequent than ``min_freq`` are set as there own modalities.
    * Other values are cut in quantiles using ``numpy.quantile``.
    * The number of quantiles is set as ``(1-freq_of_frequent_modalities)/(min_freq)``.
    * Nans are considered as a modality (and are taken into account in ``freq_of_frequent_modalities``).
    """

    def __init__(
        self,
        quantitative_features: list[str],
        min_freq: float,
        *,
        values_orders: dict[str, Any] = None,
        copy: bool = False,
        verbose: bool = False,
        str_nan: str = "__NAN__",
    ) -> None:
        """Initiates a QuantileDiscretizer.

        Parameters
        ----------
        quantitative_features : list[str]
            List of column names of quantitative features (continuous and discrete) to be dicretized

        min_freq : float
            Minimum frequency per grouped modalities.

            * Features whose most frequent modality is less frequent than `min_freq` will not be discretized.
            * Sets the number of quantiles in which to discretize the continuous features.
            * Sets the minimum frequency of a quantitative feature's modality.

            **Tip**: should be set between 0.02 (slower, preciser, less robust) and 0.05 (faster, more robust)

        values_orders : dict[str, GroupedList], optional
            Dict of feature's column names and there associated ordering.
            If lists are passed, a GroupedList will automatically be initiated, by default ``None``

        copy : bool, optional
            If ``True``, feature processing at transform is applied to a copy of the provided DataFrame, by default ``False``

        verbose : bool, optional
            If ``True``, prints raw Discretizers Fit and Transform steps, by default ``False``

        str_nan : str, optional
            String representation to input ``numpy.nan``. If ``dropna=False``, ``numpy.nan`` will be left unchanged, by default ``"__NAN__"``
        """
        # Initiating BaseDiscretizer
        super().__init__(
            features=quantitative_features,
            values_orders=values_orders,
            input_dtypes="float",
            output_dtype="str",
            str_nan=str_nan,
            copy=copy,
            verbose=verbose,
        )

        self.min_freq = min_freq
        self.q = round(1 / min_freq)  # number of quantiles

    def fit(self, X: DataFrame, y: Series = None) -> None:
        """Finds simple buckets of modalities of X.

        Parameters
        ----------
        X : DataFrame
            Dataset used to discretize. Needs to have columns has specified in ``QuantileDiscretizer.features``.

        y : Series
            Binary target feature, not used, by default None
        """
        if self.verbose:  # verbose if requested
            print(f" - [QuantileDiscretizer] Fit {str(self.quantitative_features)}")

        # computing quantiles for the feature
        quantiles = applied_to_dict_list(
            X[self.quantitative_features].apply(find_quantiles, q=self.q, axis=0)
        )

        # storing ordering
        for feature in self.quantitative_features:
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
    quantiles: list[float] = None,
) -> list[float]:
    """Finds quantiles of a Series recursively.

    * Values more frequent than ``min_freq`` are set as there own modalities.
    * Other values are cut in quantiles using ``numpy.quantile``.
    * The number of quantiles is set as ``(1-freq_of_frequent_modalities)/(min_freq)``.
    * Nans are considered as a modality (and are taken into account in ``freq_of_frequent_modalities``).

    Parameters
    ----------
    df_feature : Series
        _description_
    q : int
        _description_
    len_df : int, optional
        _description_, by default None
    quantiles : list[float], optional
        _description_, by default None

    Returns
    -------
    list[float]
        _description_
    """
    # getting dataset size
    if len_df is None:
        len_df = len(df_feature)

    # initiating liust of quantiles
    if quantiles is None:
        quantiles = []

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

    # frequencies per known value
    frequencies = (
        df_feature.value_counts(dropna=False, normalize=False).drop(nan, errors="ignore") / len_df
    )
    values, frequencies = array(frequencies.index), array(frequencies.values)

    # case 3 : there are no missing values
    # case 3.1 : there is an over-populated value
    if any(frequencies > 1 / q):
        # identifying over-represented modality
        frequent_value = values[frequencies.argmax()]

        # adding over-represented modality to the list of quantiles
        quantiles += [frequent_value]

        # computing quantiles on smaller and greater values
        quantiles_inf = find_quantiles(df_feature[df_feature < frequent_value], q, len_df=len_df)
        quantiles_sup = find_quantiles(df_feature[df_feature > frequent_value], q, len_df=len_df)

        return quantiles_inf + quantiles + quantiles_sup

    # case 3.2 : there is no over-populated value
    # reducing the size of quantiles by frequencies of over-represented modalities
    new_q = max(round(len(df_feature) / len_df * q), 1)

    # cutting values into quantiles if there are enough of them
    if new_q > 1:
        quantiles += list(
            quantile(
                df_feature.values,
                linspace(0, 1, new_q + 1)[1:-1],
                method="lower",
            )
        )

    # not enough values observed, grouping all remaining values into a quantile
    else:
        quantiles += [max(df_feature.values)]

    return quantiles
