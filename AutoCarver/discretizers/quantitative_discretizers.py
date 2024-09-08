"""Tools to build simple buckets out of Quantitative features
for a binary classification model.
"""

from numpy import array, digitize, in1d, inf, isnan, linspace, quantile, sort, unique
from pandas import DataFrame, Series

from ..features import BaseFeature, Features, GroupedList, QuantitativeFeature, get_versions
from ..utils import extend_docstring
from .utils.base_discretizer import BaseDiscretizer
from .utils.multiprocessing import imap_unordered_function


class ContinuousDiscretizer(BaseDiscretizer):
    """Automatic discretizing of continuous and discrete features, building simple groups of
    quantiles of values.

    Quantile discretization creates a lot of modalities (for example: up to 100 modalities for
    ``min_freq=0.01``).
    Set ``min_freq`` with caution.

    The number of quantiles depends on overrepresented modalities and nans:

    * Values more frequent than ``min_freq`` are set as there own modalities.
    * Other values are cut in quantiles using ``numpy.quantile``.
    * The number of quantiles is set as ``(1-freq_frequent_modals)/(min_freq)``.
    * Nans are considered as a modality (and are taken into account in ``freq_frequent_modals``).
    """

    __name__ = "ContinuousDiscretizer"

    @extend_docstring(BaseDiscretizer.__init__)
    def __init__(
        self,
        quantitatives: list[QuantitativeFeature],
        min_freq: float,
        **kwargs: dict,
    ) -> None:
        """
        Parameters
        ----------
        features : list[str]
            List of column names of quantitative features (continuous and discrete) to be dicretized

        min_freq : float
            Minimum frequency per grouped modalities.

            * Features whose most frequent modality is less than ``min_freq`` won't be discretized.
            * Sets the number of quantiles in which to discretize the continuous features.
            * Sets the minimum frequency of a quantitative feature's modality.

            **Tip**: set between ``0.02`` (slower, less robust) and ``0.05`` (faster, more robust)
        """
        # initiating features
        features = Features(quantitatives=quantitatives, **kwargs)

        # Initiating BaseDiscretizer
        super().__init__(features=features, **kwargs)

        self.min_freq = min_freq
        self.q = round(1 / min_freq)  # number of quantiles

    @extend_docstring(BaseDiscretizer.fit)
    def fit(self, X: DataFrame, y: Series = None) -> None:  # pylint: disable=W0222
        self._verbose()  # verbose if requested

        # fitting each feature
        all_orders = imap_unordered_function(
            fit_feature,
            self.features.quantitatives,
            self.n_jobs,
            X=X[get_versions(self.features.quantitatives)],
            q=self.q,
        )

        # storing into the values_orders
        self.features.fit(X, y)
        self.features.update(dict(all_orders))

        # discretizing features based on each feature's values_order
        super().fit(X, y)

        return self


def fit_feature(feature: BaseFeature, X: DataFrame, q: float) -> tuple[str, GroupedList]:
    """Fits one feature"""
    # getting quantiles for specified feature
    quantiles = find_quantiles(X[feature.version].values, q=q)

    # Converting to a groupedlist
    order = GroupedList(quantiles + [inf])

    return feature.version, order


def find_quantiles(
    df_feature: array,
    q: int,
) -> list[float]:
    """Finds quantiles of a Series recursively.

    * Values more frequent than ``min_freq`` are set as there own modalities.
    * Other values are cut in quantiles using ``numpy.quantile``.
    * The number of quantiles is set as ``(1-freq_frequent_modals)/(min_freq)``.
    * Nans are considered as a modality (and are taken into account in ``freq_frequent_modals``).

    Parameters
    ----------
    df_feature : Series
        continuous feature
    q : int
        number of quantiles

    Returns
    -------
    list[float]
        list of quantiles for the feature
    """
    return list(
        sort(
            np_find_quantiles(
                df_feature[~isnan(df_feature)],  # getting rid of missing values
                q,
                len_df=len(df_feature),  # getting raw dataset size
                quantiles=[],  # initiating list of quantiles
            )
        )
    )


def np_find_quantiles(
    df_feature: array,
    q: int,
    len_df: int = None,
    quantiles: list[float] = None,
) -> list[float]:
    """Finds quantiles of a Series recursively.

    * Values more frequent than ``min_freq`` are set as there own modalities.
    * Other values are cut in quantiles using ``numpy.quantile``.
    * The number of quantiles is set as ``(1-freq_frequent_modals)/(min_freq)``.
    * Nans are considered as a modality (and are taken into account in ``freq_frequent_modals``).

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
    # case 1: no observation, all values have been attributed there corresponding modality
    if df_feature.shape[0] == 0:
        return quantiles

    # frequencies per known value
    values, frequencies = unique(df_feature, return_counts=True)

    # case 3 : there are no missing values
    # case 3.1 : there is an over-populated value
    if any(frequencies >= len_df / q):
        # identifying over-represented modality
        frequent_values = values[frequencies >= len_df / q]

        # computing quantiles on smaller and greater values
        sub_indices = digitize(df_feature, frequent_values, right=False)
        for i in range(0, len(frequent_values) + 1):
            quantiles += np_find_quantiles(
                df_feature[(sub_indices == i) & (~in1d(df_feature, frequent_values))], q, len_df, []
            )

        # adding over-represented modality to the list of quantiles
        return quantiles + list(frequent_values)

    # case 3.2 : there is no over-populated value
    # reducing the size of quantiles by frequencies of over-represented modalities
    new_q = round(len(df_feature) / len_df * q)

    # cutting values into quantiles if there are enough of them
    if new_q > 1:
        quantiles += list(
            quantile(
                df_feature,
                linspace(0, 1, new_q + 1)[1:-1],
                method="lower",
            )
        )

    # not enough values observed, grouping all remaining values into a quantile
    else:
        quantiles += [max(df_feature)]

    return quantiles
