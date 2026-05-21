"""Tools to build simple buckets out of Quantitative features
for a binary classification model.
"""

from typing import Self

import numpy as np
import pandas as pd

from AutoCarver.discretizers.utils.base_discretizer import BaseDiscretizer, DiscretizerConfig
from AutoCarver.discretizers.utils.multiprocessing import imap_unordered_function
from AutoCarver.features import GroupedList, QuantitativeFeature, get_versions
from AutoCarver.utils import extend_docstring


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

    @extend_docstring(BaseDiscretizer.__init__, append=False, exclude=["features"])
    def __init__(
        self,
        quantitatives: list[QuantitativeFeature],
        min_freq: float,
        *,
        config: DiscretizerConfig | None = None,
    ) -> None:
        """
        Parameters
        ----------

        quantitatives : list[QuantitativeFeature]
            Quantitative features to process
        """
        super().__init__(features=quantitatives, min_freq=min_freq, config=config)

    @property
    def q(self) -> int:
        """Number of quantiles to discretize the continuous features."""
        if self.min_freq is None:
            raise ValueError(f"[{self.__name__}] min_freq must be set to compute q")
        return round(1 / self.min_freq)

    @extend_docstring(BaseDiscretizer.fit)
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:  # pylint: disable=W0222
        self._log_if_verbose()  # verbose if requested

        # fitting each feature
        all_orders = imap_unordered_function(
            fit_feature,
            self.features.quantitatives,
            self.config.n_jobs,
            X=X[get_versions(self.features.quantitatives)],
            q=self.q,
        )

        # storing into the values_orders
        self.features.fit(X, y)
        self.features.update(dict(all_orders))

        # discretizing features based on each feature's values_order
        super().fit(X, y)

        return self


def fit_feature(feature: QuantitativeFeature, X: pd.DataFrame, q: int) -> tuple[str, GroupedList]:
    """Fits one feature"""

    # getting quantiles for specified feature
    quantiles = find_quantiles(X[feature.version].values, q=q)

    # Converting to a groupedlist
    order = GroupedList(quantiles + [np.inf])

    return feature.version, order


def find_quantiles(
    df_feature: np.ndarray,
    q: int,
) -> list[float]:
    """Finds quantiles of a pd.Series in a single sort pass.

    * Values more frequent than ``min_freq`` are set as there own modalities.
    * Other values are cut in quantiles using ``numpy.quantile``.
    * The number of quantiles is set as ``(1-freq_frequent_modals)/(min_freq)``.
    * Nans are considered as a modality (and are taken into account in ``freq_frequent_modals``).

    Parameters
    ----------
    df_feature : pd.Series
        continuous feature
    q : int
        number of quantiles

    Returns
    -------
    list[float]
        list of quantiles for the feature
    """
    initial_len_df = len(df_feature)
    cleaned = df_feature[~np.isnan(df_feature)]
    if cleaned.shape[0] == 0:
        return []

    # one O(N log N) sort; all subsequent work is O(N) at most
    sorted_values = np.sort(cleaned)
    unique_values, counts = np.unique(sorted_values, return_counts=True)

    # over-represented modalities: same threshold as the recursive version
    threshold = initial_len_df / q
    is_frequent = counts >= threshold
    frequent_values = unique_values[is_frequent]

    # cum_counts[i] = index in sorted_values just past the last occurrence of unique_values[i]
    cum_counts = np.cumsum(counts)
    starts = np.concatenate(([0], cum_counts[:-1]))
    ends = cum_counts

    # sub-segments are the contiguous runs of sorted_values between frequent values
    freq_idx = np.flatnonzero(is_frequent)
    if len(freq_idx) == 0:
        segment_bounds = [(0, len(sorted_values))]
    else:
        segment_bounds = [(0, int(starts[freq_idx[0]]))]
        for i in range(len(freq_idx) - 1):
            segment_bounds.append((int(ends[freq_idx[i]]), int(starts[freq_idx[i + 1]])))
        segment_bounds.append((int(ends[freq_idx[-1]]), len(sorted_values)))

    quantiles: list[float] = []
    for lo, hi in segment_bounds:
        seg_len = hi - lo
        if seg_len == 0:
            continue
        new_q = round(seg_len / initial_len_df * q)
        if new_q < 2:
            # not enough remaining values: mirror compute_quantiles' fallback to the segment max
            quantiles.append(sorted_values[hi - 1].item())
            continue
        # np.quantile(method='lower') on a sorted array == sorted[floor(p * (N-1))]
        probs = np.linspace(0, 1, new_q + 1)[1:-1]
        indices = lo + np.floor(probs * (seg_len - 1)).astype(np.intp)
        quantiles.extend(sorted_values[indices].tolist())

    quantiles.extend(frequent_values.tolist())
    quantiles.sort()
    return quantiles


def np_find_quantiles(
    df_feature: np.ndarray,
    q: int,
    initial_len_df: int,
    quantiles: list[float],
) -> list[float]:
    """Finds quantiles of a Series recursively.

    * Values more frequent than ``min_freq`` are set as there own modalities.
    * Other values are cut in quantiles using ``numpy.quantile``.
    * The number of quantiles is set as ``(1-freq_frequent_modals)/(min_freq)``.
    * Nans are considered as a modality (and are taken into account in ``freq_frequent_modals``).

    Parameters
    ----------
    df_feature : pd.Series
        _description_
    q : int
        _description_
    initial_len_df : int, optional
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
    values, frequencies = np.unique(df_feature, return_counts=True)

    # case 2 : there is an over-represented value
    if any(frequencies >= initial_len_df / q):
        # identifying over-represented modality
        frequent_values = values[frequencies >= initial_len_df / q]

        # computing quantiles on smaller and greater values
        sub_indices = np.digitize(df_feature, frequent_values, right=False)
        for i in range(0, len(frequent_values) + 1):
            quantiles += np_find_quantiles(
                df_feature[(sub_indices == i) & (~np.isin(df_feature, frequent_values))],
                q,
                initial_len_df,
                [],
            )

        # adding over-represented modality to the list of quantiles
        return quantiles + list(frequent_values)

    # case 3 : there is no over-represented value -> computing quantiles
    quantiles += compute_quantiles(df_feature, q, initial_len_df)

    return quantiles


def compute_quantiles(df_feature: np.ndarray, q: int, initial_len_df: int) -> list[float]:
    """Computes quantiles of a Series.

    - q should not be larger than len(df_feature)
    - df_feature should not contain any missing values
    - df_feature should not contain any over-represented modality
    - df_feature should not be empty
    """

    # getting quantiles needed
    quantiles_needed = get_remaining_quantiles(len(df_feature), initial_len_df, q)

    # cutting values into quantiles if there are enough values remaining
    if len(quantiles_needed) > 0:
        return list(np.quantile(df_feature, quantiles_needed, method="lower"))

    # -> not enough values remaining, grouping all remaining values into one quantile
    # returning the maximum value remaining, because it will be grouped within
    # QuantitativeDiscretizer when using OrdinalDiscretizer
    return [max(df_feature)]


def get_remaining_quantiles(remaining_len_df: int, initial_len_df: int, q: int) -> np.ndarray:
    """Computes list of indices of quantiles needed."""

    # updating number of quantiles taking into account identified over-represented modalities
    new_q = round(remaining_len_df / initial_len_df * q)

    # list of quantiles needed
    return np.linspace(0, 1, new_q + 1)[1:-1]
