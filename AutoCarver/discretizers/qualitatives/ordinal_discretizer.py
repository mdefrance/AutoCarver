"""Tools to build simple buckets out of Qualitative features
for a binary classification model.
"""

from typing import Union

import numpy as np
from numpy import argmin
from pandas import DataFrame, Series, isna

from ...features import GroupedList, OrdinalFeature
from ...utils import extend_docstring
from ..utils.base_discretizer import BaseDiscretizer


class OrdinalDiscretizer(BaseDiscretizer):
    """Automatic discretization of ordinal features, grouping less frequent modalities with the
    closest modlity in target rate or by frequency.

    NaNs are left untouched.

    Only use for qualitative ordinal features.

    Fisrt fits :ref:`StringDiscretizer` if neccesary.
    """

    __name__ = "OrdinalDiscretizer"

    @extend_docstring(BaseDiscretizer.__init__)
    def __init__(
        self,
        ordinals: list[OrdinalFeature],
        min_freq: float,
        **kwargs: dict,
    ):
        """
        Parameters
        ----------
        ordinal_features : list[str]
            List of column names of ordinal features to be discretized. For those features a list
            of values has to be provided in the ``values_orders`` dict.

        min_freq : float
            Minimum frequency per grouped modalities.

            * Features whose most frequent modality is < than ``min_freq`` won't be discretized.
            * Sets the number of quantiles in which to discretize the continuous features.
            * Sets the minimum frequency of a quantitative feature's modality.

            **Tip**: should be set between ``0.02`` (slower, preciser, less robust) and ``0.05``
            (faster, more robust)
        """
        # Initiating BaseDiscretizer
        super().__init__(ordinals, **dict(kwargs, min_freq=min_freq))

    def _prepare_data(self, X: DataFrame, y: Series) -> DataFrame:  # pylint: disable=W0222
        """Validates format and content of X and y.

        Parameters
        ----------
        X : DataFrame
            Dataset used to discretize. Needs to have columns has specified in
            ``OrdinalDiscretizer.features``.

        y : Series
            Binary target feature.

        Returns
        -------
        DataFrame
            A formatted copy of X
        """
        # checking for binary target and copying X
        x_copy = super()._prepare_data(X, y)

        # fitting features
        self.features.fit(x_copy, y)

        # filling up nans for features that have some
        x_copy = self.features.fillna(x_copy)

        return x_copy

    @extend_docstring(BaseDiscretizer.fit)
    def fit(self, X: DataFrame, y: Series) -> None:  # pylint: disable=W0222
        # checking values orders
        x_copy = self._prepare_data(X, y)
        self._verbose()  # verbose if requested

        # grouping rare modalities for each feature
        common_modalities = {
            feature.version: find_common_modalities(
                x_copy[feature.version],
                y,
                min_freq=self.min_freq,
                order=[label for label in feature.labels if label != feature.nan],
            )
            for feature in self.features
        }

        # updating features accordingly
        self.features.update(common_modalities, convert_labels=True)

        # discretizing features based on each feature's values_order
        super().fit(x_copy, y)

        return self


def find_common_modalities(
    df_feature: Series,
    y: Series,
    min_freq: float,
    order: list[str],
) -> dict[str, Union[str, float]]:
    """finds common modalities of a ordinal feature"""
    # making a groupedlist of ordered labels
    order = GroupedList(order)

    # total size
    len_df = len(df_feature)

    # computing frequencies and target rate of each modality
    not_nans = ~isna(df_feature)  # pylint: disable=E1130
    stats = np.vstack(
        (
            df_feature[not_nans]
            .value_counts(dropna=False, normalize=False)
            .reindex(order, fill_value=0)
            .values,
            y[not_nans].groupby(df_feature[not_nans]).sum().reindex(order).values,
        )
    )

    # case 2.1: there are underrepresented modalities/values
    while any(stats[0, :] / len_df < min_freq) & (stats.shape[1] > 1):
        # identifying the underrepresented value
        discarded_idx = argmin(stats[0, :])

        # choosing amongst previous and next modality (by volume and target rate)
        kept_idx = find_closest_modality(
            discarded_idx,
            stats[0, :] / len_df,
            stats[1, :] / stats[0, :],
            min_freq,
        )

        # removing the value from the initial ordering
        order.group(order[discarded_idx], order[kept_idx])

        # adding up grouped frequencies and target counts
        stats[:, kept_idx] += stats[:, discarded_idx]

        # removing discarded modality
        stats = stats[:, np.arange(stats.shape[1]) != discarded_idx]

    # case 2.2 : no underrepresented value
    return order


def find_closest_modality(
    idx: int, frequencies: np.array, target_rates: np.array, min_freq: float
) -> int:
    """Finds the closest modality in terms of frequency and target rate"""
    # case 1: lowest ranked modality
    if idx == 0:
        return 1

    # case 2: highest ranked modality
    if idx == frequencies.shape[0] - 1:
        return idx - 1

    # case 3: modality ranked in the middle
    # modalities frequencies and target rates
    (
        (previous_freq, current_freq, next_freq),
        (previous_target, current_target, next_target),
        idx_closest_modality,
    ) = (
        frequencies[idx - 1 : idx + 2],
        target_rates[idx - 1 : idx + 2],
        idx - 1,
    )  # by default previous value

    # cases when idx + 1 is the closest
    if (
        # case 1: next modality is the only below min_freq -> underrepresented
        (next_freq < min_freq <= previous_freq)
        or (
            (
                # case 2: both are below min_freq
                ((next_freq < min_freq) and (previous_freq < min_freq))
                or
                # case 3: both are above min_freq -> representative modalities
                ((next_freq >= min_freq) and (previous_freq >= min_freq))
            )
            and (
                # no target to differentiate -> least frequent modality
                ((current_freq == 0) and (next_freq < previous_freq))
                or
                # differentiate by target -> closest target rate
                (
                    (current_target > 0)
                    and (abs(previous_target - current_target) > abs(next_target - current_target))
                )
            )
        )
    ):
        # identifying smallest modality in terms of frequency
        idx_closest_modality = idx + 1

    # finding the closest value
    return idx_closest_modality
