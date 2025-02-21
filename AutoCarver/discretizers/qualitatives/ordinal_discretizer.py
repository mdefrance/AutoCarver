"""Tools to build simple buckets out of Qualitative features
for a binary classification model.
"""

from numpy import arange, argmin, array, nan_to_num, vstack
from pandas import DataFrame, Series, notna

from ...features import GroupedList, OrdinalFeature
from ...utils import extend_docstring
from ..utils.base_discretizer import BaseDiscretizer, Sample


class OrdinalDiscretizer(BaseDiscretizer):
    """Automatic discretization of ordinal features, grouping less frequent modalities with the
    closest modlity in target rate or by frequency.

    NaNs are left untouched.

    Only use for qualitative ordinal features.

    Fisrt fits :ref:`StringDiscretizer` if neccesary.
    """

    __name__ = "OrdinalDiscretizer"

    @extend_docstring(BaseDiscretizer.__init__, append=False, exclude=["features"])
    def __init__(
        self,
        ordinals: list[OrdinalFeature],
        min_freq: float,
        **kwargs,
    ):
        """
        Parameters
        ----------
        ordinals : list[OrdinalFeature]
            Ordinal features to process
        """
        # Initiating BaseDiscretizer
        super().__init__(ordinals, **dict(kwargs, min_freq=min_freq))

    def _prepare_data(self, sample: Sample) -> Sample:  # pylint: disable=W0222
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
        sample = super()._prepare_data(sample)

        # fitting features
        self.features.fit(**sample)

        # filling up nans for features that have some
        sample.X = self.features.fillna(sample.X)

        return sample

    @extend_docstring(BaseDiscretizer.fit)
    def fit(self, X: DataFrame, y: Series) -> None:  # pylint: disable=W0222
        # checking values orders
        sample = self._prepare_data(Sample(X, y))

        # verbose if requested
        self._log_if_verbose()

        # grouping rare modalities for each feature
        common_modalities = {
            feature.version: find_common_modalities(
                sample.X[feature.version],
                sample.y,
                min_freq=self.min_freq,
                labels=[label for label in feature.labels if label != feature.nan],
            )
            for feature in self.features
        }

        # updating features accordingly
        self.features.update(common_modalities, convert_labels=True)

        # discretizing features based on each feature's values_order
        super().fit(**sample)

        return self


def find_common_modalities(
    df_feature: Series, y: Series, min_freq: float, labels: list[str]
) -> GroupedList:
    """finds common modalities of a ordinal feature"""

    # converting to grouped list
    labels = GroupedList(labels)

    # computing frequencies and target rate of each modality
    stats, len_df = compute_stats(df_feature, y, labels)

    # case 1: there are underrepresented modalities/values
    while any(stats[0, :] / len_df < min_freq) & (stats.shape[1] > 1):
        # identifying the first underrepresented value
        discarded_idx = argmin(stats[0, :])

        # choosing amongst previous and next modality (by volume and target rate)
        kept_idx = find_closest_modality(
            discarded_idx,
            stats[0, :] / len_df,
            stats[1, :] / stats[0, :],
            min_freq,
        )

        # grouping discarded idx with kept idx
        labels.group(labels[discarded_idx], labels[kept_idx])

        # updating stats accordingly
        stats = update_stats(stats, discarded_idx, kept_idx)

    # case 2 : no underrepresented value
    return labels


def update_stats(stats: array, discarded_idx: int, kept_idx: int) -> array:
    """Updates frequencies and target rates after grouping two modalities"""

    # adding up grouped frequencies and target counts
    stats[:, kept_idx] += nan_to_num(stats[:, discarded_idx], nan=0)

    # removing discarded modality
    return stats[:, arange(stats.shape[1]) != discarded_idx]


def compute_stats(df_feature: Series, y: Series, labels: GroupedList) -> tuple[array, int]:
    """Computes frequencies and target rates of each modality"""

    # filtering nans
    not_nans = notna(df_feature)

    # total size
    len_df = len(df_feature)

    # computing frequencies and target rates
    stats = vstack(
        (
            # frequencies
            df_feature[not_nans]
            .value_counts(dropna=False, normalize=False)
            .reindex(labels, fill_value=0)
            .values,
            # target rates
            y[not_nans].groupby(df_feature[not_nans]).sum().reindex(labels).values,
        )
    )

    return stats, len_df


def find_closest_modality(
    idx: int, frequencies: array, target_rates: array, min_freq: float
) -> int:
    """Finds the closest modality in terms of frequency and target rate"""

    # case 0: only one modality
    if frequencies.shape[0] == 1:
        return 0

    # case 1: lowest ranked modality
    if idx == 0:
        return 1

    # case 2: highest ranked modality
    if idx == frequencies.shape[0] - 1:
        return idx - 1

    # checking if next modality is closer
    if is_next_modality_closer(idx, frequencies, target_rates, min_freq):
        return idx + 1

    # by default closest modality is the previous one
    return idx - 1


def is_next_modality_closer(
    idx: int, frequencies: array, target_rates: array, min_freq: float
) -> bool:
    """Determines if the next modality is closer than the previous to the current one"""

    # Extract relevant frequencies and target rates
    previous_freq, current_freq, next_freq = frequencies[idx - 1 : idx + 2]

    # comparing frequencies to min_freq
    both_below_min_freq = (next_freq < min_freq) and (previous_freq < min_freq)
    both_above_min_freq = (next_freq >= min_freq) and (previous_freq >= min_freq)

    # case 1: no observation to differentiate by target rate -> least frequent modality is choosen
    if current_freq == 0:
        return next_freq < previous_freq

    # case 2: next modality is the only below min_freq -> underrepresented modality is choosen
    if next_freq < min_freq <= previous_freq:
        return True

    # case 3: both are below or above min_freq -> closest modality by target rate
    if both_below_min_freq or both_above_min_freq:
        return is_next_modality_closer_by_target_rate(idx, target_rates)

    # by default closest modality is the previous one
    return False


def is_next_modality_closer_by_target_rate(idx: int, target_rates: array) -> bool:
    """Determines if the next modality is closer in terms of target rate than the previous to the
    current one"""

    # Extract relevant frequencies and target rates
    previous_target, current_target, next_target = target_rates[idx - 1 : idx + 2]

    # absolute difference is less for the next modality
    if abs(previous_target - current_target) > abs(next_target - current_target):
        return True

    # absolute difference is less for the previous modality
    return False
