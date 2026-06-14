"""Tools to collapse several nested qualitative columns into a single robust column.

Given columns of increasing granularity (``col_a`` ⊃ ``col_b`` ⊃ ``col_c``), each
:class:`NestedFeature` rolls its rare finest modalities up to their *data-derived* parent
modality, repeatedly, until every surviving modality is frequent enough or no coarser level
remains. The result is a single output column (the finest one) where all modalities have a
sufficient number of observations — improving downstream model robustness.
"""

from dataclasses import replace
from typing import Self

import pandas as pd

from AutoCarver.discretizers.qualitatives.categorical_discretizer import series_target_rate
from AutoCarver.discretizers.utils.base_discretizer import BaseDiscretizer, ProcessingConfig, Sample
from AutoCarver.discretizers.utils.frequency_ci import is_significantly_below
from AutoCarver.discretizers.utils.type_discretizers import ensure_qualitative_dtypes
from AutoCarver.features import Features, GroupedList, NestedFeature
from AutoCarver.utils import extend_docstring


class NestedDiscretizer(BaseDiscretizer):
    """Automatic discretization of nested qualitative features.

    For each :class:`NestedFeature`, modalities of the finest (output) column whose frequency
    is significantly below ``min_freq`` (Wilson upper bound at ``config.min_freq_alpha``) are
    rolled up to the coarser modality they are nested within — derived from the data. This
    repeats level by level until all surviving modalities are frequent enough or the coarsest
    level is reached, collapsing every nested column into the single output column.
    """

    __name__ = "NestedDiscretizer"

    @extend_docstring(BaseDiscretizer.__init__, append=False, exclude=["features"])
    def __init__(
        self,
        nesteds: list[NestedFeature],
        min_freq: float,
        *,
        config: ProcessingConfig | None = None,
    ) -> None:
        """
        Parameters
        ----------
        nesteds : list[NestedFeature]
            Nested features to process.
        """
        # nested discretization never drops nans (handled by the fillna/unfillna machinery)
        if config is None:
            config = ProcessingConfig()
        config = replace(config, dropna=False)

        super().__init__(nesteds, min_freq=min_freq, config=config)

    @property
    def _parent_columns(self) -> list[str]:
        """All distinct parent columns referenced by the nested features."""
        return sorted({parent for feature in self.features for parent in feature.parents})

    def _prepare_sample(self, sample: Sample) -> Sample:
        """Validates format and content of X and y, and converts nested columns to strings."""
        # copying dataframe
        sample.X = sample.X.copy()

        # checking X/y and output (finest) columns
        sample = super()._prepare_sample(sample)

        # checking that parent columns are present
        missing = [column for column in self._parent_columns if column not in sample.X]
        if len(missing) > 0:
            raise ValueError(
                f"[{self.__name__}] Requested nesting on parent columns {missing} but those "
                "columns are missing from provided X. Please check your inputs!"
            )

        # converting non-str output columns
        sample.X = ensure_qualitative_dtypes(self.features, sample.X, config=self.config)

        # converting non-str parent columns (they are not Features themselves)
        sample.X = self._ensure_parent_dtypes(sample.X)

        # fitting features (initiates output-column modalities)
        self.features.fit(**sample)

        # filling up nans for features that have some
        sample.X = self.features.fillna(sample.X)

        return sample

    def _ensure_parent_dtypes(self, X: pd.DataFrame) -> pd.DataFrame:
        """Converts parent columns to strings, consistently with the output columns."""
        parent_columns = self._parent_columns
        if len(parent_columns) > 0:
            parent_features = Features(categoricals=parent_columns)
            X = ensure_qualitative_dtypes(parent_features, X, config=self.config)
        return X

    @extend_docstring(BaseDiscretizer.fit)
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        # preprocessing data
        sample = self._prepare_sample(Sample(X, y))
        self._log_if_verbose()  # verbose if requested

        nobs = len(sample.X)

        # rolling up rare modalities of each nested feature
        for feature in self.features:
            self._apply_rollup(feature, sample.X, nobs)

        # sorting buckets by target rate (matches CategoricalDiscretizer) when a target is given
        if sample.y is not None:
            self._target_sort(**sample)

        # every nested feature keeps a default modality so unseen finest values seen only at
        # transform have a fallback bucket (enabled after the sort to avoid an empty default
        # being dropped by sort_by)
        for feature in self.features:
            if not feature.has_default:
                feature.has_default = True

        super().fit(X, y)

        if self.config.verbose:  # verbose if requested
            print("\n")

        return self

    def _apply_rollup(self, feature: NestedFeature, X: pd.DataFrame, nobs: int) -> None:
        """Computes a feature's rollup, historizes it in its order and materializes it in X."""
        raw_to_bucket = self._compute_rollup(feature, X, nobs)

        # enabling the default modality (__OTHER__) when rare buckets were pooled into it,
        # so it becomes a valid leader to group rolled-up values under
        if feature.default in raw_to_bucket.values():
            feature.has_default = True

        # historizing the rollup in the feature's order
        order = GroupedList(feature.values)
        for bucket in dict.fromkeys(raw_to_bucket.values()):
            if bucket not in order.values:
                order.append(bucket)
        for raw_value, bucket in raw_to_bucket.items():
            if raw_value != bucket:
                order.group(raw_value, bucket)
        feature.update(order, replace=True)

        # materializing rolled-up labels in the output column (for target-rate sort)
        X[feature.version] = X[feature.version].replace(raw_to_bucket)

    def _compute_rollup(self, feature: NestedFeature, X: pd.DataFrame, nobs: int) -> dict:
        """Computes, for each raw finest modality, the bucket it rolls up into.

        Rare modalities (Wilson upper bound below ``min_freq``) are replaced level by level by
        the parent modality they co-occur with, until frequent enough or the coarsest level is
        reached. Returns a ``{raw finest value: final bucket}`` mapping.
        """
        # working columns finest-to-coarsest: the (possibly versioned) output column + parents
        level_columns = [feature.version] + feature.parents
        alpha = self.config.min_freq_alpha

        # parent maps between consecutive levels, derived (and validated) from data
        parent_maps = [
            self._derive_parent_map(X, child, parent, feature)
            for child, parent in zip(level_columns[:-1], level_columns[1:])
        ]

        # current label per row, starting from the finest column
        current = X[feature.version].copy()

        for parent_map in parent_maps:
            counts = current.value_counts(dropna=True)

            # modalities significantly below min_freq (excluding nan)
            rare = [
                value
                for value, count in counts.items()
                if value != feature.nan
                and pd.notna(value)
                and is_significantly_below(count, nobs, self.min_freq, alpha)
            ]
            if len(rare) == 0:
                break

            # rolling rare modalities up to their parent (when one exists)
            relabel = {value: parent_map[value] for value in rare if value in parent_map}
            if len(relabel) == 0:
                break

            current = current.replace(relabel)

        # terminal pooling: buckets still significantly below min_freq (couldn't reach it even at
        # the coarsest level) are pooled into the default modality (__OTHER__)
        counts = current.value_counts(dropna=True)
        terminal_rare = [
            value
            for value, count in counts.items()
            if value != feature.nan and pd.notna(value) and is_significantly_below(count, nobs, self.min_freq, alpha)
        ]
        if len(terminal_rare) > 0:
            current = current.replace(dict.fromkeys(terminal_rare, feature.default))

        # mapping each raw finest value to its final bucket (functional by construction)
        rolled = pd.DataFrame({"raw": X[feature.version], "bucket": current})
        rolled = rolled[pd.notna(rolled["raw"]) & (rolled["raw"] != feature.nan)]
        return {raw_value: buckets.iloc[0] for raw_value, buckets in rolled.groupby("raw")["bucket"]}

    def _derive_parent_map(self, X: pd.DataFrame, child: str, parent: str, feature: NestedFeature) -> dict:
        """Builds ``{child modality: parent modality}`` from data co-occurrence.

        Raises when a child modality is nested within more than one parent modality (the
        columns are not a clean hierarchy).
        """
        pairs = X[[child, parent]]
        pairs = pairs[pd.notna(pairs[child]) & (pairs[child] != feature.nan)]

        parent_map = {}
        for child_value, parents in pairs.groupby(child)[parent]:
            distinct = [value for value in parents.unique() if pd.notna(value) and value != feature.nan]
            if len(distinct) > 1:
                raise ValueError(
                    f"[{self.__name__}] Modality {child_value!r} of {child!r} is nested within "
                    f"several modalities {distinct} of {parent!r}. Columns are not properly nested."
                )
            if len(distinct) == 1:
                parent_map[child_value] = distinct[0]
        return parent_map

    def _target_sort(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Sorts each feature's buckets by target rate (mirrors CategoricalDiscretizer)."""
        target_rates = X[self.features.versions].apply(series_target_rate, y=y, axis=0)
        self.features.update(
            {feature: list(sorted_values) for feature, sorted_values in target_rates.items()},
            sorted_values=True,
        )


def check_frequencies(features: Features, X: pd.DataFrame, min_freq: float, name: str) -> None:
    """Checks features' frequencies compared to min_freq"""

    # computing features' max modality frequency (mode frequency)
    max_freq = X[features.versions].apply(
        lambda u: u.value_counts(normalize=True, dropna=False).max(),
        axis=0,
    )

    # features with no common modality (biggest value less frequent than min_freq)
    non_common = [f.version for f in features if max_freq[f.version] < min_freq]

    # features with too common modality (biggest value more frequent than 1-min_freq)
    too_common = [f.version for f in features if max_freq[f.version] > 1 - min_freq]

    # raising
    if len(too_common + non_common) > 0:
        # building error message
        error_msg = (
            f"[{name}] Features {str(too_common + non_common)} contain a too frequent modality "
            "or no frequent enough modalities. Consider decreasing min_freq or removing these "
            "features.\nINFO:\n"
        )

        # adding features with no common values
        non_common = [
            (
                f" - {features(feature)}: most frequent value has "
                f"freq={max_freq[feature]:2.2%} < min_freq={min_freq:2.2%}"
            )
            for feature in non_common
        ]

        # adding features with too common values
        too_common = [
            (
                f" - {features(feature)}: most frequent value has "
                f"freq={max_freq[feature]:2.2%} > (1-min_freq)={1 - min_freq:2.2%}"
            )
            for feature in too_common
        ]
        error_msg += "\n".join(too_common + non_common)

        raise ValueError(error_msg)
