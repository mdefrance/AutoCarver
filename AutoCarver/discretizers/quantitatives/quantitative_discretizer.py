"""Tools to build simple buckets out of Quantitative and Qualitative features
for a binary classification model.
"""

from dataclasses import replace
from typing import Self

import pandas as pd

from AutoCarver.discretizers.qualitatives import OrdinalDiscretizer
from AutoCarver.discretizers.quantitatives.continuous_discretizer import ContinuousDiscretizer
from AutoCarver.discretizers.utils.base_discretizer import BaseDiscretizer, ProcessingConfig, Sample
from AutoCarver.discretizers.utils.frequency_ci import is_significantly_below
from AutoCarver.discretizers.utils.type_discretizers import ensure_datetime_dtypes
from AutoCarver.features import Features, QuantitativeFeature
from AutoCarver.utils import extend_docstring


class QuantitativeDiscretizer(BaseDiscretizer):
    """Automatic discretization pipeline of continuous and discrete features.

    Pipeline steps: :ref:`ContinuousDiscretizer`, :ref:`OrdinalDiscretizer`

    Modalities/values of features are grouped according to there respective orders:

     * [Continuous/Discrete features] real order of the values.
    """

    __name__ = "QuantitativeDiscretizer"

    @extend_docstring(BaseDiscretizer.__init__, append=False, exclude=["features"])
    def __init__(
        self,
        quantitatives: list[QuantitativeFeature],
        min_freq: float,
        *,
        config: ProcessingConfig | None = None,
    ) -> None:
        """
        Parameters
        ----------

        quantitatives : list[QuantitativeFeature]
            Quantitative features to process
        """
        super().__init__(quantitatives, min_freq=min_freq, config=config)

    def _prepare_sample(self, sample: Sample) -> Sample:
        """Validates format and content of X and y. Converts datetime columns to timedeltas."""
        sample = super()._prepare_sample(sample)

        # converting datetime features into numeric (seconds since reference_date)
        sample.X = ensure_datetime_dtypes(self.features, sample.X, config=self.config)

        # checking for quantitative columns
        check_quantitative_dtypes(sample.X, self.features.versions, self.__name__)

        return sample

    @extend_docstring(BaseDiscretizer.fit)
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        # verbose if requested
        self._log_if_verbose("------\n---")

        # checking data before bucketization
        sample = self._prepare_sample(Sample(X, y))

        # fitting continuous features if any
        sample.X = self._fit_continuous(**sample)

        # fitting continuous features with rare modalities if any
        self._fit_continuous_with_rare_modalities(**sample)

        # discretizing features based on each feature's values_order
        super().fit(X, y)

        if self.config.verbose:  # verbose if requested
            print("------\n")

        return self

    def _fit_continuous(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit the ContinuousDiscretizer on the continuous features."""

        # copy needs to be True so the next step can check for rare modalities on a stable copy
        continuous_discretizer = ContinuousDiscretizer(
            quantitatives=self.features.quantitatives,
            min_freq=self.min_freq,
            config=replace(self.config, copy=True),
        )

        return continuous_discretizer.fit_transform(X, y)

    def _fit_continuous_with_rare_modalities(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the OrdinalDiscretizer on the continuous features with rare modalities."""

        # rare quantiles can exist because of overrepresented values (more frequent than min_freq)
        has_rare = check_frequencies(X, self.features, self.min_freq, self.config.min_freq_alpha)

        if len(has_rare) > 0:
            ordinal_discretizer = OrdinalDiscretizer(
                ordinals=has_rare,
                min_freq=self.min_freq,
                config=replace(self.config, copy=False),
            )
            ordinal_discretizer.fit(X, y)


def check_frequencies(x: pd.DataFrame, features: Features, min_freq: float, alpha: float) -> list[QuantitativeFeature]:
    """Checks for rare modalities in the provided features.

    A modality is considered rare when the Wilson upper bound of its
    observed proportion (at significance level ``alpha``) is strictly below
    ``min_freq`` — i.e. its frequency is significantly below the target.
    """

    # smallest modality count per feature
    min_counts = x[features.versions].apply(min_value_counts, features=features, axis=0)

    # CI-based rare-flag: only features whose smallest modality is significantly
    # below min_freq are sent through the OrdinalDiscretizer.
    rare_mask = is_significantly_below(min_counts.values, len(x), min_freq, alpha)
    has_rare = list(min_counts.index[rare_mask])

    # returning features with rare modalities
    return [feature for feature in features if feature.version in has_rare]


def min_value_counts(
    x: pd.Series,
    features: Features,
    dropna: bool = False,
) -> int:
    """Smallest modality count for a feature (integer count, not proportion)."""
    # getting corresponding feature (pandas Series.name is Hashable; column names are str)
    feature = features(str(x.name))

    # modality counts
    values = x.value_counts(dropna=dropna, normalize=False)

    # setting indices with known values
    if not feature.values.is_empty():
        values = values.reindex(feature.labels).fillna(0)

    # minimal count
    return int(values.values.min())


def check_quantitative_dtypes(x: pd.DataFrame, feature_versions: list[str], name: str) -> None:
    """Checks if the provided features are numeric."""

    # checking for numeric columns
    dtypes = x[feature_versions].map(type).apply(pd.unique, result_type="reduce")

    # getting non-numeric columns
    not_numeric = dtypes.apply(lambda u: str in u)

    # raising error if non-numeric columns are found
    if any(not_numeric):
        raise ValueError(
            f"[{name}] Non-numeric features: "
            f"{str(list(not_numeric[not_numeric].index))} in provided quantitative_features. "
            "Please check your inputs."
        )
