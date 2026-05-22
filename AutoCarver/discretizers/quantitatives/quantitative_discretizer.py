"""Tools to build simple buckets out of Quantitative and Qualitative features
for a binary classification model.
"""

from dataclasses import replace
from typing import Self

import pandas as pd

from AutoCarver.discretizers.qualitatives import OrdinalDiscretizer
from AutoCarver.discretizers.quantitatives.continuous_discretizer import ContinuousDiscretizer
from AutoCarver.discretizers.utils.base_discretizer import BaseDiscretizer, DiscretizerConfig, Sample
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
        config: DiscretizerConfig | None = None,
    ) -> None:
        """
        Parameters
        ----------

        quantitatives : list[QuantitativeFeature]
            Quantitative features to process
        """
        super().__init__(quantitatives, min_freq=min_freq, config=config)

    @property
    def half_min_freq(self) -> float:
        """Half of the minimal frequency of a quantile."""
        return self.min_freq / 2

    def _prepare_data(self, sample: Sample) -> Sample:  # pylint: disable=W0222
        """Validates format and content of X and y."""
        sample = super()._prepare_data(sample)

        # checking for quantitative columns
        check_quantitative_dtypes(sample.X, self.features.versions, self.__name__)

        return sample

    @extend_docstring(BaseDiscretizer.fit)
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:  # pylint: disable=W0222
        # verbose if requested
        self._log_if_verbose("------\n---")

        # checking data before bucketization
        sample = self._prepare_data(Sample(X, y))

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
        has_rare = check_frequencies(X, self.features, self.half_min_freq)

        if len(has_rare) > 0:
            ordinal_discretizer = OrdinalDiscretizer(
                ordinals=has_rare,
                min_freq=self.half_min_freq,
                config=replace(self.config, copy=False),
            )
            ordinal_discretizer.fit(X, y)


def check_frequencies(x: pd.DataFrame, features: Features, half_min_freq: float) -> list[QuantitativeFeature]:
    """Checks for rare modalities in the provided features."""

    # searching for features with rare quantiles: computing min frequency per feature
    frequencies = x[features.versions].apply(min_value_counts, features=features, axis=0)

    # identifying features that have rare modalities
    has_rare = list(frequencies[frequencies <= half_min_freq].index)

    # returning features with rare modalities
    return [feature for feature in features if feature.version in has_rare]


def min_value_counts(
    x: pd.Series,
    features: Features,
    dropna: bool = False,
    normalize: bool = True,
) -> float:
    """Minimum of modalities' frequencies."""
    # getting corresponding feature (pandas Series.name is Hashable; column names are str)
    feature = features(str(x.name))

    # modality frequency
    values = x.value_counts(dropna=dropna, normalize=normalize)

    # setting indices with known values
    if not feature.values.is_empty():
        values = values.reindex(feature.labels).fillna(0)

    # minimal frequency
    return values.values.min()


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
