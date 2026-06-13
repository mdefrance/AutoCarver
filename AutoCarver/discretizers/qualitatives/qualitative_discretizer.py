"""Tools to build simple buckets out of Quantitative and Qualitative features
for a binary classification model.
"""

from dataclasses import replace
from typing import Self

import pandas as pd

from AutoCarver.discretizers.qualitatives.categorical_discretizer import CategoricalDiscretizer
from AutoCarver.discretizers.qualitatives.nested_discretizer import NestedDiscretizer, check_frequencies
from AutoCarver.discretizers.qualitatives.ordinal_discretizer import OrdinalDiscretizer
from AutoCarver.discretizers.utils.base_discretizer import BaseDiscretizer, DiscretizerConfig, Sample
from AutoCarver.discretizers.utils.type_discretizers import ensure_qualitative_dtypes
from AutoCarver.features import Features, QualitativeFeature
from AutoCarver.utils import extend_docstring


class QualitativeDiscretizer(BaseDiscretizer):
    """Automatic discretiziation pipeline of categorical and ordinal features.

    Pipeline steps: :ref:`CategoricalDiscretizer`, :ref:`StringDiscretizer`,
    :ref:`OrdinalDiscretizer`.

    Modalities/values of features are grouped according to there respective orders:

    * [Categorical features] order based on modality target rate.
    * [Ordinal features] user-specified order.
    """

    __name__ = "QualitativeDiscretizer"

    @extend_docstring(BaseDiscretizer.__init__, append=False, exclude=["features"])
    def __init__(
        self,
        qualitatives: list[QualitativeFeature],
        min_freq: float,
        *,
        config: DiscretizerConfig | None = None,
    ) -> None:
        """
        Parameters
        ----------

        qualitatives : list[QualitativeFeature]
            Qualitative features to process
        """
        super().__init__(qualitatives, min_freq=min_freq, config=config)

    def _prepare_sample(self, sample: Sample) -> Sample:
        """Validates format and content of X and y. Converts non-string columns into strings."""
        sample.X = super()._prepare_X(sample.X)

        # checking feature values' frequencies (nested features are excluded: their finest
        # modalities are legitimately rare and get rolled up by the NestedDiscretizer)
        non_nested = [feature for feature in self.features if not feature.is_nested]
        if len(non_nested) > 0:
            check_frequencies(Features.from_list(non_nested), sample.X, self.min_freq, self.__name__)

        # converting non-str columns
        sample.X = ensure_qualitative_dtypes(self.features, sample.X, config=self.config)

        return sample

    @extend_docstring(BaseDiscretizer.fit)
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        # verbose if requested
        self._log_if_verbose("------\n---")

        # checking data before bucketization
        sample = self._prepare_sample(Sample(X, y))

        # Base discretization (useful if already discretized)
        sample.X = self._base_transform(**sample)

        # rolling up nested features first (collapses nested columns to one robust column)
        self._fit_nested(**sample)

        # fitting ordinal features if any
        self._fit_ordinals(**sample)

        # fitting categorical features if any
        self._fit_categoricals(**sample)

        # discretizing features based on each feature's values_order
        super().fit(**sample)

        if self.config.verbose:  # verbose if requested
            print("------\n")

        return self

    def _base_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Transform the data based on the previously fitted discretizers."""

        # looking for already discretized features
        discretized_features = [feature for feature in self.features if feature.is_fitted]

        if len(discretized_features) > 0:
            base_discretizer = BaseDiscretizer(
                features=discretized_features,
                config=replace(self.config, copy=True, dropna=False),
            )
            X = base_discretizer.fit_transform(X, y)

        return X

    def _fit_nested(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the NestedDiscretizer on the nested features."""

        if len(self.features.nested) > 0:
            nested_discretizer = NestedDiscretizer(
                nesteds=self.features.nested,
                min_freq=self.min_freq,
                config=replace(self.config, copy=False),
            )
            nested_discretizer.fit(X, y)

    def _fit_ordinals(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the OrdinalDiscretizer on the ordinal features."""

        if len(self.features.ordinals) > 0:
            ordinal_discretizer = OrdinalDiscretizer(
                ordinals=self.features.ordinals,
                min_freq=self.min_freq,
                config=replace(self.config, copy=False),
            )
            ordinal_discretizer.fit(X, y)

    def _fit_categoricals(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the CategoricalDiscretizer on the categorical features."""

        if len(self.features.categoricals) > 0:
            categorical_discretizer = CategoricalDiscretizer(
                categoricals=self.features.categoricals,
                min_freq=self.min_freq,
                config=replace(self.config, copy=False),
            )
            categorical_discretizer.fit(X, y)
