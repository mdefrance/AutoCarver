"""Tools to build simple buckets out of Quantitative and Qualitative features
for a binary classification model.
"""

from dataclasses import replace
from typing import Self

import pandas as pd

from AutoCarver.discretizers.qualitatives import QualitativeDiscretizer
from AutoCarver.discretizers.quantitatives import QuantitativeDiscretizer
from AutoCarver.discretizers.utils.base_discretizer import BaseDiscretizer, ProcessingConfig, Sample
from AutoCarver.features import Features
from AutoCarver.utils import extend_docstring


class Discretizer(BaseDiscretizer):
    """Automatic discretization pipeline of continuous, discrete, categorical and ordinal features.

    Pipeline steps: :ref:`QuantitativeDiscretizer`, :ref:`QualitativeDiscretizer`.

    Modalities/values of features are grouped according to there respective orders:

    * [Categorical features] order based on modality target rate.
    * [Ordinal features] user-specified order.
    * [Continuous/Discrete features] real order of the values.
    """

    __name__ = "Discretizer"

    @extend_docstring(BaseDiscretizer.__init__)
    def __init__(
        self,
        features: Features,
        min_freq: float,
        *,
        config: ProcessingConfig | None = None,
    ) -> None:
        super().__init__(features, min_freq=min_freq, config=config)

    @extend_docstring(BaseDiscretizer.fit)
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        # Checking for binary target and copying X
        sample = self._prepare_sample(Sample(X, y))

        # fitting quantitative features if any
        self._fit_quantitatives(**sample)

        # fitting qualitative features if any
        self._fit_qualitatives(**sample)

        # discretizing features based on each feature's values_order
        super().fit(**sample)

        return self

    def _fit_qualitatives(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the QualitativeDiscretizer on the qualitative features."""

        if len(self.features.qualitatives) > 0:
            qualitative_discretizer = QualitativeDiscretizer(
                qualitatives=self.features.qualitatives,
                min_freq=self.min_freq,
                config=replace(self.config, copy=False),
            )
            qualitative_discretizer.fit(X, y)

    def _fit_quantitatives(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the QuantitativeDiscretizer on the quantitative features."""

        if len(self.features.quantitatives) > 0:
            quantitative_discretizer = QuantitativeDiscretizer(
                quantitatives=self.features.quantitatives,
                min_freq=self.min_freq,
                config=replace(self.config, copy=False),
            )
            quantitative_discretizer.fit(X, y)
