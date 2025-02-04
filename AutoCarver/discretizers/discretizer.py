"""Tools to build simple buckets out of Quantitative and Qualitative features
for a binary classification model.
"""

from pandas import DataFrame, Series

from ..features import Features
from ..utils import extend_docstring
from .qualitatives import QualitativeDiscretizer
from .quantitatives import QuantitativeDiscretizer
from .utils.base_discretizer import BaseDiscretizer, Sample


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
        **kwargs,
    ) -> None:
        # Initiating BaseDiscretizer
        super().__init__(features, **dict(kwargs, min_freq=min_freq))

    @extend_docstring(BaseDiscretizer.fit)
    def fit(self, X: DataFrame, y: Series) -> None:  # pylint: disable=W0222
        # Checking for binary target and copying X
        sample = self._prepare_data(Sample(X, y))

        # fitting quantitative features if any
        self._fit_quantitatives(**sample)

        # fitting qualitative features if any
        self._fit_qualitatives(**sample)

        # discretizing features based on each feature's values_order
        super().fit(**sample)

        return self

    def _fit_qualitatives(self, X: DataFrame, y: Series) -> None:
        """Fit the QualitativeDiscretizer on the qualitative features."""

        # [Qualitative features] Grouping qualitative features
        if len(self.features.qualitatives) > 0:
            # grouping qualitative features
            qualitative_discretizer = QualitativeDiscretizer(
                qualitatives=self.features.qualitatives, **dict(self.kwargs, copy=False)
            )
            qualitative_discretizer.fit(X, y)

    def _fit_quantitatives(self, X: DataFrame, y: Series) -> None:
        """Fit the QuantitativeDiscretizer on the quantitative features."""

        # [Quantitative features] Grouping quantitative features
        if len(self.features.quantitatives) > 0:
            # grouping quantitative features
            quantitative_discretizer = QuantitativeDiscretizer(
                quantitatives=self.features.quantitatives, **dict(self.kwargs, copy=False)
            )
            quantitative_discretizer.fit(X, y)
