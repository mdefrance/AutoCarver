"""Tools to build simple buckets out of Quantitative and Qualitative features
for a binary classification model.
"""

from pandas import DataFrame, Series

from ...features import QualitativeFeature
from ...utils import extend_docstring
from ..utils.base_discretizer import BaseDiscretizer, Sample
from .categorical_discretizer import CategoricalDiscretizer
from .chained_discretizer import check_frequencies, ensure_qualitative_dtypes
from .ordinal_discretizer import OrdinalDiscretizer


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
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------

        qualitatives : list[QualitativeFeature]
            Qualitative features to process

        """
        # Initiating BaseDiscretizer
        super().__init__(qualitatives, **dict(kwargs, min_freq=min_freq))

    def _prepare_data(self, sample: Sample) -> Sample:
        """Validates format and content of X and y. Converts non-string columns into strings."""
        sample.X = super()._prepare_X(sample.X)

        # checking feature values' frequencies
        check_frequencies(self.features, sample.X, self.min_freq, self.__name__)

        # converting non-str columns
        sample.X = ensure_qualitative_dtypes(self.features, sample.X, **self.kwargs)

        return sample

    @extend_docstring(BaseDiscretizer.fit)
    def fit(self, X: DataFrame, y: Series) -> None:  # pylint: disable=W0222
        # verbose if requested
        self._log_if_verbose("------\n---")

        # checking data before bucketization
        sample = self._prepare_data(Sample(X, y))

        # Base discretization (useful if already discretized)
        sample.X = self._base_transform(**sample)

        # fitting ordinal features if any
        self._fit_ordinals(**sample)

        # fitting categorical features if any
        self._fit_categoricals(**sample)

        # discretizing features based on each feature's values_order
        super().fit(**sample)

        if self.verbose:  # verbose if requested
            print("------\n")

        return self

    def _base_transform(self, X: DataFrame, y: Series) -> DataFrame:
        """Transform the data based on the previously fitted discretizers."""

        # looking for already discretized features
        discretized_features = [feature for feature in self.features if feature.is_fitted]

        # Base discretization (useful if already discretized)
        if len(discretized_features) > 0:
            base_discretizer = BaseDiscretizer(
                features=discretized_features, **dict(self.kwargs, copy=True, dropna=False)
            )
            X = base_discretizer.fit_transform(X, y)

        return X

    def _fit_ordinals(self, X: DataFrame, y: Series) -> None:
        """Fit the OrdinalDiscretizer on the ordinal features."""

        # [Qualitative ordinal features] Grouping rare values into closest common one
        if len(self.features.ordinals) > 0:
            ordinal_discretizer = OrdinalDiscretizer(
                ordinals=self.features.ordinals,
                **dict(self.kwargs, min_freq=self.min_freq, copy=False),
            )
            ordinal_discretizer.fit(X, y)

    def _fit_categoricals(self, X: DataFrame, y: Series) -> None:
        """Fit the CategoricalDiscretizer on the categorical features."""

        # [Qualitative non-ordinal features] Grouping rare values into default '__OTHER__'
        if len(self.features.categoricals) > 0:
            categorical_discretizer = CategoricalDiscretizer(
                categoricals=self.features.categoricals,
                **dict(self.kwargs, min_freq=self.min_freq, copy=False),
            )
            categorical_discretizer.fit(X, y)
