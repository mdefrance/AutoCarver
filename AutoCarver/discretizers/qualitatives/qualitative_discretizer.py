"""Tools to build simple buckets out of Quantitative and Qualitative features
for a binary classification model.
"""

from pandas import DataFrame, Series

from ...features import CategoricalFeature
from ...utils import extend_docstring
from .categorical_discretizer import CategoricalDiscretizer
from .ordinal_discretizer import OrdinalDiscretizer
from .chained_discretizer import check_dtypes, check_frequencies
from ..utils.base_discretizer import BaseDiscretizer


class QualitativeDiscretizer(BaseDiscretizer):
    """Automatic discretiziation pipeline of categorical and ordinal features.

    Pipeline steps: :ref:`CategoricalDiscretizer`, :ref:`StringDiscretizer`,
    :ref:`OrdinalDiscretizer`.

    Modalities/values of features are grouped according to there respective orders:

    * [Categorical features] order based on modality target rate.
    * [Ordinal features] user-specified order.
    """

    __name__ = "QualitativeDiscretizer"

    @extend_docstring(BaseDiscretizer.__init__)
    def __init__(
        self,
        min_freq: float,
        qualitatives: list[CategoricalFeature] = None,
        **kwargs: dict,
    ) -> None:
        """
        Parameters
        ----------
        qualitative_features : list[str]
            List of column names of qualitative features (non-ordinal) to be discretized

        min_freq : float
            Minimum frequency per grouped modalities.

            * Features whose most frequent modality is < ``min_freq`` will not be discretized.
            * Sets the number of quantiles in which to discretize the continuous features.
            * Sets the minimum frequency of a quantitative feature's modality.

            **Tip**: set between ``0.02`` (slower, less robust) and ``0.05`` (faster, more robust)

        ordinal_features : list[str], optional
            List of column names of ordinal features to be discretized. For those features a list
            of values has to be provided in the ``values_orders`` dict, by default ``None``

        input_dtypes : Union[str, dict[str, str]], optional
            Input data type, converted to a dict of the provided type for each feature,
            by default ``"str"``

            * If ``"str"``, features are considered as qualitative.
            * If ``"float"``, features are considered as quantitative.
        """
        super().__init__(qualitatives, **kwargs)  # Initiating BaseDiscretizer
        self.min_freq = min_freq  # minimum frequency per modality

    def _prepare_data(self, X: DataFrame, y: Series = None) -> DataFrame:
        """Validates format and content of X and y. Converts non-string columns into strings.

        Parameters
        ----------
        X : DataFrame
            Dataset used to discretize. Needs to have columns has specified in
            ``QualitativeDiscretizer.features``.

        y : Series
            Binary target feature with wich the association is maximized.

        Returns
        -------
        DataFrame
            A formatted copy of X
        """
        # checking for binary target, copying X
        x_copy = super()._prepare_data(X, y)

        # checking feature values' frequencies
        check_frequencies(self.features, x_copy, self.min_freq, self.__name__)

        # converting non-str columns
        x_copy = check_dtypes(self.features, x_copy, **self.kwargs)

        return x_copy

    @extend_docstring(BaseDiscretizer.fit)
    def fit(self, X: DataFrame, y: Series) -> None:  # pylint: disable=W0222
        self._verbose("------\n---")  # verbose if requested

        # checking data before bucketization
        x_copy = self._prepare_data(X, y)

        # Base discretization (useful if already discretized)
        base_discretizer = BaseDiscretizer(
            features=[feature for feature in self.features if feature.is_fitted],
            **dict(self.kwargs, copy=True, dropna=False),
        )
        x_copy = base_discretizer.fit_transform(x_copy, y)

        # [Qualitative ordinal features] Grouping rare values into closest common one
        if len(self.features.ordinals) > 0:
            ordinal_discretizer = OrdinalDiscretizer(
                ordinals=self.features.ordinals,
                min_freq=self.min_freq,
                **dict(self.kwargs, copy=False),
            )
            ordinal_discretizer.fit(x_copy, y)

        # [Qualitative non-ordinal features] Grouping rare values into str_default '__OTHER__'
        if len(self.features.categoricals) > 0:
            default_discretizer = CategoricalDiscretizer(
                categoricals=self.features.categoricals,
                min_freq=self.min_freq,
                **dict(self.kwargs, copy=False),
            )
            default_discretizer.fit(x_copy, y)

        # discretizing features based on each feature's values_order
        super().fit(X, y)

        if self.verbose:  # verbose if requested
            print("------\n")

        return self
