"""Tools to build simple buckets out of Quantitative and Qualitative features
for a binary classification model.
"""

from pandas import DataFrame, Series, unique

from ..features import Features, CategoricalFeature, QuantitativeFeature
from .utils.base_discretizer import BaseDiscretizer, extend_docstring
from .qualitative_discretizers import (
    CategoricalDiscretizer,
    OrdinalDiscretizer,
    check_frequencies,
    check_dtypes,
)
from .quantitative_discretizers import ContinuousDiscretizer


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
        min_freq: float,
        features: Features,
        **kwargs: dict,
    ) -> None:
        """
        Parameters
        ----------
        quantitative_features : list[str]
            List of column names of quantitative features (continuous and discrete) to be dicretized

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
        """
        super().__init__(features, **kwargs)  # Initiating BaseDiscretizer
        self.min_freq = min_freq  # minimum frequency per modality

    @extend_docstring(BaseDiscretizer.fit)
    def fit(self, X: DataFrame, y: Series) -> None:  # pylint: disable=W0222
        # Checking for binary target and copying X
        x_copy = self._prepare_data(X, y)

        kept_features: list[str] = []  # list of viable features

        # [Qualitative features] Grouping qualitative features
        if len(self.features.get_qualitatives()) > 0:
            # grouping qualitative features
            qualitative_discretizer = QualitativeDiscretizer(
                qualitatives=self.features.get_qualitatives(),
                min_freq=self.min_freq,
                copy=False,  # always False as x_copy is already a copy (if requested)
                verbose=self.verbose,
                n_jobs=self.n_jobs,
            )
            qualitative_discretizer.fit(x_copy, y)

            # saving kept features
            kept_features += qualitative_discretizer.features.get_versions()

        # [Quantitative features] Grouping quantitative features
        if len(self.features.get_quantitatives()) > 0:
            # grouping quantitative features
            quantitative_discretizer = QuantitativeDiscretizer(
                quantitatives=self.features.get_quantitatives(),
                min_freq=self.min_freq,
                verbose=self.verbose,
                n_jobs=self.n_jobs,
            )
            quantitative_discretizer.fit(x_copy, y)

            # saving kept features
            kept_features += quantitative_discretizer.features.get_versions()

        # removing dropped features
        self.features.keep(kept_features)

        # discretizing features based on each feature's values_order
        super().fit(X, y)

        return self


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


class QuantitativeDiscretizer(BaseDiscretizer):
    """Automatic discretization pipeline of continuous and discrete features.

    Pipeline steps: :ref:`ContinuousDiscretizer`, :ref:`OrdinalDiscretizer`

    Modalities/values of features are grouped according to there respective orders:

     * [Continuous/Discrete features] real order of the values.
    """

    __name__ = "QuantitativeDiscretizer"

    @extend_docstring(BaseDiscretizer.__init__)
    def __init__(
        self,
        quantitatives: list[QuantitativeFeature],
        min_freq: float,
        **kwargs: dict,
    ) -> None:
        """
        Parameters
        ----------
        quantitative_features : list[str]
            List of column names of quantitative features (continuous and discrete) to be dicretized

        min_freq : float
            Minimum frequency per grouped modalities.

            * Features whose most frequent modality is < ``min_freq`` will not be discretized.
            * Sets the number of quantiles in which to discretize the continuous features.
            * Sets the minimum frequency of a quantitative feature's modality.

            **Tip**: set between ``0.02`` (slower, less robust) and ``0.05`` (faster, more robust)

        input_dtypes : Union[str, dict[str, str]], optional
            Input data type, converted to a dict of the provided type for each feature,
            by default ``"str"``

            * If ``"str"``, features are considered as qualitative.
            * If ``"float"``, features are considered as quantitative.
        """
        super().__init__(quantitatives, **kwargs)  # Initiating BaseDiscretizer
        self.min_freq = min_freq  # minimum frequency per modality

    def _prepare_data(self, X: DataFrame, y: Series) -> DataFrame:  # pylint: disable=W0222
        """Validates format and content of X and y.

        Parameters
        ----------
        X : DataFrame
            Dataset used to discretize. Needs to have columns has specified in
            ``QuantitativeDiscretizer.features``.

        y : Series
            Binary target feature with wich the association is maximized.

        Returns
        -------
        DataFrame
            A formatted copy of X
        """
        # checking for binary target and copying X
        x_copy = super()._prepare_data(X, y)

        # checking for quantitative columns
        dtypes = x_copy[self.features.get_versions()].map(type).apply(unique, result_type="reduce")
        not_numeric = dtypes.apply(lambda u: str in u)
        if any(not_numeric):
            raise ValueError(
                f" - [{self.__name__}] Non-numeric features: "
                f"{str(list(not_numeric[not_numeric].index))} in provided quantitative_features. "
                "Please check your inputs."
            )
        return x_copy

    @extend_docstring(BaseDiscretizer.fit)
    def fit(self, X: DataFrame, y: Series) -> None:  # pylint: disable=W0222
        self._verbose("------\n---")  # verbose if requested

        # checking data before bucketization
        x_copy = self._prepare_data(X, y)

        # [Quantitative features] Grouping values into quantiles
        continuous_discretizer = ContinuousDiscretizer(
            quantitatives=self.features.quantitatives,
            min_freq=self.min_freq,
            **dict(self.kwargs, copy=True),  # needs to be True not to transform x_copy
        )

        x_copy = continuous_discretizer.fit_transform(x_copy, y)

        # [Quantitative features] Grouping rare quantiles into closest common one
        #  -> can exist because of overrepresented values (values more frequent than min_freq)
        # searching for features with rare quantiles: computing min frequency per feature
        frequencies = x_copy[self.features.get_versions()].apply(
            min_value_counts, features=self.features, axis=0
        )

        # minimal frequency of a quantile
        q_min_freq = self.min_freq / 2

        # identifying features that have rare modalities
        has_rare = list(frequencies[frequencies <= q_min_freq].index)

        # Grouping rare modalities
        if len(has_rare) > 0:
            ordinal_discretizer = OrdinalDiscretizer(
                ordinals=[feature for feature in self.features if feature.version in has_rare],
                min_freq=q_min_freq,
                copy=False,
                verbose=self.verbose,
                n_jobs=self.n_jobs,
            )
            ordinal_discretizer.fit(x_copy, y)

        # discretizing features based on each feature's values_order
        super().fit(X, y)

        if self.verbose:  # verbose if requested
            print("------\n")

        return self


def min_value_counts(
    x: Series,
    features: Features,
    dropna: bool = False,
    normalize: bool = True,
) -> float:
    """Minimum of modalities' frequencies."""
    # getting corresponding feature
    feature = features(x.name)

    # modality frequency
    values = x.value_counts(dropna=dropna, normalize=normalize)

    # setting indices with known values
    if feature.values is not None:
        values = values.reindex(feature.labels).fillna(0)

    # minimal frequency
    return values.values.min()
