"""Tools to build simple buckets out of Quantitative and Qualitative features
for a binary classification model.
"""

from pandas import DataFrame, Series, unique

from ...features import Features, QuantitativeFeature
from ...utils import extend_docstring
from ..qualitatives import OrdinalDiscretizer
from ..utils.base_discretizer import BaseDiscretizer
from .continuous_discretizer import ContinuousDiscretizer


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

        # Initiating BaseDiscretizer
        super().__init__(quantitatives, **dict(kwargs, min_freq=min_freq))

    @property
    def half_min_freq(self) -> float:
        """Half of the minimal frequency of a quantile."""
        return self.min_freq / 2

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
        check_quantitative_dtypes(x_copy, self.features.versions, self.__name__)

        return x_copy

    @extend_docstring(BaseDiscretizer.fit)
    def fit(self, X: DataFrame, y: Series) -> None:  # pylint: disable=W0222
        # verbose if requested
        self.log_if_verbose("------\n---")

        # checking data before bucketization
        x_copy = self._prepare_data(X, y)

        # fitting continuous features if any
        x_copy = self._fit_continuous(x_copy, y)

        # fitting continuous features with rare modalities if any
        self._fit_continuous_with_rare_modalities(x_copy, y)

        # discretizing features based on each feature's values_order
        super().fit(X, y)

        if self.verbose:  # verbose if requested
            print("------\n")

        return self

    def _fit_continuous(self, x_copy: DataFrame, y: Series) -> DataFrame:
        """Fit the ContinuousDiscretizer on the continuous features."""

        # [Quantitative features] Grouping values into quantiles
        continuous_discretizer = ContinuousDiscretizer(
            quantitatives=self.features.quantitatives,
            # copy needs to be True not to check for rare modalities
            **dict(self.kwargs, min_freq=self.min_freq, copy=True),
        )

        return continuous_discretizer.fit_transform(x_copy, y)

    def _fit_continuous_with_rare_modalities(self, x_copy: DataFrame, y: Series) -> None:
        """Fit the OrdinalDiscretizer on the continuous features with rare modalities."""

        # [Quantitative features] Grouping rare quantiles into closest common one
        #  -> can exist because of overrepresented values (values more frequent than min_freq)
        # identifying features that have rare modalities
        has_rare = check_frequencies(x_copy, self.features, self.half_min_freq)

        # Grouping rare modalities
        if len(has_rare) > 0:
            ordinal_discretizer = OrdinalDiscretizer(
                ordinals=has_rare, **dict(self.kwargs, min_freq=self.half_min_freq, copy=False)
            )
            ordinal_discretizer.fit(x_copy, y)


def check_frequencies(
    x: DataFrame, features: Features, half_min_freq: float
) -> list[QuantitativeFeature]:
    """Checks for rare modalities in the provided features."""

    # searching for features with rare quantiles: computing min frequency per feature
    frequencies = x[features.versions].apply(min_value_counts, features=features, axis=0)

    # identifying features that have rare modalities
    has_rare = list(frequencies[frequencies <= half_min_freq].index)

    # returning features with rare modalities
    return [feature for feature in features if feature.version in has_rare]


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


def check_quantitative_dtypes(x: DataFrame, feature_versions: list[str], name: str) -> None:
    """Checks if the provided features are numeric."""

    # checking for numeric columns
    dtypes = x[feature_versions].map(type).apply(unique, result_type="reduce")

    # getting non-numeric columns
    not_numeric = dtypes.apply(lambda u: str in u)

    # raising error if non-numeric columns are found
    if any(not_numeric):
        raise ValueError(
            f"[{name}] Non-numeric features: "
            f"{str(list(not_numeric[not_numeric].index))} in provided quantitative_features. "
            "Please check your inputs."
        )
