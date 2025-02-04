"""Tools to build simple buckets out of Quantitative and Qualitative features
for a binary classification model.
"""

from pandas import DataFrame, Series, unique

from ...features import Features, QuantitativeFeature
from ...utils import extend_docstring
from ..qualitatives import OrdinalDiscretizer
from ..utils.base_discretizer import BaseDiscretizer, Sample
from .continuous_discretizer import ContinuousDiscretizer


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
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------

        quantitatives : list[QuantitativeFeature]
            Quantitative features to process
        """

        # Initiating BaseDiscretizer
        super().__init__(quantitatives, **dict(kwargs, min_freq=min_freq))

    @property
    def half_min_freq(self) -> float:
        """Half of the minimal frequency of a quantile."""
        return self.min_freq / 2

    def _prepare_data(self, sample: Sample) -> Sample:  # pylint: disable=W0222
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
        sample = super()._prepare_data(sample)

        # checking for quantitative columns
        check_quantitative_dtypes(sample.X, self.features.versions, self.__name__)

        return sample

    @extend_docstring(BaseDiscretizer.fit)
    def fit(self, X: DataFrame, y: Series) -> None:  # pylint: disable=W0222
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

        if self.verbose:  # verbose if requested
            print("------\n")

        return self

    def _fit_continuous(self, X: DataFrame, y: Series) -> DataFrame:
        """Fit the ContinuousDiscretizer on the continuous features."""

        # [Quantitative features] Grouping values into quantiles
        continuous_discretizer = ContinuousDiscretizer(
            quantitatives=self.features.quantitatives,
            # copy needs to be True not to check for rare modalities
            **dict(self.kwargs, min_freq=self.min_freq, copy=True),
        )

        return continuous_discretizer.fit_transform(X, y)

    def _fit_continuous_with_rare_modalities(self, X: DataFrame, y: Series) -> None:
        """Fit the OrdinalDiscretizer on the continuous features with rare modalities."""

        # [Quantitative features] Grouping rare quantiles into closest common one
        #  -> can exist because of overrepresented values (values more frequent than min_freq)
        # identifying features that have rare modalities
        has_rare = check_frequencies(X, self.features, self.half_min_freq)

        # Grouping rare modalities
        if len(has_rare) > 0:
            ordinal_discretizer = OrdinalDiscretizer(
                ordinals=has_rare, **dict(self.kwargs, min_freq=self.half_min_freq, copy=False)
            )
            ordinal_discretizer.fit(X, y)


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
