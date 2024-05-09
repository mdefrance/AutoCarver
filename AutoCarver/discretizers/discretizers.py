"""Tools to build simple buckets out of Quantitative and Qualitative features
for a binary classification model.
"""

from typing import Any, Union
from warnings import warn

from numpy import nan
from pandas import DataFrame, Series, unique

from ..config import DEFAULT, NAN
from .utils.base_discretizers import BaseDiscretizer, extend_docstring
from ..features import GroupedList
from .utils.qualitative_discretizers import CategoricalDiscretizer, OrdinalDiscretizer
from .utils.quantitative_discretizers import ContinuousDiscretizer
from .utils.type_discretizers import StringDiscretizer


from ..features import GroupedList, Features, BaseFeature


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
        categoricals: list[str] = None,
        ordinals: list[str] = None,
        quantitatives: list[str] = None,
        *,
        ordinal_values: dict[str, GroupedList] = None,
        # copy: bool = False,
        # verbose: bool = False,
        # n_jobs: int = 1,
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
        # initiating features
        features = Features(
            quantitatives=quantitatives,
            categoricals=categoricals,
            ordinals=ordinals,
            ordinal_values=ordinal_values,
            **kwargs,
        )
        super().__init__(features=features, **kwargs)  # Initiating BaseDiscretizer
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
                categoricals=self.features.categoricals,
                ordinals=self.features.ordinals,
                min_freq=self.min_freq,
                copy=False,  # always False as x_copy is already a copy (if requested)
                verbose=self.verbose,
                n_jobs=self.n_jobs,
            )
            qualitative_discretizer.fit(x_copy, y)

            # saving kept features
            kept_features += qualitative_discretizer.features.get_names()

        # [Quantitative features] Grouping quantitative features
        if len(self.features.get_quantitatives()) > 0:

            # grouping quantitative features
            quantitative_discretizer = QuantitativeDiscretizer(
                quantitatives=self.features.quantitatives,
                min_freq=self.min_freq,
                verbose=self.verbose,
                n_jobs=self.n_jobs,
            )
            quantitative_discretizer.fit(x_copy, y)

            # saving kept features
            kept_features += quantitative_discretizer.features.get_names()

        # removing dropped features
        self.features.keep_features(kept_features)

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
        categoricals: list[str] = None,
        ordinals: list[str] = None,
        ordinal_values: dict[str, GroupedList] = None,
        # *,
        # copy: bool = False,
        # verbose: bool = False,
        # n_jobs: int = 1,
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
        # initiating features
        features = Features(
            categoricals=categoricals, ordinals=ordinals, ordinal_values=ordinal_values, **kwargs
        )
        super().__init__(features=features, **kwargs)  # Initiating BaseDiscretizer
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

        # getting feature values' frequencies
        max_frequencies = x_copy[self.features.get_names()].apply(
            lambda u: u.value_counts(normalize=True, dropna=False).max(),
            axis=0,
        )
        # features with no common modality (biggest value less frequent than min_freq)
        non_common = [f.name for f in self.features if max_frequencies[f.name] < self.min_freq]
        # features with too common modality (biggest value more frequent than 1-min_freq)
        too_common = [f.name for f in self.features if max_frequencies[f.name] > 1 - self.min_freq]
        # raising
        if len(too_common + non_common) > 0:

            # building error message
            error_msg = (
                f" - [{self.__name__}] Features {str(too_common + non_common)} contain a too "
                "frequent modality or no frequent enough modality. Consider decreasing min_freq or removing"
                " these feature.\nINFO:\n"
            )
            # adding features with no common values
            error_msg += "\n".join(
                [
                    (
                        f" - {self.features(f)}: most frequent value has "
                        f"freq={max_frequencies[f]:2.2%} < min_freq={self.min_freq:2.2%}"
                    )
                    for f in non_common
                ]
            )
            # adding features with too common values
            error_msg += "\n".join(
                [
                    (
                        f" - {self.features(f)}: most frequent value has "
                        f"freq={max_frequencies[f]:2.2%} > (1-min_freq)={1-self.min_freq:2.2%}"
                    )
                    for f in too_common
                ]
            )

            raise ValueError(error_msg)

        # checking for columns containing floats or integers even with filled nans
        dtypes = (
            x_copy.fillna({f.name: f.nan for f in self.features if f.has_nan})[
                self.features.get_names()
            ]
            .map(type)
            .apply(unique, result_type="reduce")
        )
        not_object = dtypes.apply(lambda u: any(typ != str for typ in u))

        # converting detected non-string features
        if any(not_object):
            # converting specified features into qualitative features
            string_discretizer = StringDiscretizer(
                features=[
                    feature
                    for feature in self.features
                    if feature.name in not_object.index[not_object]
                ],
                **self.kwargs,
            )
            x_copy = string_discretizer.fit_transform(x_copy, y)

        return x_copy

    @extend_docstring(BaseDiscretizer.fit)
    def fit(self, X: DataFrame, y: Series) -> None:  # pylint: disable=W0222
        if self.verbose:  # verbose if requested
            print("------")
            self._verbose("---")

        # checking data before bucketization
        x_copy = self._prepare_data(X, y)

        # Base discretization (useful if already discretized)
        base_discretizer = BaseDiscretizer(
            features=[feature for feature in self.features if feature.is_fitted],
            dropna=False,
            copy=True,
            verbose=self.verbose,
            n_jobs=self.n_jobs,
        )
        x_copy = base_discretizer.fit_transform(x_copy, y)

        # [Qualitative ordinal features] Grouping rare values into closest common one
        if len(self.features.ordinals) > 0:
            ordinal_discretizer = OrdinalDiscretizer(
                ordinals=self.features.ordinals,
                min_freq=self.min_freq,
                verbose=self.verbose,
                copy=False,
                n_jobs=self.n_jobs,
            )
            ordinal_discretizer.fit(x_copy, y)

        # [Qualitative non-ordinal features] Grouping rare values into str_default '__OTHER__'
        if len(self.features.categoricals) > 0:
            default_discretizer = CategoricalDiscretizer(
                categoricals=self.features.categoricals,
                min_freq=self.min_freq,
                verbose=self.verbose,
                copy=False,
                n_jobs=self.n_jobs,
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
        quantitatives: list[str],
        min_freq: float,
        # *,
        # verbose: bool = False,
        # copy: bool = False,
        # n_jobs: int = 1,
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
        features = Features(quantitatives=quantitatives, **kwargs)  # initiating features
        super().__init__(features=features, **kwargs)  # Initiating BaseDiscretizer
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
        dtypes = x_copy[self.features.get_names()].map(type).apply(unique, result_type="reduce")
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
        if self.verbose:  # verbose if requested
            print("------")
            self._verbose("---")

        # checking data before bucketization
        x_copy = self._prepare_data(X, y)

        # [Quantitative features] Grouping values into quantiles
        continuous_discretizer = ContinuousDiscretizer(
            quantitatives=self.features.quantitatives,
            min_freq=self.min_freq,
            copy=True,  # needs to be True so that it does not transform x_copy
            verbose=self.verbose,
            n_jobs=self.n_jobs,
        )

        x_copy = continuous_discretizer.fit_transform(x_copy, y)

        # [Quantitative features] Grouping rare quantiles into closest common one
        #  -> can exist because of overrepresented values (values more frequent than min_freq)
        # searching for features with rare quantiles: computing min frequency per feature
        frequencies = x_copy[self.features.get_names()].apply(
            min_value_counts, features=self.features, axis=0
        )

        # minimal frequency of a quantile
        q_min_freq = self.min_freq / 2

        # identifying features that have rare modalities
        has_rare = list(frequencies[frequencies <= q_min_freq].index)

        # Grouping rare modalities
        if len(has_rare) > 0:
            ordinal_discretizer = OrdinalDiscretizer(
                ordinals=[feature for feature in self.features if feature.name in has_rare],
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
