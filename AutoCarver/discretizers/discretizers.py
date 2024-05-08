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
        quantitative_features: list[str],
        qualitative_features: list[str],
        min_freq: float,
        *,
        ordinal_features: list[str] = None,
        values_orders: dict[str, GroupedList] = None,
        copy: bool = False,
        verbose: bool = False,
        n_jobs: int = 1,
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
        # Lists of features per type
        if ordinal_features is None:
            ordinal_features = []
        self.ordinal_features = list(set(ordinal_features))
        self.features = list(set(quantitative_features + qualitative_features + ordinal_features))

        # initializing input_dtypes
        self.input_dtypes = {feature: "str" for feature in qualitative_features + ordinal_features}
        self.input_dtypes.update({feature: "float" for feature in quantitative_features})

        # Initiating BaseDiscretizer
        super().__init__(
            features=self.features,
            values_orders=values_orders,
            input_dtypes=self.input_dtypes,
            output_dtype="str",
            str_nan=kwargs.get("nan", NAN),
            str_default=kwargs.get("default", DEFAULT),
            copy=copy,
            verbose=verbose,
            n_jobs=n_jobs,
        )

        # checking for missing orders
        no_order_provided = [
            feature for feature in self.ordinal_features if feature not in self.values_orders
        ]
        assert len(no_order_provided) == 0, (
            " - [Discretizer] No ordering was provided for following features: "
            f"{str(no_order_provided)}. Please make sure you defined values_orders correctly."
        )

        # class specific attributes
        self.min_freq = min_freq

    def _remove_feature(self, feature: str) -> None:
        """Removes a feature from all ``feature`` attributes

        Parameters
        ----------
        feature : str
            Column name of the feature to remove
        """
        if feature in self.features:
            super()._remove_feature(feature)
            if feature in self.ordinal_features:
                self.ordinal_features.remove(feature)

    @extend_docstring(BaseDiscretizer.fit)
    def fit(self, X: DataFrame, y: Series) -> None:  # pylint: disable=W0222
        # Checking for binary target and copying X
        x_copy = self._prepare_data(X, y)

        # [Qualitative features] Grouping qualitative features
        if len(self.qualitative_features) > 0:
            if self.verbose:  # verbose if requested
                print("------\n[Discretizer] Fit Qualitative Features\n---")

            # grouping qualitative features
            qualitative_discretizer = QualitativeDiscretizer(
                qualitative_features=self.qualitative_features,
                ordinal_features=self.ordinal_features,
                min_freq=self.min_freq,
                values_orders=self.values_orders,
                input_dtypes=self.input_dtypes,
                str_nan=self.str_nan,
                str_default=self.str_default,
                copy=False,  # always False as x_copy is already a copy (if requested)
                verbose=self.verbose,
                n_jobs=self.n_jobs,
            )
            qualitative_discretizer.fit(x_copy, y)

            # storing orders of grouped features
            self.values_orders.update(qualitative_discretizer.values_orders)

            # removing dropped features
            removed_features = [
                feature
                for feature in self.qualitative_features
                if feature not in qualitative_discretizer.features
            ]
            for feature in removed_features:
                self._remove_feature(feature)

            if self.verbose:  # verbose if requested
                print("------\n")

        # [Quantitative features] Grouping quantitative features
        if len(self.quantitative_features) > 0:
            if self.verbose:  # verbose if requested
                print("------\n[Discretizer] Fit Quantitative Features\n---")

            # grouping quantitative features
            quantitative_discretizer = QuantitativeDiscretizer(
                quantitative_features=self.quantitative_features,
                min_freq=self.min_freq,
                values_orders=self.values_orders,
                input_dtypes=self.input_dtypes,
                str_nan=self.str_nan,
                verbose=self.verbose,
                n_jobs=self.n_jobs,
            )
            quantitative_discretizer.fit(x_copy, y)

            # storing orders of grouped features
            self.values_orders.update(quantitative_discretizer.values_orders)

            # removing dropped features
            removed_features = [
                feature
                for feature in self.quantitative_features
                if feature not in quantitative_discretizer.features
            ]
            for feature in removed_features:
                self._remove_feature(feature)

            if self.verbose:  # verbose if requested
                print("------\n")

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

        # checking for columns without any common modality
        all_features = self.features.get_names()  # prevents inplace removal to break loop
        max_frequencies = x_copy[all_features].apply(
            lambda u: u.value_counts(normalize=True, dropna=False).drop(nan, errors="ignore").max(),
            axis=0,
        )
        # for each feature, checking that at least one value is more frequent than min_freq
        for feature in all_features:
            # no common modality found
            if max_frequencies[feature] < self.min_freq:
                warn(
                    f" - [{self.__name__}] For feature '{feature}', the largest modality"
                    f" has {max_frequencies[feature]:2.2%} observations which is lower than "
                    f"min_freq={self.min_freq:2.2%}. This feature will not be Discretized. "
                    "Consider decreasing min_freq or removing this feature.",
                    UserWarning,
                )
                self.features.remove(feature)

        # checking for columns containing floats or integers even with filled nans
        dtypes = (
            x_copy.fillna({f.name: f.nan for f in self.features if f.has_nan})[
                self.features.get_names()
            ]
            .map(type)
            .apply(unique, result_type="reduce")
        )
        not_object = dtypes.apply(lambda u: any(typ != str for typ in u))

        # non-qualitative features detected
        if any(not_object):
            features_to_convert = list(not_object.index[not_object])
            unexpected_dtypes = [typ for dtyp in dtypes[not_object] for typ in dtyp if typ != str]
            warn(
                f" - [{self.__name__}] Non-string features: {str(features_to_convert)}. "
                "Converting them with type_discretizers.StringDiscretizer. "
                f"Unexpected data types: {str(set(unexpected_dtypes))}.",
                UserWarning,
            )

            # converting specified features into qualitative features
            string_discretizer = StringDiscretizer(
                features=[
                    feature for feature in self.features if feature.name in features_to_convert
                ],
                **self.kwargs,
            )
            x_copy = string_discretizer.fit_transform(x_copy, y)

        return x_copy

    @extend_docstring(BaseDiscretizer.fit)
    def fit(self, X: DataFrame, y: Series) -> None:  # pylint: disable=W0222
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
