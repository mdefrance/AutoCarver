"""Tools to build simple buckets out of Quantitative and Qualitative features
for a binary classification model.
"""

from typing import Union

from numpy import nan
from pandas import DataFrame, Series, unique

from .utils.base_discretizers import BaseDiscretizer, min_value_counts
from .utils.grouped_list import GroupedList
from .utils.qualitative_discretizers import DefaultDiscretizer, OrdinalDiscretizer
from .utils.quantitative_discretizers import QuantileDiscretizer
from .utils.type_discretizers import StringDiscretizer


class Discretizer(BaseDiscretizer):
    """Automatic discretization pipeline of continuous, discrete, categorical and ordinal features.

    Pipeline steps: :ref:`QuantitativeDiscretizer`, :ref:`QualitativeDiscretizer`.

    Modalities/values of features are grouped according to there respective orders:

    * [Qualitative features] order based on modality target rate.
    * [Ordinal features] user-specified order.
    * [Quantitative features] real order of the values.
    """

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
        str_nan: str = "__NAN__",
        str_default: str = "__OTHER__",
    ) -> None:
        """Initiates a ``Discretizer`` (pipeline).

        Parameters
        ----------
        quantitative_features : list[str]
            List of column names of quantitative features (continuous and discrete) to be dicretized

        qualitative_features : list[str]
            List of column names of qualitative features (non-ordinal) to be discretized

        min_freq : float
            Minimum frequency per grouped modalities.

            * Features whose most frequent modality is less frequent than `min_freq` will not be discretized.
            * Sets the number of quantiles in which to discretize the continuous features.
            * Sets the minimum frequency of a quantitative feature's modality.

            **Tip**: should be set between 0.02 (slower, preciser, less robust) and 0.05 (faster, more robust)

        ordinal_features : list[str], optional
            List of column names of ordinal features to be discretized. For those features a list of values has to be provided in the `values_orders` dict, by default None

        values_orders : dict[str, GroupedList], optional
            Dict of feature's column names and there associated ordering.
            If lists are passed, a GroupedList will automatically be initiated, by default None

        copy : bool, optional
            If `copy=True`, feature processing at transform is applied to a copy of the provided DataFrame, by default False

        verbose : bool, optional
            If `verbose=True`, prints raw Discretizers Fit and Transform steps, by default False

        str_nan : str, optional
            String representation to input `numpy.nan`. If `dropna=False`, `numpy.nan` will be left unchanged, by default "__NAN__"

        str_default : str, optional
            String representation for default qualitative values, i.e. values less frequent than `min_freq`, by default "__OTHER__"

        Examples
        --------
        See `AutoCarver examples <https://autocarver.readthedocs.io/en/latest/index.html>`_
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
            str_nan=str_nan,
            str_default=str_default,
            copy=copy,
            verbose=verbose,
        )

        # checking for missing orders
        no_order_provided = [
            feature for feature in self.ordinal_features if feature not in self.values_orders
        ]
        assert (
            len(no_order_provided) == 0
        ), f" - [Discretizer] No ordering was provided for following features: {str(no_order_provided)}. Please make sure you defined ``values_orders`` correctly."

        # class specific attributes
        self.min_freq = min_freq

    def _remove_feature(self, feature: str) -> None:
        """Removes a feature from all `feature` attributes

        Parameters
        ----------
        feature : str
            Column name of the feature to remove
        """
        if feature in self.features:
            super()._remove_feature(feature)
            if feature in self.ordinal_features:
                self.ordinal_features.remove(feature)

    def fit(self, X: DataFrame, y: Series) -> None:
        """Finds relevant buckets of modalities of X to provide the best association with y.

        Parameters
        ----------
        X : DataFrame
            Dataset used to discretize. Needs to have columns has specified in `features`.

        y : Series
            Binary target feature.
        """
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

    Pipeline steps: :ref:`DefaultDiscretizer`, :ref:`StringDiscretizer`, :ref:`OrdinalDiscretizer`.

    Modalities/values of features are grouped according to there respective orders:

    * [Qualitative features] order based on modality target rate.
    * [Ordinal features] user-specified order.
    """

    def __init__(
        self,
        qualitative_features: list[str],
        min_freq: float,
        *,
        ordinal_features: list[str] = None,
        values_orders: dict[str, GroupedList] = None,
        input_dtypes: Union[str, dict[str, str]] = "str",
        copy: bool = False,
        verbose: bool = False,
        str_nan: str = "__NAN__",
        str_default: str = "__OTHER__",
    ) -> None:
        """Initiates a ``QualitativeDiscretizer`` (pipeline).

        Parameters
        ----------
        qualitative_features : list[str]
            List of column names of qualitative features (non-ordinal) to be discretized

        min_freq : float
            Minimum frequency per grouped modalities.

            * Features whose most frequent modality is less frequent than ``min_freq`` will not be discretized.
            * Sets the number of quantiles in which to discretize the continuous features.
            * Sets the minimum frequency of a quantitative feature's modality.

            **Tip**: should be set between 0.02 (slower, preciser, less robust) and 0.05 (faster, more robust)

        ordinal_features : list[str], optional
            List of column names of ordinal features to be discretized. For those features a list of values has to be provided in the ``values_orders`` dict, by default ``None``

        values_orders : dict[str, GroupedList], optional
            Dict of feature's column names and there associated ordering.
            If lists are passed, a GroupedList will automatically be initiated, by default ``None``

        input_dtypes : Union[str, dict[str, str]], optional
            Input data type, converted to a dict of the provided type for each feature, by default ``"str"``

            * If ``"str"``, features are considered as qualitative.
            * If ``"float"``, features are considered as quantitative.

        copy : bool, optional
            If ``True``, feature processing at transform is applied to a copy of the provided DataFrame, by default ``False``

        verbose : bool, optional
            If ``True``, prints raw Discretizers Fit and Transform steps, by default ``False``

        str_nan : str, optional
            String representation to input ``numpy.nan``. If ``dropna=False``, ``numpy.nan`` will be left unchanged, by default ``"__NAN__"``

        str_default : str, optional
            String representation for default qualitative values, i.e. values less frequent than ``min_freq``, by default ``"__OTHER__"``

        Examples
        --------
        See `AutoCarver examples <https://autocarver.readthedocs.io/en/latest/index.html>`_
        """
        # Lists of features
        if ordinal_features is None:
            ordinal_features = []
        self.ordinal_features = list(set(ordinal_features))
        self.features = list(set(qualitative_features + ordinal_features))

        # class specific attributes
        self.min_freq = min_freq

        # Initiating BaseDiscretizer
        super().__init__(
            features=self.features,
            values_orders=values_orders,
            input_dtypes=input_dtypes,
            output_dtype="str",
            str_nan=str_nan,
            str_default=str_default,
            copy=copy,
            verbose=verbose,
        )

        # checking for missing orders
        no_order_provided = [
            feature for feature in self.ordinal_features if feature not in self.values_orders
        ]
        assert (
            len(no_order_provided) == 0
        ), f" - [QualitativeDiscretizer] No ordering was provided for following features: {str(no_order_provided)}. Please make sure you defined ``values_orders`` correctly."

        # non-ordinal qualitative features
        self.non_ordinal_features = [
            feature for feature in self.qualitative_features if feature not in self.ordinal_features
        ]

    def _prepare_data(self, X: DataFrame, y: Series = None) -> DataFrame:
        """Validates format and content of X and y. Converts non-string columns into strings.

        Parameters
        ----------
        X : DataFrame
            Dataset used to discretize. Needs to have columns has specified in ``QualitativeDiscretizer.features``.

        y : Series
            Binary target feature with wich the association is maximized.

        Returns
        -------
        DataFrame
            A formatted copy of X
        """
        # checking for binary target, copying X
        x_copy = super()._prepare_data(X, y)

        # checking for ids (unique value per row)
        max_frequencies = x_copy[self.features].apply(
            lambda u: u.value_counts(normalize=True, dropna=False).drop(nan, errors="ignore").max(),
            axis=0,
        )
        # for each feature, checking that at least one value is more frequent than min_freq
        all_features = self.features[:]  # features are being removed from self.features
        for feature in all_features:
            if max_frequencies[feature] < self.min_freq:
                print(
                    f" - [QualitatitveDiscretizer] For feature '{feature}', the largest modality has {max_frequencies[feature]:2.2%} observations which is lower than min_freq={self.min_freq:2.1%}. This feature will not be Discretized. Consider decreasing parameter min_freq or removing this feature."
                )
                self._remove_feature(feature)

        # checking for columns containing floats or integers even with filled nans
        dtypes = x_copy[self.features].fillna(self.str_nan).applymap(type).apply(unique)
        not_object = dtypes.apply(lambda u: any(typ != str for typ in u))

        # non qualitative features detected
        if any(not_object):
            features_to_convert = list(not_object.index[not_object])
            if self.verbose:
                unexpected_dtypes = [
                    typ for dtyp in dtypes[not_object] for typ in dtyp if typ != str
                ]
                print(
                    f""" - [QualitatitveDiscretizer] Non-string features: {str(features_to_convert)}. Trying to convert them using type_discretizers.StringDiscretizer, otherwise convert them manually. Unexpected data types: {str(list(unexpected_dtypes))}."""
                )

            # converting specified features into qualitative features
            string_discretizer = StringDiscretizer(
                qualitative_features=features_to_convert,
                values_orders=self.values_orders,
                verbose=self.verbose,
            )
            x_copy = string_discretizer.fit_transform(x_copy)

            # updating values_orders accordingly
            self.values_orders.update(string_discretizer.values_orders)

        # adding known nans at training
        for feature in self.features:
            if feature in self.values_orders:
                order = self.values_orders[feature]
                if any(x_copy[feature].isna()) and (self.str_nan not in order):
                    order.append(self.str_nan)
                    self.values_orders.update({feature: order})

        # checking that all unique values in X are in values_orders
        self._check_new_values(x_copy, features=self.ordinal_features)

        return x_copy

    def _remove_feature(self, feature: str) -> None:
        """Removes a feature from all `feature` attributes

        Parameters
        ----------
        feature : str
            Column name of the feature to remove
        """
        if feature in self.features:
            super()._remove_feature(feature)
            if feature in self.ordinal_features:
                self.ordinal_features.remove(feature)
            if feature in self.non_ordinal_features:
                self.non_ordinal_features.remove(feature)

    def fit(self, X: DataFrame, y: Series) -> None:
        """Finds relevant buckets of modalities of X to provide the best association with y.

        Parameters
        ----------
        X : DataFrame
            Dataset used to discretize. Needs to have columns has specified in ``QualitativeDiscretizer.features``.

        y : Series
            Binary target feature.
        """
        # checking data before bucketization
        x_copy = self._prepare_data(X, y)

        # Base discretization (useful if already discretized)
        base_discretizer = BaseDiscretizer(
            features=[feature for feature in self.features if feature in self.values_orders],
            values_orders=self.values_orders,
            input_dtypes="str",
            output_dtype="str",
            dropna=False,
            copy=True,
            verbose=self.verbose,
            str_nan="__NAN__",
            str_default="__OTHER__",
        )
        x_copy = base_discretizer.fit_transform(x_copy, y)

        # [Qualitative ordinal features] Grouping rare values into closest common one
        if len(self.ordinal_features) > 0:
            ordinal_discretizer = OrdinalDiscretizer(
                ordinal_features=self.ordinal_features,
                values_orders=self.values_orders,
                input_dtypes=self.input_dtypes,
                min_freq=self.min_freq,
                verbose=self.verbose,
                str_nan=self.str_nan,
                copy=False,
            )
            ordinal_discretizer.fit(x_copy, y)

            # storing orders of grouped features
            self.values_orders.update(ordinal_discretizer.values_orders)

        # [Qualitative non-ordinal features] Grouping rare values into str_default '__OTHER__'
        if len(self.non_ordinal_features) > 0:
            default_discretizer = DefaultDiscretizer(
                qualitative_features=self.non_ordinal_features,
                min_freq=self.min_freq,
                values_orders=self.values_orders,
                str_nan=self.str_nan,
                str_default=self.str_default,
                verbose=self.verbose,
                copy=False,
            )
            default_discretizer.fit(x_copy, y)

            # storing orders of grouped features
            self.values_orders.update(default_discretizer.values_orders)

        # discretizing features based on each feature's values_order
        super().fit(X, y)

        return self


class QuantitativeDiscretizer(BaseDiscretizer):
    """Automatic discretization pipeline of continuous and discrete features.

    Pipeline steps: :ref:`QuantileDiscretizer`, :ref:`OrdinalDiscretizer`

    Modalities/values of features are grouped according to there respective orders:

     * [Quantitative features] real order of the values.
    """

    def __init__(
        self,
        quantitative_features: list[str],
        min_freq: float,
        *,
        values_orders: dict[str, GroupedList] = None,
        input_dtypes: Union[str, dict[str, str]] = "float",
        verbose: bool = False,
        copy: bool = False,
        str_nan: str = "__NAN__",
    ) -> None:
        """Initiates a ``QuantitativeDiscretizer`` (pipeline).

        Parameters
        ----------
        quantitative_features : list[str]
            List of column names of quantitative features (continuous and discrete) to be dicretized

        min_freq : float
            Minimum frequency per grouped modalities.

            * Features whose most frequent modality is less frequent than ``min_freq`` will not be discretized.
            * Sets the number of quantiles in which to discretize the continuous features.
            * Sets the minimum frequency of a quantitative feature's modality.

            **Tip**: should be set between 0.02 (slower, preciser, less robust) and 0.05 (faster, more robust)

        values_orders : dict[str, GroupedList], optional
            Dict of feature's column names and there associated ordering.
            If lists are passed, a GroupedList will automatically be initiated, by default ``None``

        input_dtypes : Union[str, dict[str, str]], optional
            Input data type, converted to a dict of the provided type for each feature, by default ``"str"``

            * If ``"str"``, features are considered as qualitative.
            * If ``"float"``, features are considered as quantitative.

        copy : bool, optional
            If ``True``, feature processing at transform is applied to a copy of the provided DataFrame, by default ``False``

        verbose : bool, optional
            If ``True``, prints raw Discretizers Fit and Transform steps, by default ``False``

        str_nan : str, optional
            String representation to input ``numpy.nan``. If ``dropna=False``, ``numpy.nan`` will be left unchanged, by default ``"__NAN__"``

        Examples
        --------
        See `AutoCarver examples <https://autocarver.readthedocs.io/en/latest/index.html>`_
        """
        # Initiating BaseDiscretizer
        super().__init__(
            features=quantitative_features,
            values_orders=values_orders,
            input_dtypes=input_dtypes,
            output_dtype="str",
            str_nan=str_nan,
            copy=copy,
            verbose=verbose,
        )

        # class specific attributes
        self.min_freq = min_freq

    def _prepare_data(self, X: DataFrame, y: Series) -> DataFrame:
        """Validates format and content of X and y.

        Parameters
        ----------
        X : DataFrame
            Dataset used to discretize. Needs to have columns has specified in ``QuantitativeDiscretizer.features``.

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
        dtypes = x_copy[self.features].applymap(type).apply(unique)
        not_numeric = dtypes.apply(lambda u: str in u)
        assert all(
            ~not_numeric
        ), f" - [QuantitativeDiscretizer] Non-numeric features: {str(list(not_numeric[not_numeric].index))} in provided quantitative_features. Please check your inputs."

        return x_copy

    def fit(self, X: DataFrame, y: Series) -> None:
        """Finds relevant buckets of modalities of X to provide the best association with y.

        Parameters
        ----------
        X : DataFrame
            Dataset used to discretize. Needs to have columns has specified in ``QuantitativeDiscretizer.features``.

        y : Series
            Binary target feature.
        """
        # checking data before bucketization
        x_copy = self._prepare_data(X, y)

        # [Quantitative features] Grouping values into quantiles
        quantile_discretizer = QuantileDiscretizer(
            quantitative_features=self.features,
            min_freq=self.min_freq,
            values_orders=self.values_orders,
            str_nan=self.str_nan,
            copy=True,  # needs to be True so that it does not transform x_copy
            verbose=self.verbose,
        )
        x_copy = quantile_discretizer.fit_transform(x_copy, y)

        # storing orders of grouped features
        self.values_orders.update(quantile_discretizer.values_orders)

        # [Quantitative features] Grouping rare quantiles into closest common one
        #  -> can exist because of overrepresented values (values more frequent than min_freq)
        # searching for features with rare quantiles: computing min frequency per feature
        frequencies = x_copy[self.features].apply(
            min_value_counts, values_orders=self.values_orders, axis=0
        )

        # minimal frequency of a quantile
        q_min_freq = self.min_freq / 2

        # identifying features that have rare modalities
        has_rare = list(frequencies[frequencies <= q_min_freq].index)

        # Grouping rare modalities
        if len(has_rare) > 0:
            ordinal_discretizer = OrdinalDiscretizer(
                ordinal_features=has_rare,
                min_freq=self.min_freq,
                values_orders=self.values_orders,
                str_nan=self.str_nan,
                copy=False,
                verbose=self.verbose,
                input_dtypes=self.input_dtypes,
            )
            ordinal_discretizer.fit(x_copy, y)

            # storing orders of grouped features
            self.values_orders.update(ordinal_discretizer.values_orders)

        # discretizing features based on each feature's values_order
        super().fit(X, y)

        return self
