"""Tools to build simple buckets out of Quantitative and Qualitative features
for a binary classification model.
"""

from typing import Any, Dict, List, Union
from numpy import nan
from pandas import DataFrame, Series, unique

from .utils.base_discretizers import (
    GroupedList,
    GroupedListDiscretizer,
    check_new_values,
    min_value_counts,
)
from .utils.qualitative_discretizers import DefaultDiscretizer, OrdinalDiscretizer
from .utils.quantitative_discretizers import QuantileDiscretizer
from .utils.type_discretizers import StringDiscretizer


class Discretizer(GroupedListDiscretizer):
    """Automatic discretizing of continuous, categorical and categorical ordinal features.

    Modalities/values of features are grouped according to there respective orders:
     - [Qualitative features] order based on modality target rate.
     - [Qualitative ordinal features] user-specified order.
     - [Quantitative features] real order of the values.

    Parameters
    ----------
    quantitative_features: list
        Contains quantitative (continuous) features to be discretized.

    qualitative_features: list
        Contains qualitative (categorical and categorical ordinal) features to be discretized.

    min_freq: int, default None
        [Qualitative features] Minimal frequency of a modality.
         - NaNs are considered a specific modality but will not be grouped.
         - [Qualitative features] Less frequent modalities are grouped in the `__OTHER__` modality.
         - [Qualitative ordinal features] Less frequent modalities are grouped to the closest modality
        (smallest frequency or closest target rate), between the superior and inferior values (specified
        in the `values_orders` dictionnary).
        Recommandation: `min_freq` should be set from 0.01 (preciser) to 0.05 (faster, increased stability).

    q: int, default None
        [Quantitative features] Number of quantiles to initialy cut the feature.
         - NaNs are considered a specific value but will not be grouped.
         - Values more frequent than `1/q` will be set as their own group and remaining frequency will be
        cut into proportionaly less quantiles (`q:=max(round(non_frequent * q), 1)`).
        Exemple: if q=10 and the value numpy.nan represent 50 % of the observed values, non-NaNs will be
        cut in q=5 quantiles.
        Recommandation: `q` should be set from 10 (faster) to 20 (preciser).

    values_orders: dict, default {}
        [Qualitative ordinal features] dict of features values and list of orders of their values.
         - [Qualitative ordinal features] Less frequent modalities are grouped to the closest modality
        (smallest frequency or closest target rate), between the superior and inferior values (described
        by the `values_orders`).
        Exemple: for an `age` feature, `values_orders` could be `{'age': ['0-18', '18-30', '30-50', '50+']}`.
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
        """_summary_

        Parameters
        ----------
        quantitative_features : list[str]
            _description_
        qualitative_features : list[str]
            _description_
        min_freq : float
            _description_
        ordinal_features : list[str], optional
            _description_, by default None
        values_orders : dict[str, GroupedList], optional
            _description_, by default None
        input_dtypes : Union[str, dict[str, str]], optional
            String of type to be considered for all features or
            Dict of column names and associated types:
            - if 'float' uses transform_quantitative
            - if 'str' uses transform_qualitative,
            default 'str'
        copy : bool, optional
            _description_, by default False
        verbose : bool, optional
            _description_, by default False
        str_nan : str, optional
            _description_, by default '__NAN__'
        str_default : str, optional
            _description_, by default '__OTHER__'
        """
        # Lists of features per type
        self.features = list(set(quantitative_features + qualitative_features + ordinal_features))
        if ordinal_features is None:
            ordinal_features = []
        self.ordinal_features = list(set(ordinal_features))

        # initializing input_dtypes
        self.input_dtypes = {feature: "str" for feature in qualitative_features + ordinal_features}
        self.input_dtypes.update({feature: "float" for feature in quantitative_features})

        # Initiating GroupedListDiscretizer
        super().__init__(
            features=self.features,
            values_orders=values_orders,
            input_dtypes=self.input_dtypes,
            output_dtype='str',
            str_nan=str_nan,
            copy=copy,
        )

        # class specific attributes
        self.min_freq = min_freq
        self.str_default = str_default
        self.verbose = verbose
    
    def remove_feature(self, feature: str) -> None:
        """Removes a feature from the Discretizer

        Parameters
        ----------
        feature : str
            Column name of the feature
        """
        if feature in self.features:
            super().remove_feature(feature)
            if feature in self.ordinal_features:
                self.ordinal_features.remove(feature)

    def fit(self, X: DataFrame, y: Series) -> None:
        """_summary_

        Parameters
        ----------
        X : DataFrame
            _description_
        y : Series
            _description_
        """
        # [Qualitative features] Grouping qualitative features
        if len(self.qualitative_features) > 0:
            if self.verbose:  # verbose if requested
                print("\n---\n[Discretizer] Fit Qualitative Features")

            # grouping qualitative features
            discretizer = QualitativeDiscretizer(
                qualitative_features=self.qualitative_features,
                ordinal_features=self.ordinal_features,
                min_freq=self.min_freq,
                values_orders=self.values_orders,
                input_dtypes=self.input_dtypes,
                str_nan=self.str_nan,
                str_default=self.str_default,
                copy=self.copy,
                verbose=self.verbose,
            )
            discretizer.fit(X, y)

            # storing orders of grouped features
            self.values_orders.update(discretizer.values_orders)

            # removing dropped features
            for feature in self.qualitative_features:
                if feature not in discretizer.values_orders:
                    self.remove_feature(feature)

        # [Quantitative features] Grouping quantitative features
        if len(self.quantitative_features) > 0:
            if self.verbose:  # verbose if requested
                print("\n---\n[Discretizer] Fit Quantitative Features")

            # grouping quantitative features
            discretizer = QuantitativeDiscretizer(
                quantitative_features=self.quantitative_features,
                min_freq=self.min_freq,
                values_orders=self.values_orders,
                input_dtypes=self.input_dtypes,
                str_nan=self.str_nan,
                verbose=self.verbose,
            )
            discretizer.fit(X, y)

            # storing orders of grouped features
            self.values_orders.update(discretizer.values_orders)

            # removing dropped features
            for feature in self.quantitative_features:
                if feature not in discretizer.values_orders:
                    self.remove_feature(feature)

        # discretizing features based on each feature's values_order
        super().fit(X, y)

        return self


class QualitativeDiscretizer(GroupedListDiscretizer):
    """Automatic discretizing of categorical and categorical ordinal features.

    Modalities/values of features are grouped according to there respective orders:
     - [Qualitative features] order based on modality target rate.
     - [Qualitative ordinal features] user-specified order.

    Parameters
    ----------
    features: list
        Contains qualitative (categorical and categorical ordinal) features to be discretized.

    min_freq: int
        [Qualitative features] Minimal frequency of a modality.
         - NaNs are considered a specific modality but will not be grouped.
         - [Qualitative features] Less frequent modalities are grouped in the `__OTHER__` modality.
         - [Qualitative ordinal features] Less frequent modalities are grouped to the closest modality
        (smallest frequency or closest target rate), between the superior and inferior values (specified
        in the `values_orders` dictionnary).
        Recommandation: `min_freq` should be set from 0.01 (preciser) to 0.05 (faster, increased stability).

    values_orders: dict, default {}
        [Qualitative ordinal features] dict of features values and list of orders of their values.
         - [Qualitative ordinal features] Less frequent modalities are grouped to the closest modality
        (smallest frequency or closest target rate), between the superior and inferior values (described
        by the `values_orders`).
        Exemple: for an `age` feature, `values_orders` could be `{'age': ['0-18', '18-30', '30-50', '50+']}`.
    """

    def __init__(
        self,
        qualitative_features: list[str],
        min_freq: float,
        *,
        ordinal_features: list[str] = None,
        values_orders: dict[str, GroupedList] = None,
        input_dtypes: Union[str, dict[str, str]] = "str",
        str_nan: str = "__NAN__",
        str_default: str = "__OTHER__",
        copy: bool = False,
        verbose: bool = False,
    ) -> None:
        """_summary_

        Parameters
        ----------
        qualitative_features : list[str]
            _description_
        min_freq : float
            _description_
        ordinal_features : list[str], optional
            _description_, by default None
        values_orders : dict[str, GroupedList], optional
            _description_, by default None
        input_dtypes : Union[str, dict[str, str]], optional
            String of type to be considered for all features or
            Dict of column names and associated types:
            - if 'float' uses transform_quantitative
            - if 'str' uses transform_qualitative,
            default 'str'
        str_nan : str, optional
            _description_, by default '__NAN__'
        copy : bool, optional
            _description_, by default False
        verbose : bool, optional
            _description_, by default False
        """
        # Lists of features
        self.features = list(set(qualitative_features + ordinal_features))
        if ordinal_features is None:
            ordinal_features = []
        self.ordinal_features = list(set(ordinal_features))

        # class specific attributes
        self.min_freq = min_freq
        self.str_default = str_default
        self.verbose = verbose

        # Initiating GroupedListDiscretizer
        super().__init__(
            features=self.features,
            values_orders=values_orders,
            input_dtypes=input_dtypes,
            output_dtype='str',
            str_nan=str_nan,
            copy=copy,
        )

        # non-ordinal qualitative features
        self.non_ordinal_features = [
            feature for feature in self.qualitative_features if feature not in self.ordinal_features
        ]

    def prepare_data(self, X: DataFrame, y: Series = None) -> DataFrame:
        """Prepares the data for bucketization, checks column types.
        Converts non-string columns into strings.

        Parameters
        ----------
        X : DataFrame
            Dataset to be bucketized
        y : Series
            Model target, by default None

        Returns
        -------
        DataFrame
            Formatted X for bucketization
        """
        # checking for binary target, copying X
        x_copy = super().prepare_data(X, y)

        # checking for ids (unique value per row)
        frequencies = x_copy[self.features].apply(
            lambda u: u.value_counts(normalize=True, dropna=False).drop(nan, errors='ignore').max(), axis=0
        )
        # for each feature, checking that at least one value is more frequent than min_freq
        for feature in self.features:
            if frequencies[feature] < self.min_freq:
                print(f"For feature '{feature}', the largest modality has {frequencies[feature]:2.2%} observations which is lower than {self.min_freq:2.2%}. This feature will not be Discretized. Consider decreasing parameter min_freq or removing this feature.")
                self.remove_feature(feature)

        # checking for columns containing floats or integers even with filled nans
        dtypes = x_copy[self.features].fillna(self.str_nan).applymap(type).apply(unique)
        not_object = dtypes.apply(lambda u: any(typ != str for typ in u))

        # non qualitative features detected
        if any(not_object):
            features_to_convert = list(not_object.index[not_object])
            if self.verbose:
                unexpected_dtypes = [typ for dtyp in dtypes[not_object] for typ in dtyp if typ != str]
                print(
                    f"""Non-string features: {str(features_to_convert)}. Trying to convert them using type_discretizers.StringDiscretizer, otherwise convert them manually. Unexpected data types: {str(list(unexpected_dtypes))}."""
                )

            # converting specified features into qualitative features
            stringer = StringDiscretizer(features=features_to_convert, values_orders=self.values_orders)
            x_copy = stringer.fit_transform(x_copy)

            # updating values_orders accordingly
            self.values_orders.update(stringer.values_orders)

        # all known values for features
        known_values = {feature: values.values() for feature, values in self.values_orders.items()}

        # checking that all unique values in X are in values_orders
        check_new_values(x_copy, self.ordinal_features, known_values)

        return x_copy
    
    def remove_feature(self, feature: str) -> None:
        """Removes a feature from the Discretizer

        Parameters
        ----------
        feature : str
            Column name of the feature
        """
        if feature in self.features:
            super().remove_feature(feature)
            if feature in self.ordinal_features:
                self.ordinal_features.remove(feature)
            if feature in self.non_ordinal_features:
                self.non_ordinal_features.remove(feature)

    def fit(self, X: DataFrame, y: Series) -> None:
        """Learning TRAIN distribution

        Parameters
        ----------
        X : DataFrame
            _description_
        y : Series
            _description_

        Returns
        -------
        _type_
            _description_
        """

        # checking data before bucketization
        Xc = self.prepare_data(X, y)

        # [Qualitative ordinal features] Grouping rare values into closest common one
        if len(self.ordinal_features) > 0:
            discretizer = OrdinalDiscretizer(
                features=self.ordinal_features,
                values_orders=self.values_orders,
                min_freq=self.min_freq,
                verbose=self.verbose,
                str_nan=self.str_nan,
            )
            discretizer.fit(Xc, y)

            # storing orders of grouped features
            self.values_orders.update(discretizer.values_orders)

        # [Qualitative non-ordinal features] Grouping rare values into str_default '__OTHER__'
        if len(self.non_ordinal_features) > 0:
            discretizer = DefaultDiscretizer(
                self.non_ordinal_features,
                min_freq=self.min_freq,
                values_orders=self.values_orders,
                str_nan=self.str_nan,
                str_default=self.str_default,
                verbose=self.verbose,
            )
            discretizer.fit(Xc, y)

            # storing orders of grouped features
            self.values_orders.update(discretizer.values_orders)

        # discretizing features based on each feature's values_order
        super().fit(X, y)

        return self


class QuantitativeDiscretizer(GroupedListDiscretizer):
    """Automatic discretizing of continuous features.

    Modalities/values of features are grouped according to there respective orders:
     - [Quantitative features] real order of the values.

    Parameters
    ----------
    features: list
        Contains quantitative (continuous) features to be discretized.

    min_freq: float, default None
        [Quantitative features] Inverse of the number of quantiles `q` to initialy cut the feature.
         - NaNs are considered a specific value but will not be grouped.
         - Values more frequent than `1/q` will be set as their own group and remaining frequency will be
        cut into proportionaly less quantiles (`q:=max(round(non_frequent * q), 1)`).
        Exemple: if q=10 and the value numpy.nan represent 50 % of the observed values, non-NaNs will be
        cut in q=5 quantiles.
        Recommandation: `q` should be set from 10 (faster) to 20 (preciser).

    """

    def __init__(
        self,
        quantitative_features: list[str],
        min_freq: float,
        *,
        values_orders: dict[str, GroupedList] = None,
        input_dtypes: Union[str, dict[str, str]] = "float",
        str_nan: str = "__NAN__",
        verbose: bool = False,
    ) -> None:
        """_summary_

        Parameters
        ----------
        features : list[str]
            _description_
        min_freq : float
            _description_
        values_orders : dict[str, GroupedList], optional
            _description_, by default None
        input_dtypes : Union[str, dict[str, str]], optional
            String of type to be considered for all features or
            Dict of column names and associated types:
            - if 'float' uses transform_quantitative
            - if 'str' uses transform_qualitative,
            default 'str'
        str_nan : str, optional
            _description_, by default '__NAN__'
        copy : bool, optional
            _description_, by default False
        verbose : bool, optional
            _description_, by default False
        """
        # Initiating GroupedListDiscretizer
        super().__init__(
            features=quantitative_features,
            values_orders=values_orders,
            input_dtypes=input_dtypes,
            output_dtype='str',
            str_nan=str_nan,
            copy=True,
        )

        # class specific attributes
        self.min_freq = min_freq
        self.verbose = verbose

    def prepare_data(self, X: DataFrame, y: Series) -> DataFrame:
        """Checking data for bucketization"""
        # checking for binary target and copying X
        x_copy = super().prepare_data(X, y)

        # checking for quantitative columns
        dtypes = x_copy[self.features].applymap(type).apply(unique)
        not_numeric = dtypes.apply(lambda u: str in u)
        assert all(~not_numeric), f"Non-numeric features: {str(list(not_numeric[not_numeric].index))}"

        return x_copy

    def fit(self, X: DataFrame, y: Series) -> None:
        """Learning TRAIN distribution"""

        # checking data before bucketization
        x_copy = self.prepare_data(X, y)

        # [Quantitative features] Grouping values into quantiles
        discretizer = QuantileDiscretizer(
            self.features,
            min_freq=self.min_freq,
            values_orders=self.values_orders,
            str_nan=self.str_nan,
            copy=False,
        )
        x_copy = discretizer.fit_transform(x_copy, y)

        # storing orders of grouped features
        self.values_orders.update(discretizer.values_orders)

        # [Quantitative features] Grouping rare quantiles into closest common one
        #  -> can exist because of overrepresented values (values more frequent than 1/q)
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
            discretizer = OrdinalDiscretizer(
                has_rare,
                min_freq=self.min_freq,
                values_orders=self.values_orders,
                str_nan=self.str_nan,
                verbose=self.verbose,
                input_dtypes=self.input_dtypes,
            )
            discretizer.fit(x_copy, y)

            # storing orders of grouped features
            self.values_orders.update(discretizer.values_orders)

        # discretizing features based on each feature's values_order
        super().fit(X, y)

        return self
