"""Tools to build simple buckets out of Quantitative and Qualitative features
for a binary classification model.
"""

from typing import Any, Dict, List

from pandas import DataFrame, Series, unique
from pandas.api.types import is_numeric_dtype, is_string_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from .utils.base_discretizers import GroupedList, min_value_counts, nan_unique, GroupedListDiscretizer
from .utils.qualitative_discretizers import DefaultDiscretizer, OrdinalDiscretizer
from .utils.quantitative_discretizers import QuantileDiscretizer

from ..converters import StringConverter


class Discretizer(BaseEstimator, TransformerMixin):
    """Automatic discretizing of continuous, categorical and categorical ordinal features.

    Modalities/values of features are grouped according to there respective orders:
     - [Qualitative features] order based on modality target rate.
     - [Qualitative ordinal features] user-specified order.
     - [Quantitative features] real order of the values.

    Parameters
    ----------
    quanti_features: list
        Contains quantitative (continuous) features to be discretized.

    quali_features: list
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
        quanti_features: List[str],
        quali_features: List[str],
        min_freq: float,
        *,
        values_orders: Dict[str, Any] = None,
        copy: bool = False,
        verbose: bool = False,
    ) -> None:
        """_summary_

        Parameters
        ----------
        quanti_features : List[str]
            _description_
        quali_features : List[str]
            _description_
        min_freq : float
            _description_
        values_orders : Dict[str, Any], optional
            _description_, by default None
        copy : bool, optional
            _description_, by default False
        verbose : bool, optional
            _description_, by default False
        """
        self.features = quanti_features[:] + quali_features[:]
        self.quanti_features = quanti_features[:]
        assert len(list(set(quanti_features))) == len(
            quanti_features
        ), "Column duplicates in quanti_features"
        self.quali_features = quali_features[:]
        assert len(list(set(quali_features))) == len(
            quali_features
        ), "Column duplicates in quali_features"
        if values_orders is None:
            values_orders = {}
        self.values_orders = {k: GroupedList(v) for k, v in values_orders.items()}
        self.min_freq = min_freq
        self.q = int(1 / min_freq)  # number of quantiles
        self.pipe: List[BaseEstimator] = []
        self.copy = copy
        self.verbose = verbose

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
        if len(self.quali_features) > 0:
            # verbose if requested
            if self.verbose:
                print("\n---\n[Discretizer] Fit Qualitative Features")

            # grouping qualitative features
            discretizer = QualitativeDiscretizer(
                self.quali_features,
                min_freq=self.min_freq,
                values_orders=self.values_orders,
                copy=self.copy,
                verbose=self.verbose,
            )
            discretizer.fit(X, y)

            # storing results
            self.values_orders.update(
                discretizer.values_orders
            )  # adding orders of grouped features
            self.pipe += discretizer.pipe  # adding discretizer to pipe

        # [Quantitative features] Grouping quantitative features
        if len(self.quanti_features) > 0:
            # verbose if requested
            if self.verbose:
                print("\n---\n[Discretizer] Fit Quantitative Features")

            # grouping quantitative features
            discretizer = QuantitativeDiscretizer(
                self.quanti_features,
                q=self.q,
                values_orders=self.values_orders,
                copy=self.copy,
                verbose=self.verbose,
            )
            discretizer.fit(X, y)

            # storing results
            self.values_orders.update(
                discretizer.values_orders
            )  # adding orders of grouped features
            self.pipe += discretizer.pipe  # adding discretizer to pipe

        return self

    def transform(self, X: DataFrame, y: Series = None) -> DataFrame:
        """_summary_

        Parameters
        ----------
        X : DataFrame
            _description_
        y : Series, optional
            _description_, by default None

        Returns
        -------
        DataFrame
            _description_
        """
        # verbose if requested
        if self.verbose:
            print("\n---\n[Discretizer] Transform Features")

        # copying dataframe if requested
        Xc = X
        if self.copy:
            Xc = X.copy()

        # iterating over each transformer
        for _, step in self.pipe:
            Xc = step.transform(Xc)

        return Xc


class QualitativeDiscretizer(BaseEstimator, TransformerMixin):
    """Automatic discretizing of categorical and categorical ordinal features.

    Modalities/values of features are grouped according to there respective orders:
     - [Qualitative features] order based on modality target rate.
     - [Qualitative ordinal features] user-specified order.

    TODO: pass ordinal_features/qualitati_features as parameters to be able to pass values_orders with other orders (ex: from chaineddiscretizer)

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
        features: List[str],
        min_freq: float,
        *,
        values_orders: Dict[str, Any] = None,
        copy: bool = False,
        verbose: bool = False,
    ) -> None:
        """_summary_

        Parameters
        ----------
        features : List[str]
            _description_
        min_freq : float
            _description_
        values_orders : Dict[str, Any], optional
            _description_, by default None
        copy : bool, optional
            _description_, by default False
        verbose : bool, optional
            _description_, by default False
        """
        self.features = features[:]
        if values_orders is None:
            values_orders = {}
        self.values_orders = {k: GroupedList(v) for k, v in values_orders.items()}
        self.ordinal_features = [
            f for f in values_orders if f in features
        ]  # ignores non qualitative features
        self.non_ordinal_features = [f for f in features if f not in self.ordinal_features]
        self.min_freq = min_freq
        self.pipe: List[BaseEstimator] = []
        self.copy = copy
        self.verbose = verbose

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

        # copying dataframe
        Xc = X.copy()

        # checking for quantitative columns
        is_object = Xc[self.features].dtypes.apply(is_string_dtype)
        if not all(is_object):  # non qualitative features detected
            if self.verbose:
                print(
                    f"""Non-string features: {', '.join(is_object[~is_object].index)}, will be converted using Converters.StringConverter."""
                )

            # converting specified features into qualitative features
            stringer = StringConverter(features=self.features)
            Xc = stringer.fit_transform(Xc)

            # append the string converter to the feature engineering pipeline
            self.pipe += [("StringConverter", stringer)]

        # checking for binary target
        y_values = unique(y)
        assert (0 in y_values) & (
            1 in y_values
        ), "y must be a binary Series (int or float, not object)"
        assert len(y_values) == 2, "y must be a binary Series (int or float, not object)"

        # checking that all unique values in X are in values_orders
        # TODO: use base_discretizers.check_new_value
        uniques = Xc[self.features].apply(nan_unique)
        for feature in self.ordinal_features:
            missing = [val for val in uniques[feature] if val not in self.values_orders[feature]]
            assert (
                len(missing) == 0
            ), f"The ordering for {', '.join(missing)} of feature '{feature}' must be specified in values_orders (str-only)."

        return Xc

    def fit(self, X: DataFrame, y: Series) -> None:
        """Learning TRAIN distribution"""

        # checking data before bucketization
        Xc = self.prepare_data(X, y)

        # [Qualitative ordinal features] Grouping rare values into closest common one
        if len(self.ordinal_features) > 0:
            # discretizing
            ordinal_orders = {
                k: GroupedList(v)
                for k, v in self.values_orders.items()
                if k in self.ordinal_features
            }
            discretizer = OrdinalDiscretizer(
                ordinal_orders, min_freq=self.min_freq, verbose=self.verbose
            )
            discretizer.fit(Xc, y)

            # storing results
            self.values_orders.update(
                discretizer.values_orders
            )  # adding orders of grouped features
            self.pipe += [
                ("QualitativeOrdinalDiscretizer", discretizer)
            ]  # adding discretizer to pipe

        # [Qualitative non-ordinal features] Grouping rare values into default_value '__OTHER__'
        if len(self.non_ordinal_features) > 0:
            # Grouping rare modalities
            discretizer = DefaultDiscretizer(
                self.non_ordinal_features,
                min_freq=self.min_freq,
                values_orders=self.values_orders,
                verbose=self.verbose,
            )
            discretizer.fit(Xc, y)

            # storing results
            self.values_orders.update(
                discretizer.values_orders
            )  # adding orders of grouped features
            self.pipe += [("DefaultDiscretizer", discretizer)]  # adding discretizer to pipe

        return self

    def transform(self, X: DataFrame, y: Series = None) -> DataFrame:
        """Applying learned bucketization on TRAIN and/or TEST"""

        # copying dataframe if requested
        Xc = X
        if self.copy:
            Xc = X.copy()

        # iterating over each transformer
        for _, step in self.pipe:
            Xc = step.transform(Xc)

        return Xc


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
        features: List[str],
        min_freq: float,
        *,
        values_orders: Dict[str, Any] = None,
        str_nan: str = None,
        copy: bool = False,
        verbose: bool = False,
    ) -> None:
        """_summary_

        Parameters
        ----------
        features : List[str]
            _description_
        q : int
            _description_
        values_orders : Dict[str, Any], optional
            _description_, by default None
        copy : bool, optional
            _description_, by default False
        verbose : bool, optional
            _description_, by default False
        """        
        self.features = features[:]
        if values_orders is None:
            values_orders = {}
        self.values_orders = {k: GroupedList(v) for k, v in values_orders.items()}
        self.min_freq = min_freq
        self.str_nan = str_nan
        self.copy = copy
        self.verbose = verbose

    def prepare_data(self, X: DataFrame, y: Series) -> DataFrame:
        """Checking data for bucketization"""

        # checking for quantitative columns
        is_numeric = X[self.features].dtypes.apply(is_numeric_dtype)
        assert all(is_numeric), f"Non-numeric features: {', '.join(is_numeric[~is_numeric].index)}"

        # checking for binary target
        y_values = unique(y)
        assert (0 in y_values) & (
            1 in y_values
        ), "y must be a binary Series (int or float, not object)"
        assert len(y_values) == 2, "y must be a binary Series (int or float, not object)"

        # copying dataframe
        Xc = X.copy()

        return Xc

    def fit(self, X: DataFrame, y: Series) -> None:
        """Learning TRAIN distribution"""

        # checking data before bucketization
        Xc = self.prepare_data(X, y)

        # [Quantitative features] Grouping values into quantiles
        discretizer = QuantileDiscretizer(
            self.features,
            min_freq=self.min_freq,
            values_orders=self.values_orders,
            verbose=self.verbose,
            str_nan=self.str_nan,
            copy=False
        )
        Xc = discretizer.fit_transform(Xc, y)

        # storing orders of grouped features
        self.values_orders.update(discretizer.values_orders)
        print("self.values_orders", self.values_orders)

        # [Quantitative features] Grouping rare quantiles into closest common one
        #  -> can exist because of overrepresented values (values more frequent than 1/q)
        # searching for features with rare quantiles: computing min frequency per feature
        frequencies = Xc[self.features].apply(min_value_counts, values_orders=self.values_orders, axis=0)
        # TODO: change the values_orders for the formatted one
        print("frequencies", frequencies)

        # minimal frequency of a quantile
        q_min_freq = self.min_freq / 2

        # identifying features that have rare modalities
        has_rare = list(frequencies[frequencies <= q_min_freq].index)
        print("has_rare", has_rare)

        # Grouping rare modalities
        if len(has_rare) > 0:
            discretizer = OrdinalDiscretizer(
                has_rare, min_freq=self.min_freq, values_orders=self.values_orders, str_nan=self.str_nan, verbose=self.verbose
            )
            discretizer.fit(Xc, y)

            # storing orders of grouped features
            self.values_orders.update(discretizer.values_orders)

        # discretizing features based on each feature's values_order
        super().__init__(
            self.features,
            self.values_orders,
            copy=self.copy,
            input_dtype='float',
            output_dtype='str',
            str_nan=self.str_nan,
        )
        super().fit(X, y)

        return self
