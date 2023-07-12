"""Tools to build simple buckets out of Quantitative and Qualitative features
for a binary classification model.
"""

from typing import Any, Dict, List, Union

from pandas import DataFrame, Series, unique
from pandas.api.types import is_numeric_dtype
from .utils.base_discretizers import GroupedList, min_value_counts, GroupedListDiscretizer, check_new_values
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
        quantitative_features: List[str],
        qualitative_features: List[str],
        min_freq: float,
        *,
        ordinal_features: List[str] = None,
        values_orders: Dict[str, GroupedList] = None,
        copy: bool = False,
        verbose: bool = False,
        str_nan: str = '__NAN__',
        str_default: str = '__OTHER__'
    ) -> None:
        """_summary_

        Parameters
        ----------
        quantitative_features : List[str]
            _description_
        qualitative_features : List[str]
            _description_
        min_freq : float
            _description_
        ordinal_features : List[str], optional
            _description_, by default None
        values_orders : Dict[str, GroupedList], optional
            _description_, by default None
        input_dtypes : Union[str, Dict[str, str]], optional
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
        self.quantitative_features = quantitative_features[:]
        assert len(list(set(quantitative_features))) == len(
            quantitative_features
        ), "Column duplicates in quantitative_features"

        self.qualitative_features = qualitative_features[:]
        assert len(list(set(qualitative_features))) == len(
            qualitative_features
        ), "Column duplicates in qualitative_features"
        
        if ordinal_features is None:
            ordinal_features = []
        self.ordinal_features = ordinal_features[:]
        if values_orders is None:
            values_orders = {}
        self.values_orders = {k: GroupedList(v) for k, v in values_orders.items()}
        self.min_freq = min_freq

        self.copy = copy
        self.verbose = verbose
        self.str_nan = str_nan
        self.str_default = str_default
        
        self.features = list(set(quantitative_features + qualitative_features + ordinal_features))

        # initializing input_dtypes
        self.input_dtypes = {feature: 'str' for feature in self.features}
        self.input_dtypes.update({feature: 'float' for feature in quantitative_features})

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

        # [Quantitative features] Grouping quantitative features
        if len(self.quantitative_features) > 0:
            if self.verbose:  # verbose if requested
                print("\n---\n[Discretizer] Fit Quantitative Features")

            # grouping quantitative features
            discretizer = QuantitativeDiscretizer(
                features=self.quantitative_features,
                min_freq=self.min_freq,
                values_orders=self.values_orders,
                str_nan=self.str_nan,
                copy=self.copy,
                verbose=self.verbose,
            )
            discretizer.fit(X, y)

            # storing orders of grouped features
            self.values_orders.update(discretizer.values_orders)

        # discretizing features based on each feature's values_order
        super().__init__(
            self.features,
            self.values_orders,
            copy=self.copy,
            input_dtypes=self.input_dtypes,
            output_dtype='str',  # TODO: it won't work up to auto carver
            str_nan=self.str_nan,
        )
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
        qualitative_features: List[str],
        min_freq: float,
        *,
        ordinal_features: List[str] = None,
        values_orders: Dict[str, Any] = None,
        input_dtypes: Union[str, Dict[str, str]] = 'str',
        str_nan: str = '__NAN__',
        str_default: str = '__OTHER__',
        copy: bool = False,
        verbose: bool = False,
    ) -> None:
        """_summary_

        Parameters
        ----------
        qualitative_features : List[str]
            _description_
        min_freq : float
            _description_
        ordinal_features : List[str], optional
            _description_, by default None
        values_orders : Dict[str, Any], optional
            _description_, by default None
        input_dtypes : Union[str, Dict[str, str]], optional
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
        self.qualitative_features = list(set(qualitative_features))
        if values_orders is None:
            values_orders = {}
        self.values_orders = {feature: GroupedList(values) for feature, values in values_orders.items()}
        if ordinal_features is None:
            ordinal_features = []
        self.ordinal_features = list(set(ordinal_features))
        self.min_freq = min_freq
        self.str_nan = str_nan
        self.str_default = str_default
        self.copy = copy
        self.verbose = verbose

        # non-ordinal qualitative features
        self.non_ordinal_features = [feature for feature in qualitative_features if feature not in self.ordinal_features]
        # all unique features
        self.features = list(set(self.ordinal_features + self.non_ordinal_features))

        if input_dtypes is None:
            input_dtypes = {feature: 'str' for feature in self.features}
        if isinstance(input_dtypes, str):
            input_dtypes = {feature: input_dtypes for feature in self.features}
        self.input_dtypes = input_dtypes
        
    def prepare_data(self, X: DataFrame, y: Series = None) -> DataFrame:
        """Prepares the data for bucketization, checks column types.
        Converts non-string columns into strings.

        TODO: check that features dont have too many values (-> quantitative features?)

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
        x_copy = X.copy()

        # checking for columns containing floats or integers even with filled nans
        dtypes = x_copy[self.features].fillna(self.str_nan).applymap(type).apply(unique)
        not_object = dtypes.apply(lambda u: float in u or int in u)

        # non qualitative features detected
        if any(not_object):
            if self.verbose:
                print(
                    f"""Non-string features: {', '.join(not_object[not_object].index)}, will be converted using type_discretizers.StringDiscretizer."""
                )

            # converting specified features into qualitative features
            stringer = StringDiscretizer(features=list(not_object.index[not_object]))
            x_copy = stringer.fit_transform(x_copy)

            # updating values_orders accordingly
            non_ordinal_orders = {feature: value for feature, value in stringer.values_orders.items() if feature not in self.ordinal_features}
            self.values_orders.update(non_ordinal_orders)

        # checking for binary target
        y_values = unique(y)
        assert (0 in y_values) & (
            1 in y_values
        ), "y must be a binary Series (int or float, not object)"
        assert len(y_values) == 2, "y must be a binary Series (int or float, not object)"

        # all known values for features
        known_values = {feature: order.values() for feature, order in self.values_orders.items()}

        # checking that all unique values in X are in values_orders
        check_new_values(X, self.ordinal_features, known_values)

        return x_copy

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
                str_nan=self.str_nan
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
        super().__init__(
            self.features,
            self.values_orders,
            copy=self.copy,
            input_dtypes=self.input_dtypes,
            output_dtype='str',
            str_nan=self.str_nan,
        )
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
        features: List[str],
        min_freq: float,
        *,
        values_orders: Dict[str, Any] = None,
        input_dtypes: Union[str, Dict[str, str]] = 'float',
        str_nan: str = '__NAN__',
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
        input_dtypes : Union[str, Dict[str, str]], optional
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
        self.features = list(set(features))
        if values_orders is None:
            values_orders = {}
        self.values_orders = {feature: GroupedList(values) for feature, values in values_orders.items()}
        self.min_freq = min_freq
        self.str_nan = str_nan
        self.copy = copy
        self.verbose = verbose

        if input_dtypes is None:
            input_dtypes = {feature: 'float' for feature in self.features}
        if isinstance(input_dtypes, str):
            input_dtypes = {feature: input_dtypes for feature in self.features}
        self.input_dtypes = input_dtypes

    def prepare_data(self, X: DataFrame, y: Series) -> DataFrame:
        """Checking data for bucketization"""

        # checking for quantitative columns
        dtypes = X[self.features].applymap(type).apply(unique)
        not_numeric = dtypes.apply(lambda u: str in u)
        assert all(~not_numeric), f"Non-numeric features: {', '.join(not_numeric[not_numeric].index)}"

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

        # [Quantitative features] Grouping rare quantiles into closest common one
        #  -> can exist because of overrepresented values (values more frequent than 1/q)
        # searching for features with rare quantiles: computing min frequency per feature
        frequencies = Xc[self.features].apply(min_value_counts, values_orders=self.values_orders, axis=0)

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
                input_dtypes=self.input_dtypes
            )
            discretizer.fit(Xc, y)

            # storing orders of grouped features
            self.values_orders.update(discretizer.values_orders)

        # discretizing features based on each feature's values_order
        super().__init__(
            self.features,
            self.values_orders,
            copy=self.copy,
            input_dtypes=self.input_dtypes,
            output_dtype='str',
            str_nan=self.str_nan,
        )
        super().fit(X, y)

        return self
