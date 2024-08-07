"""Base tools to build simple buckets out of Quantitative and Qualitative features
for a binary classification model.
"""

import json
from typing import Any, Union, Type

from numpy import nan, select
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator, TransformerMixin

from ...features import BaseFeature, Features
from ...features.utils.grouped_list import GroupedList
from .multiprocessing import apply_async_function
from abc import ABC


class BaseDiscretizer(ABC, BaseEstimator, TransformerMixin):
    """Applies discretization using a dict of GroupedList to transform a DataFrame's columns."""

    __name__ = "BaseDiscretizer"

    def __init__(
        self,
        features: Features,
        **kwargs: dict,
    ) -> None:
        # features : list[str]
        #     List of column names of features (continuous, discrete, categorical or ordinal)
        # to be dicretized

        # input_dtypes : Union[str, dict[str, str]], optional
        #     Input data type, converted to a dict of the provided type for each feature,
        # by default ``"str"``

        #     * If ``"str"``, features are considered as qualitative.
        #     * If ``"float"``, features are considered as quantitative.

        # ordinal_encoding : str, optional
        #     To be choosen amongst ``["float", "str"]``, by default ``"str"``

        #     * If ``"float"``, grouped modalities will be converted to there corresponding
        #  floating rank.
        #     * If ``"str"``, a per-group modality will be set for all the modalities of a group.

        # dropna : bool, optional
        #     * If ``True``, ``numpy.nan`` will be attributed there label.
        #     * If ``False``, ``numpy.nan`` will be restored after discretization,
        # by default ``True``

        # str_default : str, optional
        #     String representation for default qualitative values, i.e. values less frequent than
        # ``min_freq``, by default ``"__OTHER__"``

        # features_casting : dict[str, list[str]], optional
        #     By default ``None``, target is considered as continuous or binary.
        #     Multiclass target: Dict of raw DataFrame columns associated to the names of copies
        # that will be created.
        """
        values_orders : dict[str, GroupedList], optional
            Dict of column names and there associated ordering.
            If lists are passed, a :class:`GroupedList` will automatically be initiated,
            by default ``None``

        copy : bool, optional
            If ``True``, applies transform to a copy of the provided DataFrame, by default ``False``

        verbose : bool, optional
            If ``True``, prints raw Discretizers Fit and Transform steps, by default ``False``

        n_jobs : int, optional
            Number of processes used by multiprocessing, by default ``1``

        **kwargs: dict
            Pass values for ``str_default`` and ``str_nan`` (default string values)

        Examples
        --------
        See `Discretizers examples <https://autocarver.readthedocs.io/en/latest/index.html>`_
        """
        # features and values
        self.features = features
        if isinstance(features, list):
            self.features = Features(features)

        # saving kwargs
        self.kwargs = kwargs

        # whether or not to copy input dataframe
        self.copy = kwargs.get("copy", True)

        # output type
        self.ordinal_encoding = kwargs.get("ordinal_encoding", False)

        # whether or not to reinstate numpy nan after bucketization
        self.dropna = kwargs.get("dropna", False)

        # whether to print info
        self.verbose = kwargs.get("verbose", True)

        # setting number of jobs
        self.n_jobs = kwargs.get("n_jobs", 1)

        # check if the discretizer has already been fitted
        self.is_fitted = False

        # initiating things for super().__repr__
        self.ordinals = None
        self.categoricals = None
        self.quantitatives = None
        self.ordinal_values = None
        self.min_freq = kwargs.get("min_freq", None)
        self.sort_by = kwargs.get("sort_by", None)

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:
        _ = N_CHAR_MAX  # unused attribute
        # truncating features if too long
        str_features = str(self.features)
        if len(str_features) > N_CHAR_MAX:
            str_features = str_features[:N_CHAR_MAX] + "..."
        return f"{self.__name__}({str_features})"

    def _remove_feature(self, feature: str) -> None:
        """Removes a feature from all ``BaseDiscretizer.feature`` attributes

        Parameters
        ----------
        feature : str
            Column name of the feature to remove
        """
        if feature in self.features:
            self.features.remove(feature)

    def _cast_features(self, X: DataFrame) -> DataFrame:
        """Casts the features of a DataFrame using feature versions to duplicate columns

        Parameters
        ----------
        X : DataFrame
            Dataset used to discretize. Needs to have columns has specified in
            ``BaseDiscretizer.features``, by default None.

        Returns
        -------
        DataFrame
            A formatted X
        """

        # duplicating features with versions disctinct from names (= multiclass target)
        X = X.assign(
            **{
                feature.version: X[feature.name]
                for feature in self.features
                if feature.version != feature.name and feature.version not in X
            }
        )

        return X

    def _prepare_data(self, X: DataFrame, y: Series = None) -> DataFrame:
        """Validates format and content of X and y.

        Parameters
        ----------
        X : DataFrame
            Dataset used to discretize. Needs to have columns has specified in
            ``BaseDiscretizer.features``, by default None.

        y : Series
            Binary target feature, by default None.

        Returns
        -------
        DataFrame
            A formatted copy of X
        """
        # pointer to X
        x_copy = X

        # checking DataFrame of features
        if X is not None:
            if not isinstance(X, DataFrame):  # checking for X's type
                raise ValueError(
                    f" - [{self.__name__}] X must be a pandas.DataFrame, passed {type(X)}"
                )

            # copying X
            if self.copy:
                x_copy = X.copy()

            # casting features for multiclass targets
            x_copy = self._cast_features(x_copy)

            # checking for input columns
            missing_columns = [
                feature for feature in self.features if feature.version not in x_copy
            ]
            if len(missing_columns) > 0:
                raise ValueError(
                    f" - [{self.__name__}] Requested discretization of {str(missing_columns)} but "
                    "those columns are missing from provided X. Please check your inputs! "
                )

            # checking target Series
            if y is not None:
                if not isinstance(y, Series):  # checking for y's type
                    raise ValueError(
                        f" - [{self.__name__}] y must be a pandas.Series, passed {type(y)}"
                    )

                if any(y.isna()):  # checking for nans in the target
                    raise ValueError(f" - [{self.__name__}] y should not contain numpy.nan")

                if not all(y.index == X.index):  # checking for matching indices
                    raise ValueError(f" - [{self.__name__}] X and y must have the same indices.")

        return x_copy

    __prepare_data = _prepare_data  # private copy

    def fit(self, X: DataFrame = None, y: Series = None) -> None:
        """Learns simple discretization of values of X according to values of y.

        Parameters
        ----------
        X : DataFrame
            Training dataset, to determine features' optimal carving
            Needs to have columns has specified in ``features`` attribute.

        y : Series
            Target with wich the association is maximized.
        """
        _, _ = X, y  # unused arguments

        # checking for previous fits of the discretizer that could cause unwanted errors
        if self.is_fitted:
            raise RuntimeError(
                " - [Discretizer] This Discretizer has already been fitted. "
                "Fitting it anew could break established orders. Please initialize a new one."
            )

        # checking that all features were fitted
        missing_features = [feature.version for feature in self.features if not feature.is_fitted]
        if len(missing_features) != 0:
            raise ValueError(f" - [Discretizer] Features not fitted: {str(missing_features)}.")

        # for each feature, getting label associated to each value
        self.features.update_labels(self.ordinal_encoding)

        # setting fitted as True to raise alerts
        self.is_fitted = True

        return self

    def transform(self, X: DataFrame, y: Series = None) -> DataFrame:
        """Applies discretization to a DataFrame's columns.

        Parameters
        ----------
        X : DataFrame
            Dataset to be carved.
            Needs to have columns has specified in ``features`` attribute.
        y : Series, optional
            Target, by default ``None``

        Returns
        -------
        DataFrame
            Discretized X.
        """
        # * For each feature's value, the associated group label is attributed (as definid by
        # ``values_orders``).
        # * If ``ordinal_encoding="float"``, converts labels into floats.
        # * Data types are matched as ``input_dtypes=="str"`` for qualitative features and
        # ``input_dtypes=="float"`` for quantitative ones.
        # * If ``copy=True``, the input DataFrame will be copied.

        # copying dataframes and casting for multiclass
        x_copy = self.__prepare_data(X, y)

        # filling up nans for features that have some
        x_copy = self.features.fillna(x_copy)

        # checking that all unique values in X are in features
        self.features.check_values(x_copy)

        # transforming quantitative features
        if len(self.features.get_quantitatives()) > 0:
            x_copy = self._transform_quantitative(x_copy, y)

        # transforming qualitative features
        if len(self.features.get_qualitatives()) > 0:
            x_copy = self._transform_qualitative(x_copy, y)

        # reinstating nans when not supposed to group them
        x_copy = self.features.unfillna(x_copy)

        return x_copy

    def _transform_quantitative(self, X: DataFrame, y: Series) -> DataFrame:
        """Applies discretization to a DataFrame's Quantitative columns.

        * Data types are defined by:
            * ``input_dtypes=="str"`` for qualitative features
            * ``input_dtypes=="float"`` for quantitative features
        * If ``copye=True``, the input DataFrame will be copied.

        Parameters
        ----------
        X : DataFrame
            Contains columns named after ``BaseDiscretizer.features`` attribute, by default None
        y : Series, optional
            Model target, by default None

        Returns
        -------
        DataFrame
            Discretized X.
        """
        _ = y  # unused argument

        # dataset length
        x_len = X.shape[0]

        # transforming all features
        transformed = apply_async_function(
            transform_quantitative_feature,
            self.features.get_quantitatives(),
            self.n_jobs,
            X,
            x_len,
        )

        # unpacking transformed series
        X[[feature for feature, _ in transformed]] = DataFrame(dict(transformed), index=X.index)

        return X

    def _transform_qualitative(self, X: DataFrame, y: Series = None) -> DataFrame:
        """Applies discretization to a DataFrame's Qualitative columns.

        * Data types are defined by:
            * ``input_dtypes=="str"`` for qualitative features
            * ``input_dtypes=="float"`` for quantitative features
        * If ``copye=True``, the input DataFrame will be copied.

        Parameters
        ----------
        X : DataFrame
            Contains columns named after ``BaseDiscretizer.features`` attribute, by default None
        y : Series, optional
            Model target, by default None

        Returns
        -------
        DataFrame
            Discretized X.
        """
        _ = y  # unused argument

        # list of qualitative features
        qualitatives = self.features.get_qualitatives()

        # replacing values for there corresponding label
        X.replace(
            {feature.version: feature.label_per_value for feature in qualitatives}, inplace=True
        )

        return X

    def _verbose(self, prefix: str = " -") -> None:
        """prints logs if requested"""
        if self.verbose:
            print(f"{prefix} [{self.__name__}] Fit {str(self.features)}")

    def to_dict(self) -> dict[str, GroupedList]:
        """Converts Discretizer to dict"""
        return self.features.get_content()

    def to_json(self, light_mode: bool = False) -> str:
        """Converts to JSON format.

        To be used with ``json.dump``.

        Parameters
        ----------
        light_mode: bool, optional
            Whether or not to save features' history and statistics, by default False

        Returns
        -------
        str
            JSON serialized object
        """
        # adding all parameters
        return {
            "features": self.features.to_json(light_mode),
            "dropna": self.dropna,
            "min_freq": self.min_freq,
            "sort_by": self.sort_by,
            "is_fitted": self.is_fitted,
            "n_jobs": self.n_jobs,
            "verbose": self.verbose,
            "ordinal_encoding": self.ordinal_encoding,
        }

    def save(self, file_name: str, light_mode: bool = False) -> None:
        """Saves pipeline to .json file.

        Parameters
        ----------
        file_name : str
            String of .json file name.
        light_mode: bool, optional
            Whether or not to save features' history and statistics, by default False

        Returns
        -------
        str
            JSON serialized object
        """
        # checking for input
        if file_name.endswith(".json"):
            with open(file_name, "w", encoding="utf-8") as json_file:
                json.dump(self.to_json(light_mode), json_file)
        # raising for non-json file name
        else:
            raise ValueError(f" - [{self.__name__}] Provide a file_name that ends with .json.")

    @classmethod
    def load_discretizer(cls: Type["BaseDiscretizer"], file_name: str) -> "BaseDiscretizer":
        """Allows one to load an Discretizer saved as a .json file.

        The Discretizer has to be saved with ``Discretizer.save()``, otherwise there
        can be no guarantee for it to be restored.

        Parameters
        ----------
        file_name : str
            String of saved Discretizer's .json file name.

        Returns
        -------
        BaseDiscretizer
            A fitted Discretizer.
        """
        # reading file
        with open(file_name, "r", encoding="utf-8") as json_file:
            discretizer_json = json.load(json_file)

        # deserializing features
        features = Features.load(
            discretizer_json.pop("features"), discretizer_json.get("ordinal_encoding")
        )

        # initiating BaseDiscretizer
        loaded_discretizer = cls(features=features, **discretizer_json)
        loaded_discretizer.fit()

        return loaded_discretizer

    # def history(self, feature: str = None) -> DataFrame:
    #     """Historic of tested combinations and there association with the target.

    #     By default:

    #         * ``str_default="__OTHER__"`` is added for features with non-representative modalities.
    #         * ``str_nan="__NAN__"`` is added for features that contain ``numpy.nan``.
    #         * Whatever the value of ``dropna``, the association is computed for non-missing values.

    #     Parameters
    #     ----------
    #     feature : str, optional
    #         Specify for which feature to return the history, by default ``None``

    #     Returns
    #     -------
    #     DataFrame
    #         Historic of features' tested combinations.
    #     """
    #     # checking for an history
    #     if self._history is not None:
    #         # getting feature's history
    #         if feature is not None:
    #             assert (
    #                 feature in self._history.keys()
    #             ), f"Carving of feature {feature} was not requested."
    #             histo = self._history[feature]

    #         # getting all features' history
    #         else:
    #             histo = []
    #             for feature in self._history.keys():
    #                 feature_histories = self._history[feature]
    #                 for feature_history in feature_histories:
    #                     feature_history.update({"feature": feature})
    #                 histo += feature_histories

    #         # formatting combinations
    #         # history["combination"] = history["combination"].apply(format_for_history)

    #         return DataFrame(histo)

    #     else:
    #         return self._history

    def summary(self, feature: str = None) -> DataFrame:
        """Summarizes the data discretization process.

        By default:

            * ``str_default="__OTHER__"`` is added for non-representative modalities.
            * ``str_nan="__NAN__"`` is adde for features that contain ``numpy.nan``.

        Parameters
        ----------
        feature : str, optional
            Specify for which feature to return the summary, by default ``None``

        Returns
        -------
        DataFrame
            A summary of features' values per modalities.
        """
        # checking for requested specific feature
        if feature is not None:
            summary = self.features(feature).get_summary()
            return DataFrame(summary).set_index(["feature", "label"])

        return self.features.get_summaries()

    # def update_discretizer(
    #     self,
    #     feature: str,
    #     mode: str,
    #     discarded_value: Union[str, float],
    #     kept_value: Union[str, float],
    # ) -> None:
    #     """Allows one to update the discretization groups

    #     Use with caution: no viability checks are performed.

    #     Parameters
    #     ----------
    #     feature : str
    #         Specify for which feature to update the discritezer
    #     mode : str
    #         * For ``mode="replace"``, ``discarded_value`` will be replaced by ``kept_value``
    #         * For ``mode="group"``, ``discarded_value`` will be grouped with ``kept_value``
    #     discarded_value : Union[str, float]
    #         A group value that won't exist after completion
    #     new_value : Union[str, float]
    #         A group value that will persist after completion
    #     """
    #     # checking for mode
    #     assert mode in ["group", "replace"], " - [Discretizer] Choose mode in ['group', 'replace']"

    #     # checking for nans
    #     if isnan(discarded_value):
    #         discarded_value = self.nan
    #         self.features_dropna[feature] = True
    #     assert not isnan(
    #         kept_value
    #     ), " - [Discretizer] missing values can only be grouped with an existing modality"

    #     # copying values_orders
    #     values_orders = {k: v for k, v in self.values_orders.items()}
    #     order = values_orders[feature]

    #     # checking that discarded_value is not already in new_value
    #     if order.get_group(discarded_value) == kept_value:
    #         warn(
    #             f" - [Discretizer] {discarded_value} is already grouped within {kept_value}",
    #             UserWarning,
    #         )

    #     # otherwise, proceeding
    #     else:
    #         # adding kept_value if it does not exists yet
    #         if not order.contains(kept_value):
    #             order.append(kept_value)

    #         # grouping discarded value in kept_value
    #         if mode == "group":
    #             # adding discarded_value if it does not exists yet
    #             if not order.contains(discarded_value):
    #                 order.append(discarded_value)
    #             # grouping discarded_value with kept_value
    #             order.group(discarded_value, kept_value)

    #         # replacing group leader if requested
    #         elif mode == "replace":
    #             # grouping kept_value with discarded_value
    #             order.group(kept_value, discarded_value)

    #             # checking that kept_value is in discarded_value
    #             assert order.get_group(kept_value) == discarded_value, (
    #                 f" - [Discretizer] Can not proceed! {kept_value} already is a "
    #                 f"member of another group ({order.get_group(kept_value)})"
    #             )

    #             # replacing group leader
    #             order.replace_group_leader(discarded_value, kept_value)

    #         # updating Carver values_orders and labels_per_values
    #         self.values_orders.update({feature: order})
    #         self.labels_per_values = self._get_labels_per_values(self.ordinal_encoding)


def transform_quantitative_feature(
    feature: BaseFeature, df_feature: Series, x_len: int
) -> tuple[str, Series]:
    """Transforms a quantitative feature"""
    # identifying nans
    feature_nans = (df_feature == feature.nan) | df_feature.isna()

    # converting nans to there corresponding quantile (if it was grouped to a quantile)
    if any(feature_nans):
        # value with which nans have grouped
        nan_group = feature.values.get_group(feature.nan)

        # checking that nans have been grouped to a quantile
        if nan_group != feature.nan:
            df_feature.mask(feature_nans, nan_group, inplace=True)

        # otherwise they are left as NaNs for comparison purposes
        else:
            df_feature.mask(feature_nans, nan, inplace=True)

    # list of masks of values to replace with there respective group
    values_to_group = [df_feature <= value for value in feature.values if value != feature.nan]

    # corressponding group for each value
    group_labels = [
        [feature.label_per_value[value]] * x_len for value in feature.values if value != feature.nan
    ]

    # checking for values to group
    if len(values_to_group) > 0:
        df_feature = Series(select(values_to_group, group_labels, default=df_feature))

    # reinstating nans otherwise nan is converted to 'nan' by numpy
    if any(feature_nans):
        df_feature[feature_nans] = feature.label_per_value.get(feature.nan, nan)

    return feature.version, list(df_feature)


def applied_to_dict_list(applied: Union[DataFrame, Series]) -> dict[str, list[Any]]:
    """Converts a DataFrame or a List in a Dict of lists

    Parameters
    ----------
    applied : Union[DataFrame, Series]
        Result of pandas.DataFrame.apply

    Returns
    -------
    Dict[list[Any]]
        Dict of lists of rows values
    """
    # case when it's a Series
    converted = applied.to_dict()

    # case when it's a DataFrame
    if isinstance(applied, DataFrame):
        converted = applied.to_dict(orient="list")

    return converted


class extend_docstring:
    """Used to extend a Child's method docstring with the Parent's method docstring"""

    def __init__(self, method):
        self.doc = method.__doc__

    def __call__(self, function):
        if self.doc is not None:
            doc = function.__doc__
            function.__doc__ = self.doc
            if doc is not None:
                function.__doc__ = doc + function.__doc__
        return function
