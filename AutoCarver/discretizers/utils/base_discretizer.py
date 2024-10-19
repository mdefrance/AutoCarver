"""Base tools to build simple buckets out of Quantitative and Qualitative features
for a binary classification model.
"""

import json
from abc import ABC
from dataclasses import dataclass

from numpy import nan, select
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator, TransformerMixin

from ...features import BaseFeature, Features
from ...utils import get_attribute, get_bool_attribute
from .multiprocessing import apply_async_function
from ...combinations import CombinationEvaluator


@dataclass
class Sample:
    """sample class to store X and y"""

    X: DataFrame
    y: Series = None

    def __getitem__(self, key):
        if key == "X":
            return self.X
        if key == "y":
            return self.y

        raise KeyError(key)

    def __iter__(self):
        return iter(["X", "y"])

    def keys(self):
        return ["X", "y"]

    @property
    def shape(self):
        return self.X.shape

    @property
    def index(self):
        return self.X.index

    @property
    def columns(self):
        return self.X.columns

    def __len__(self):
        return len(self.X)


class BaseDiscretizer(ABC, BaseEstimator, TransformerMixin):
    """Applies discretization using a dict of GroupedList to transform a DataFrame's columns."""

    __name__ = "BaseDiscretizer"

    def __init__(
        self,
        features: Features,
        **kwargs,
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

        # checking types of bool attributes
        self.copy = get_bool_attribute(kwargs, "copy", True)
        self.ordinal_encoding = get_bool_attribute(kwargs, "ordinal_encoding", False)
        self.dropna = get_bool_attribute(kwargs, "dropna", False)
        self._verbose = get_bool_attribute(kwargs, "verbose", True)

        # setting number of jobs
        self.n_jobs = get_attribute(kwargs, "n_jobs", 1)

        # check if the discretizer has already been fitted
        self.is_fitted = get_bool_attribute(kwargs, "is_fitted", False)

        # carver attributes
        self.min_freq = kwargs.get("min_freq")  # default to None
        self.combinations = kwargs.get("combinations")  # default to None

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:
        """Returns the string representation of the Discretizer"""
        _ = N_CHAR_MAX  # unused attribute
        # truncating features if too long
        str_features = str(self.features)
        if len(str_features) > N_CHAR_MAX:
            str_features = str_features[:N_CHAR_MAX] + "..."
        return f"{self.__name__}({str_features})"

    @property
    def verbose(self) -> bool:
        """Returns the verbose attribute"""
        return self._verbose

    @verbose.setter
    def verbose(self, value: bool) -> None:
        """Sets the verbose attribute"""
        self._verbose = value

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

        # duplicating columns that have several versions
        casted_columns = {
            feature.version: X[feature.name]
            for feature in self.features
            if feature.version != feature.name and feature.version not in X
        }

        # duplicating features with versions disctinct from names (= multiclass target)
        if len(casted_columns) > 0:  # checking for casted feature to not break inplace
            X = X.assign(**casted_columns)

        return X

    def _prepare_y(self, y: Series) -> None:
        """Validates input y"""

        if not isinstance(y, Series):  # checking for y's type
            raise ValueError(f"[{self.__name__}] y must be a pandas.Series, passed {type(y)}")

        if any(y.isna()):  # checking for nans in the target
            raise ValueError(f"[{self.__name__}] y should not contain numpy.nan")

    def _prepare_X(self, X: DataFrame) -> DataFrame:
        """Validates input X"""

        # checking for X's type
        if not isinstance(X, DataFrame):
            raise ValueError(f"[{self.__name__}] X must be a pandas.DataFrame, passed {type(X)}")

        # copying X
        x_copy = X
        if self.copy:
            x_copy = X.copy()

        # checking for input columns by feature name
        missing_columns = [feature for feature in self.features if feature.name not in x_copy]
        if len(missing_columns) > 0:
            raise ValueError(
                f"[{self.__name__}] Requested discretization of {str(missing_columns)} but "
                "those columns are missing from provided X. Please check your inputs! "
            )

        # casting features for multiclass targets
        x_copy = self._cast_features(x_copy)

        # checking for input columns by feature version
        missing_columns = [feature for feature in self.features if feature.version not in x_copy]
        if len(missing_columns) > 0:
            raise ValueError(
                f"[{self.__name__}] Requested discretization of {str(missing_columns)} but "
                "those columns are missing from provided X. Please check your inputs! "
            )

        return x_copy

    def _prepare_data(self, sample: Sample) -> Sample:
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

        # checking DataFrame of features
        if sample.X is not None:
            sample.X = self._prepare_X(sample.X)

            # checking target Series
            if sample.y is not None:
                self._prepare_y(sample.y)

                # checking for matching indices
                if not len(sample.y.index) == len(sample.X.index) or not all(
                    sample.y.index == sample.X.index
                ):
                    raise ValueError(f"[{self.__name__}] X and y must have the same indices.")

        return sample

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
                f"[{self.__name__}] Already fitted. "
                "Fitting it anew could break it. Please initialize a new one."
            )

        # checking that all features were fitted
        missing_features = [feature.version for feature in self.features if not feature.is_fitted]
        if len(missing_features) != 0:
            raise RuntimeError(f"[{self.__name__}] Features not fitted: {str(missing_features)}.")

        # setting features in ordinal encoding mode
        self.features.ordinal_encoding = self.ordinal_encoding

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

        # checking that it was fitted
        if not self.is_fitted:
            raise RuntimeError(f"[{self.__name__}] Call fit method first.")

        # copying dataframes and casting for multiclass
        sample = self.__prepare_data(Sample(X, y))

        # filling up nans for features that have some
        sample.X = self.features.fillna(sample.X)

        # checking that all unique values in X are in features
        self.features.check_values(sample.X)

        # transforming quantitative features
        if len(self.features.quantitatives) > 0:
            sample = self._transform_quantitative(sample)

        # transforming qualitative features
        if len(self.features.qualitatives) > 0:
            sample = self._transform_qualitative(sample)

        # reinstating nans when not supposed to group them
        return self.features.unfillna(sample.X)

    def _transform_quantitative(self, sample: Sample) -> Sample:
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
        # transforming all features
        transformed = apply_async_function(
            transform_quantitative_feature,
            self.features.quantitatives,
            self.n_jobs,
            sample.X,
            sample.shape[0],
        )

        # unpacking transformed series
        sample.X[[feature for feature, _ in transformed]] = DataFrame(
            dict(transformed), index=sample.index
        )

        return sample

    def _transform_qualitative(self, sample: Sample) -> Sample:
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
        # list of qualitative features
        qualitatives = self.features.qualitatives

        # replacing values for there corresponding label
        sample.X.replace(
            {feature.version: feature.label_per_value for feature in qualitatives}, inplace=True
        )

        return sample

    def log_if_verbose(self, prefix: str = " -") -> None:
        """prints logs if requested"""
        if self.verbose:
            print(f"{prefix} [{self.__name__}] Fit {str(self.features)}")

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
        content = {
            "features": self.features.to_json(light_mode),
            "dropna": self.dropna,
            "min_freq": self.min_freq,
            "combinations": self.combinations,
            "is_fitted": self.is_fitted,
            "n_jobs": self.n_jobs,
            "verbose": self.verbose,
            "ordinal_encoding": self.ordinal_encoding,
        }

        # adding combinations if it exists
        if isinstance(self.combinations, CombinationEvaluator):
            content["combinations"] = self.combinations.to_json()

        return content

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
            raise ValueError(f"[{self.__name__}] Provide a file_name that ends with .json.")

    @classmethod
    def load(cls, file_name: str) -> "BaseDiscretizer":
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
        features = Features.load(discretizer_json.pop("features"))

        # initiating BaseDiscretizer
        return cls(features=features, **discretizer_json)

    # def history(self, feature: str = None) -> DataFrame:
    #     """Historic of tested combinations and there association with the target.

    #     By default:

    #         * ``str_default="__OTHER__"`` is added for features with non-representative
    #  modalities.
    #         * ``str_nan="__NAN__"`` is added for features that contain ``numpy.nan``.
    #         * Whatever the value of ``dropna``, the association is computed for non-missing
    # values.

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

    # def update(
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
    #     assert mode in ["group", "replace"], " - [Discretizer] Choose mode in
    # ['group', 'replace']"

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
    #             f"[Discretizer] {discarded_value} is already grouped within {kept_value}",
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
    #                 f"[Discretizer] Can not proceed! {kept_value} already is a "
    #                 f"member of another group ({order.get_group(kept_value)})"
    #             )

    #             # replacing group leader
    #             order.replace_group_leader(discarded_value, kept_value)

    #         # updating Carver values_orders and labels_per_values
    #         self.values_orders.update({feature: order})
    #         self.labels_per_values = self._get_labels_per_values(self.ordinal_encoding)


def transform_quantitative_feature(
    feature: BaseFeature, df_feature: Series, x_len: int
) -> tuple[str, list]:
    """Transforms a quantitative feature"""

    # keeping track of original index
    raw_index = df_feature.index

    # identifying nans
    feature_nans = (df_feature == feature.nan) | df_feature.isna()

    # converting nans to there corresponding quantile (if it was grouped to a quantile)
    if any(feature_nans):
        # quantile with which nans have been grouped
        nan_group = feature.values.get_group(feature.nan)

        # checking that nans have been grouped to a quantile
        if nan_group == feature.nan:
            nan_group = nan

        # converting to quantile value if grouped else keeping np.nan
        df_feature.mask(feature_nans, nan_group, inplace=True)

    # list of masks of values to replace with there respective group
    values_to_group = [df_feature <= value for value in feature.values if value != feature.nan]

    # corressponding group for each value
    group_labels = [
        [feature.label_per_value[value]] * x_len for value in feature.values if value != feature.nan
    ]

    # checking for values to group
    # if len(values_to_group) > 0:  # TODO check if this is needed
    df_feature = Series(select(values_to_group, group_labels, default=df_feature), index=raw_index)

    # reinstating nans otherwise nan is converted to 'nan' by numpy
    if any(feature_nans):
        df_feature[feature_nans] = feature.label_per_value.get(feature.nan, nan)

    return feature.version, list(df_feature)
