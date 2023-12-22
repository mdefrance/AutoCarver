"""Base tools to build simple buckets out of Quantitative and Qualitative features
for a binary classification model.
"""

from multiprocessing import Pool
from typing import Any, Union
from warnings import warn

from numpy import floating, integer, isfinite, isnan, nan, select
from pandas import DataFrame, Series, isna, notna, unique
from sklearn.base import BaseEstimator, TransformerMixin

from .grouped_list import GroupedList
from .serialization import json_deserialize_values_orders, json_serialize_values_orders


class BaseDiscretizer(BaseEstimator, TransformerMixin):
    """Applies discretization using a dict of GroupedList to transform a DataFrame's columns."""

    def __init__(
        self,
        features: list[str],
        *,
        values_orders: dict[str, GroupedList] = None,  #
        input_dtypes: Union[str, dict[str, str]] = "str",
        output_dtype: str = "str",
        dropna: bool = True,
        copy: bool = True,  #
        verbose: bool = True,  #
        str_nan: str = None,  #
        str_default: str = None,
        features_casting: dict[str, list[str]] = None,
        features_dropna: dict[str, bool] = None,
        n_jobs: int = 1,
    ) -> None:
        # features : list[str]
        #     List of column names of features (continuous, discrete, categorical or ordinal)
        # to be dicretized

        # input_dtypes : Union[str, dict[str, str]], optional
        #     Input data type, converted to a dict of the provided type for each feature,
        # by default ``"str"``

        #     * If ``"str"``, features are considered as qualitative.
        #     * If ``"float"``, features are considered as quantitative.

        # output_dtype : str, optional
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
        self.features = list(set(features))
        if values_orders is None:
            values_orders: dict[str, GroupedList] = {}
        self.values_orders = {
            feature: GroupedList(values) for feature, values in values_orders.items()
        }
        self.copy = copy

        # input feature types
        if isinstance(input_dtypes, str):
            input_dtypes = {feature: input_dtypes for feature in features}
        self.input_dtypes = input_dtypes

        # output type
        self.output_dtype = output_dtype

        # string value of numpy.nan
        self.str_nan = str_nan

        # whether or not to reinstate numpy nan after bucketization
        self.dropna = dropna
        if features_dropna is None:
            features_dropna = {feature: self.dropna for feature in self.features}
        self.features_dropna = features_dropna

        # string value of rare values
        self.str_default = str_default

        # whether to print info
        self.verbose = verbose

        # identifying qualitative features by there type
        self.qualitative_features = [
            feature for feature in features if self.input_dtypes[feature] == "str"
        ]

        # identifying quantitative features by there type
        self.quantitative_features = [
            feature for feature in features if self.input_dtypes[feature] == "float"
        ]

        # setting number of jobs
        self.n_jobs = n_jobs

        # for each feature, getting label associated to each value
        self.labels_per_values: dict[str, dict[Any, Any]] = {}  # will be initiated during fit

        # check if the discretizer has already been fitted
        self.is_fitted = False

        # target classes multiclass vs binary vs continuous
        if features_casting is None:
            features_casting: list[Any] = {feature: [feature] for feature in self.features}
        self.features_casting = {
            feature: casting[:] for feature, casting in features_casting.items()
        }

    def _get_labels_per_values(self, output_dtype: str) -> dict[str, dict[Any, Any]]:
        """Creates a dict that contains, for each feature, for each value, the associated label

        Parameters
        ----------
        output_dtype : str
            Whether or not to convert the output to float.

        Returns
        -------
        dict[str, dict[Any, Any]]
            Dict of labels per values per feature
        """
        # initiating dict of labels per values per feature
        labels_per_values: dict[str, dict[Any, Any]] = {}

        # iterating over each feature
        for feature in self.features:
            # known values of the feature(grouped)
            values = self.values_orders[feature]

            # case 0: quantitative feature -> labels per quantile (removes str_nan)
            if feature in self.quantitative_features:
                labels = get_labels(values, self.str_nan)
            # case 1: qualitative feature -> by default, labels are values
            else:
                labels = [value for value in values if value != self.str_nan]  # (removing str_nan)

            # add NaNs if there are any
            if self.str_nan in values:
                labels += [self.str_nan]

            # requested float output (AutoCarver) -> converting to integers
            if output_dtype == "float":
                labels = [n for n, _ in enumerate(labels)]

            # building label per value
            label_per_value: dict[Any, Any] = {}
            for group_of_values, label in zip(values, labels):
                for value in values.get(group_of_values):
                    label_per_value.update({value: label})

            # storing in full dict
            labels_per_values.update({feature: label_per_value})

        return labels_per_values

    def _remove_feature(self, feature: str) -> None:
        """Removes a feature from all ``BaseDiscretizer.feature`` attributes

        Parameters
        ----------
        feature : str
            Column name of the feature to remove
        """
        if feature in self.features:
            self.features.remove(feature)
            if feature in self.qualitative_features:
                self.qualitative_features.remove(feature)
            if feature in self.quantitative_features:
                self.quantitative_features.remove(feature)
            if feature in self.values_orders:
                self.values_orders.pop(feature)
            if feature in self.input_dtypes:
                self.input_dtypes.pop(feature)
            if feature in self.labels_per_values:
                self.labels_per_values.pop(feature)
            if feature in self.features_dropna:
                self.features_dropna.pop(feature)

            # getting corresponding raw_feature
            raw_feature = next(
                raw_feature
                for raw_feature, casting in self.features_casting.items()
                if feature in casting
            )
            casting = self.features_casting.get(raw_feature)
            # removing feature from casting
            casting.remove(feature)
            self.features_casting.update({raw_feature: casting})
            # removing raw feature if there are no more casting for it
            if len(self.features_casting.get(raw_feature)) == 0:
                self.features_casting.pop(raw_feature)

    def _cast_features(self, X: DataFrame) -> DataFrame:
        """Casts the features of a DataFrame using features_casting to duplicate columns

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
        # for binary/continuous targets
        if all(len(feature_casting) == 1 for feature_casting in self.features_casting.values()):
            X.rename(
                columns={
                    feature: feature_casting[0]
                    for feature, feature_casting in self.features_casting.items()
                },
                inplace=True,
            )

        # for multiclass targets
        else:
            # duplicating features
            X = X.assign(
                **{
                    casted_feature: X[feature]
                    for feature, feature_casting in self.features_casting.items()
                    for casted_feature in feature_casting
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
        x_copy = X
        if X is not None:
            # checking for X's type
            assert isinstance(
                X, DataFrame
            ), f" - [Discretizer] X must be a pandas.DataFrame, instead {type(X)} was passed"

            # copying X
            x_copy = X
            if self.copy:
                x_copy = X.copy()

            # casting features for multiclass targets
            x_copy = self._cast_features(x_copy)

            # checking for input columns
            missing_columns = [feature for feature in self.features if feature not in x_copy]
            assert len(missing_columns) == 0, (
                f" - [Discretizer] Requested discretization of {str(missing_columns)} but those"
                " columns are missing from provided X. Please check your inputs! "
            )

            if y is not None:
                # checking for y's type
                assert isinstance(
                    y, Series
                ), f" - [Discretizer] y must be a pandas.Series, instead {type(y)} was passed"

                # checking for nans in the target
                assert not any(y.isna()), " - [Discretizer] y should not contain numpy.nan"

                # checking indices
                assert all(
                    y.index == X.index
                ), " - [Discretizer] X and y must have the same indices."

        return x_copy

    __prepare_data = _prepare_data  # private copy

    def _check_new_values(self, X: DataFrame, features: list[str]) -> None:
        """Checks for new, unexpected values, in X

        Parameters
        ----------
        X : DataFrame
            New DataFrame (at transform time)
        features : list[str]
            List of column names
        """
        # unique non-nan values in new dataframe
        uniques = X[features].apply(
            nan_unique,
            axis=0,
            result_type="expand",
        )
        uniques = applied_to_dict_list(uniques)

        # replacing unknwon values if there was a default discretization
        X.replace(
            {
                feature: {
                    val: self.str_default
                    for val in uniques[feature]
                    if val not in self.values_orders[feature].values()
                    and val != self.str_nan
                    and self.str_default in self.values_orders[feature].values()
                }
                for feature in features
            },
            inplace=True,
        )
        # updating unique non-nan values in new dataframe
        uniques = X[features].apply(
            nan_unique,
            axis=0,
            result_type="expand",
        )
        uniques = applied_to_dict_list(uniques)

        # checking for unexpected values for each feature
        for feature in features:
            # unexpected values for this feature
            unexpected = [
                val for val in uniques[feature] if val not in self.values_orders[feature].values()
            ]
            assert len(unexpected) == 0, (
                " - [Discretizer] Unexpected value! The ordering for values: "
                f"{str(list(unexpected))} of feature '{feature}' was not provided. "
                "There might be new values in your test/dev set. Consider taking a bigger "
                f"test/dev set or dropping the column {feature}."
            )

        return X

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
        assert not self.is_fitted, (
            " - [Discretizer] This Discretizer has already been fitted. "
            "Fitting it anew could break established orders. Please initialize a new one."
        )

        # checking that all features to discretize are in values_orders
        missing_features = [
            feature for feature in self.features if feature not in self.values_orders
        ]
        assert len(missing_features) == 0, (
            " - [Discretizer] Missing values_orders for following features "
            f"{str(missing_features)}."
        )

        # for each feature, getting label associated to each value
        self.labels_per_values = self._get_labels_per_values(self.output_dtype)

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
        # * If ``output_dtype="float"``, converts labels into floats.
        # * Data types are matched as ``input_dtypes=="str"`` for qualitative features and
        # ``input_dtypes=="float"`` for quantitative ones.
        # * If ``copy=True``, the input DataFrame will be copied.

        # copying dataframes and casting for multiclass
        x_copy = self.__prepare_data(X, y)

        # transforming quantitative features
        if len(self.quantitative_features) > 0:
            x_copy = self._transform_quantitative(x_copy, y)

        # transforming qualitative features
        if len(self.qualitative_features) > 0:
            x_copy = self._transform_qualitative(x_copy, y)

        # reinstating nans
        for feature, dropna in self.features_dropna.items():
            if not dropna:  # checking whether we should have dropped nans or not
                label_per_value = self.labels_per_values[feature]
                # checking that nans were grouped
                if self.str_nan in label_per_value:
                    x_copy[feature] = x_copy[feature].replace(label_per_value[self.str_nan], nan)

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

        # no multiprocessing
        if self.n_jobs <= 1:
            all_transformed = [
                transform_quantitative_feature(
                    feature,
                    X[feature],
                    self.values_orders,
                    self.str_nan,
                    self.labels_per_values,
                    x_len,
                )
                for feature in self.quantitative_features
            ]

        # asynchronous transform of each feature
        else:
            with Pool(processes=self.n_jobs) as pool:
                all_transformed_async = [
                    pool.apply_async(
                        transform_quantitative_feature,
                        (
                            feature,
                            X[feature],
                            self.values_orders,
                            self.str_nan,
                            self.labels_per_values,
                            x_len,
                        ),
                    )
                    for feature in self.quantitative_features
                ]

                #  waiting for the results
                all_transformed = [result.get() for result in all_transformed_async]

        # unpacking transformed series
        X.update(DataFrame({feature: values for feature, values in all_transformed}))

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

        # filling up nans from features that have some
        if self.str_nan:
            X[self.qualitative_features] = X[self.qualitative_features].fillna(self.str_nan)

        # checking that all unique values in X are in values_orders
        X = self._check_new_values(X, features=self.qualitative_features)

        # replacing values for there corresponding label
        X = X.replace(
            {
                feature: label_per_value
                for feature, label_per_value in self.labels_per_values.items()
                if feature in self.qualitative_features
            }
        )

        return X

    def to_json(self) -> str:
        """Converts to .json format.

        To be used with ``json.dump``.

        Returns
        -------
        str
            JSON serialized object
        """
        # extracting content dictionnaries
        json_serialized_base_discretizer = {
            "features": self.features,
            "values_orders": json_serialize_values_orders(self.values_orders),
            "features_casting": self.features_casting,
            "input_dtypes": self.input_dtypes,
            "output_dtype": self.output_dtype,
            "str_nan": self.str_nan,
            "str_default": self.str_default,
            "dropna": self.dropna,
            "features_dropna": self.features_dropna,
            "copy": self.copy,
        }

        # dumping as json
        return json_serialized_base_discretizer

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
        # storing requested feature for later
        requested_features = self.features[:]
        if feature is not None:
            requested_features = [feature]
            assert feature in self.features, (
                f"Discretization of feature {feature} was not " "requested or it has been dropped."
            )

        # raw label per value with output_dtype 'str'
        raw_labels_per_values = self._get_labels_per_values(output_dtype="str")

        # initiating summaries
        summaries: list[dict[str, Any]] = []
        for feature in requested_features:
            # adding each value/label
            for value, label in self.labels_per_values[feature].items():
                # checking that nan where dropped
                if not (not self.dropna and value == self.str_nan):
                    # initiating feature summary (default value/label)
                    feature_summary = {
                        "feature": feature,
                        "dtype": self.input_dtypes[feature],
                        "label": label,
                        "content": value,
                    }

                    # case 0: qualitative feature -> not adding floats and integers str_default
                    if feature in self.qualitative_features:
                        if not isinstance(value, floating) and not isinstance(
                            value, float
                        ):  # checking for floats
                            if not isinstance(value, integer) and not isinstance(
                                value, int
                            ):  # checking for ints
                                if value != self.str_default:  # checking for str_default
                                    summaries += [feature_summary]

                    # case 1: quantitative feature -> take the raw label per value
                    elif feature in self.quantitative_features:
                        feature_summary.update({"content": raw_labels_per_values[feature][value]})
                        summaries += [feature_summary]

        # adding nans for quantitative features (when nan has been grouped)
        for feature in self.quantitative_features:
            # initiating feature summary (no value/label)
            feature_summary = {"feature": feature, "dtype": self.input_dtypes[feature]}
            # if there are nans -> if already added it will be dropped afterwards (unique content)
            if self.str_nan in raw_labels_per_values[feature]:
                nan_group = self.values_orders[feature].get_group(self.str_nan)
                feature_summary.update(
                    {"label": self.labels_per_values[feature][nan_group], "content": self.str_nan}
                )
                summaries += [feature_summary]

        # aggregating unique values per label
        summaries = (
            DataFrame(summaries)
            .groupby(["feature", "dtype", "label"])["content"]
            .apply(lambda u: list(unique(u)))
            .reset_index()
        )
        # sorting content
        sorted_contents: list[list[Any]] = []
        for content in summaries["content"]:
            content.sort(key=repr)
            sorted_contents += [content]
        summaries["content"] = sorted_contents
        # sorting and seting index
        summaries = summaries.sort_values(["dtype", "feature"]).set_index(["feature", "dtype"])

        return summaries

    def update_discretizer(
        self,
        feature: str,
        mode: str,
        discarded_value: Union[str, float],
        kept_value: Union[str, float],
    ) -> None:
        """Allows one to update the discretization groups

        Use with caution: no viability checks are performed.

        Parameters
        ----------
        feature : str
            Specify for which feature to update the discritezer
        mode : str
            * For ``mode="replace"``, ``discarded_value`` will be replaced by ``kept_value``
            * For ``mode="group"``, ``discarded_value`` will be grouped with ``kept_value``
        discarded_value : Union[str, float]
            A group value that won't exist after completion
        new_value : Union[str, float]
            A group value that will persist after completion
        """
        # checking for mode
        assert mode in ["group", "replace"], " - [Discretizer] Choose mode in ['group', 'replace']"

        # checking for nans
        if isnan(discarded_value):
            discarded_value = self.str_nan
            self.features_dropna[feature] = True
        assert not isnan(
            kept_value
        ), " - [Discretizer] missing values can only be grouped with an existing modality"

        # copying values_orders
        values_orders = {k: v for k, v in self.values_orders.items()}
        order = values_orders[feature]

        # checking that discarded_value is not already in new_value
        if order.get_group(discarded_value) == kept_value:
            warn(
                f" - [Discretizer] {discarded_value} is already grouped within {kept_value}",
                UserWarning,
            )

        # otherwise, proceeding
        else:
            # adding kept_value if it does not exists yet
            if not order.contains(kept_value):
                order.append(kept_value)

            # grouping discarded value in kept_value
            if mode == "group":
                # adding discarded_value if it does not exists yet
                if not order.contains(discarded_value):
                    order.append(discarded_value)
                # grouping discarded_value with kept_value
                order.group(discarded_value, kept_value)

            # replacing group leader if requested
            elif mode == "replace":
                # grouping kept_value with discarded_value
                order.group(kept_value, discarded_value)

                # checking that kept_value is in discarded_value
                assert order.get_group(kept_value) == discarded_value, (
                    f" - [Discretizer] Can not proceed! {kept_value} already is a "
                    f"member of another group ({order.get_group(kept_value)})"
                )

                # replacing group leader
                order.replace_group_leader(discarded_value, kept_value)

            # updating Carver values_orders and labels_per_values
            self.values_orders.update({feature: order})
            self.labels_per_values = self._get_labels_per_values(self.output_dtype)


def transform_quantitative_feature(
    feature, df_feature, values_orders, str_nan, labels_per_values, x_len
):
    """Transforms a quantitative feature"""
    print(feature)

    # feature's labels associated to each quantile
    feature_values = values_orders[feature]

    # keeping track of nans
    nans = isna(df_feature)

    # converting nans to there corresponding quantile (if it was grouped to a quantile)
    if any(nans):
        assert feature_values.contains(str_nan), (
            " - [Discretizer] Unexpected value! Missing values found for feature "
            f"'{feature}' at transform step but not during fit. There might be new values "
            "in your test/dev set. Consider taking a bigger test/dev set or dropping the "
            f"column {feature}."
        )
        nan_value = feature_values.get_group(str_nan)
        # checking that nans have been grouped to a quantile otherwise they are left as
        # numpy.nan (for comparison purposes)
        if nan_value != str_nan:
            df_feature[nans] = nan_value

    # list of masks of values to replace with there respective group
    values_to_group = [df_feature <= value for value in feature_values if value != str_nan]

    # corressponding group for each value
    group_labels = [
        [labels_per_values[feature][value]] * x_len for value in feature_values if value != str_nan
    ]

    # checking for values to group
    if len(values_to_group) > 0:
        df_feature = select(values_to_group, group_labels, default=df_feature)

    # converting nans to there value
    if any(nans):
        df_feature[nans] = labels_per_values[feature].get(nan_value, str_nan)

    return feature, list(df_feature)


def convert_to_labels(
    features: list[str],
    quantitative_features: list[str],
    values_orders: dict[str, GroupedList],
    str_nan: str,
    dropna: bool = True,
) -> dict[str, GroupedList]:
    """Converts a values_orders values (quantiles) to labels"""
    # copying values_orders without nans
    labels_orders = {
        feature: GroupedList([value for value in values_orders[feature] if value != str_nan])
        for feature in features
    }

    # for quantitative features getting labels per quantile
    if any(quantitative_features):
        # getting group "name" per quantile
        quantiles_labels, _ = get_quantiles_labels(quantitative_features, values_orders, str_nan)

        # applying alliases to known orders
        for feature in quantitative_features:
            labels_orders.update(
                {
                    feature: GroupedList(
                        [quantiles_labels[feature][quantile] for quantile in labels_orders[feature]]
                    )
                }
            )

    # adding back nans if requested
    if not dropna:
        for feature in features:
            if str_nan in values_orders[feature]:
                order = labels_orders[feature]
                order.append(str_nan)  # adding back nans at the end of the order
                labels_orders.update({feature: order})

    return labels_orders


def convert_to_values(
    features: list[str],
    quantitative_features: list[str],
    values_orders: dict[str, GroupedList],
    label_orders: dict[str, GroupedList],
    str_nan: str,
) -> dict[str, Any]:
    """Converts a values_orders labels to values (quantiles)"""
    # for quantitative features getting labels per quantile
    if any(quantitative_features):
        # getting quantile per group "name"
        _, labels_to_quantiles = get_quantiles_labels(quantitative_features, values_orders, str_nan)

    # updating feature orders (that keeps NaNs and quantiles)
    for feature in features:
        # initial complete ordering with NAN and quantiles
        order = values_orders[feature]

        # checking for grouped modalities
        groups_to_discard = label_orders[feature].content

        # grouping the raw quantile values
        for kept_value, group_to_discard in groups_to_discard.items():
            # for qualitative features grouping as is
            # for quantitative features getting quantile per alias
            if feature in quantitative_features:
                # getting raw quantiles to be grouped
                group_to_discard = [
                    labels_to_quantiles[feature][label_discarded]
                    if label_discarded != str_nan
                    else str_nan
                    for label_discarded in group_to_discard
                ]

                # choosing the value to keep as the group
                which_to_keep = [value for value in group_to_discard if value != str_nan]
                # case 0: keeping the largest value amongst the discarded (otherwise not grouped)
                if len(which_to_keep) > 0:
                    kept_value = max(which_to_keep)
                # case 1: there is only str_nan in the group (it was not grouped)
                else:
                    kept_value = group_to_discard[0]

            # grouping quantiles
            order.group_list(group_to_discard, kept_value)

        # updating ordering
        values_orders.update({feature: order})

    return values_orders


def value_counts(x: Series, dropna: bool = False, normalize: bool = True) -> dict:
    """Counts the values of each modality of a series into a dictionnary

    Parameters
    ----------
    x : Series
        _description_
    dropna : bool, optional
        _description_, by default False
    normalize : bool, optional
        _description_, by default True

    Returns
    -------
    dict
        _description_
    """

    values = x.value_counts(dropna=dropna, normalize=normalize)

    return values.to_dict()


def target_rate(x: Series, y: Series, dropna: bool = True, ascending=True) -> dict:
    """Target y rate per modality of x into a dictionnary

    Parameters
    ----------
    x : Series
        _description_
    y : Series
        _description_
    dropna : bool, optional
        _description_, by default True
    ascending : bool, optional
        _description_, by default True

    Returns
    -------
    dict
        _description_
    """

    rates = y.groupby(x, dropna=dropna).mean().sort_values(ascending=ascending)

    return rates.to_dict()


def get_labels(quantiles: list[float], str_nan: str) -> list[str]:
    """_summary_

    Parameters
    ----------
    feature : str
        _description_
    order : GroupedList
        _description_
    str_nan : str
        _description_

    Returns
    -------
    list[str]
        _description_
    """
    # filtering out nan and inf for formatting
    quantiles = [val for val in quantiles if val != str_nan and isfinite(val)]

    # converting quantiles in string
    labels = format_quantiles(quantiles)

    return labels


def get_quantiles_labels(
    features: list[str], values_orders: dict[str, GroupedList], str_nan: str
) -> tuple[dict[str, GroupedList], dict[str, GroupedList]]:
    """Converts a values_orders of quantiles into a values_orders of string quantiles

    Parameters
    ----------
    features : list[str]
        _description_
    values_orders : dict[str, GroupedList]
        _description_
    str_nan : str
        _description_

    Returns
    -------
    dict[str, GroupedList]
        _description_
    """
    # applying quantiles formatting to orders of specified features
    quantiles_to_labels = {}
    labels_to_quantiles = {}
    for feature in features:
        quantiles = list(values_orders[feature])
        labels = get_labels(quantiles, str_nan)

        # associates quantiles to their respective labels
        quantiles_to_labels.update(
            {feature: {quantile: alias for quantile, alias in zip(quantiles, labels)}}
        )
        # associates quantiles to their respective labels
        labels_to_quantiles.update(
            {feature: {alias: quantile for quantile, alias in zip(quantiles, labels)}}
        )

    return quantiles_to_labels, labels_to_quantiles


def format_quantiles(a_list: list[float]) -> list[str]:
    """Formats a list of float quantiles into a list of boundaries.

    Rounds quantiles to the closest power of 1000.

    Parameters
    ----------
    a_list : list[float]
        Sorted list of quantiles to convert into string

    Returns
    -------
    list[str]
        List of boundaries per quantile
    """
    # scientific formatting
    formatted_list = [f"{number:.3e}" for number in a_list]

    # stripping whitespaces
    formatted_list = [string.strip() for string in formatted_list]

    # low and high bounds per quantiles
    upper_bounds = formatted_list + [nan]
    lower_bounds = [nan] + formatted_list
    order: list[str] = []
    for lower, upper in zip(lower_bounds, upper_bounds):
        if isna(lower):
            order += [f"x <= {upper}"]
        elif isna(upper):
            order += [f"{lower} < x"]
        else:
            order += [f"{lower} < x <= {upper}"]

    return order


def nan_unique(x: Series) -> list[Any]:
    """Unique non-NaN values.

    Parameters
    ----------
    x : Series
        Values to be deduplicated.

    Returns
    -------
    list[Any]
        List of unique non-nan values
    """

    # unique values
    uniques = unique(x)

    # filtering out nans
    uniques = [u for u in uniques if notna(u)]

    return uniques


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


def check_missing_values(
    X: DataFrame, features: list[str], known_values: dict[str, list[Any]]
) -> None:
    """Checks for missing values from X, (unexpected values in values_orders)

    Parameters
    ----------
    X : DataFrame
        New DataFrame (at transform time)
    features : list[str]
        List of column names
    known_values : dict[str, list[Any]]
        Dict of known values per column name
    """
    # unique non-nan values in new dataframe
    uniques = X[features].apply(
        nan_unique,
        axis=0,
        result_type="expand",
    )
    uniques = applied_to_dict_list(uniques)

    # checking for unexpected values for each feature
    for feature in features:
        unexpected = [val for val in known_values[feature] if val not in uniques[feature]]
        assert len(unexpected) == 0, (
            f"Unexpected value! The ordering for values: {str(list(unexpected))} of feature '"
            f"{feature}' was provided but there are not matching value in provided X. You should"
            f" check 'values_orders['{feature}']' for unwanted values."
        )


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


def load_discretizer(discretizer_json: dict) -> BaseDiscretizer:
    """Allows one to load a Discretizer saved as a .json file.

    The Discretizer has to be saved with ``json.dump(f, Discretizer.to_json())``, otherwise there
    can be no guarantee for it to be restored.

    Parameters
    ----------
    discretizer_json : str
        Loaded .json file using ``json.load(f)``.

    Returns
    -------
    BaseDiscretizer
        A fitted Discretizer.
    """
    # deserializing values_orders
    values_orders = json_deserialize_values_orders(discretizer_json["values_orders"])

    # updating auto_carver attributes
    discretizer_json.update({"values_orders": values_orders})

    # initiating BaseDiscretizer
    discretizer = BaseDiscretizer(**discretizer_json)
    discretizer.fit()

    return discretizer
