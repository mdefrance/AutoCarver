"""Base tools to build simple buckets out of Quantitative and Qualitative features
for a binary classification model.
"""

from typing import Any, Union

from numpy import array, floating, inf, integer, isfinite, nan, select
from pandas import DataFrame, Series, isna, notna, unique
from sklearn.base import BaseEstimator, TransformerMixin

from .grouped_list import GroupedList
from .serialization import json_serialize_values_orders


class BaseDiscretizer(BaseEstimator, TransformerMixin):
    """Applies discretization using a dict of GroupedList to transform a DataFrame's columns."""

    def __init__(
        self,
        features: list[str],
        *,
        values_orders: dict[str, GroupedList] = None,
        input_dtypes: Union[str, dict[str, str]] = "str",
        output_dtype: str = "str",
        dropna: bool = True,
        copy: bool = True,
        verbose: bool = True,
        str_nan: str = None,
        str_default: str = None,
    ) -> None:
        """Initiates a ``BaseDiscretizer``.

        Parameters
        ----------
        features : list[str]
            List of column names of features (continuous, discrete, categorical or ordinal) to be dicretized

        values_orders : dict[str, GroupedList], optional
            Dict of feature's column names and there associated ordering.
            If lists are passed, a GroupedList will automatically be initiated, by default ``None``

        input_dtypes : Union[str, dict[str, str]], optional
            Input data type, converted to a dict of the provided type for each feature, by default ``"str"``

            * If ``"str"``, features are considered as qualitative.
            * If ``"float"``, features are considered as quantitative.

        output_dtype : str, optional
            To be choosen amongst ``["float", "str"]``, by default ``"str"``

            * If ``"float"``, grouped modalities will be converted to there corresponding floating rank.
            * If ``"str"``, a per-group modality will be set for all the modalities of a group.

        dropna : bool, optional
            * If ``True``, ``numpy.nan`` will be attributed there label.
            * If ``False``, ``numpy.nan`` will be restored after discretization, by default ``True``

        copy : bool, optional
            If ``True``, feature processing at transform is applied to a copy of the provided DataFrame, by default ``False``

        verbose : bool, optional
            If ``True``, prints raw Discretizers Fit and Transform steps, as long as
            information on AutoCarver's processing and tables of target rates and frequencies for
            X, by default ``False``

        str_nan : str, optional
            String representation to input ``numpy.nan``. If ``dropna=False``, ``numpy.nan`` will be left unchanged, by default ``"__NAN__"``

        str_default : str, optional
            String representation for default qualitative values, i.e. values less frequent than ``min_freq``, by default ``"__OTHER__"``
        """
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

        # for each feature, getting label associated to each value
        self.labels_per_values: dict[str, dict[Any, Any]] = {}  # will be initiated during fit

        # check if the discretizer has already been fitted
        self.is_fitted = False

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

    def _prepare_data(self, X: DataFrame, y: Series = None) -> DataFrame:
        """Validates format and content of X and y.

        Parameters
        ----------
        X : DataFrame
            Dataset used to discretize. Needs to have columns has specified in ``BaseDiscretizer.features``, by default None.

        y : Series
            Binary target feature, by default None.

        Returns
        -------
        DataFrame
            A formatted copy of X
        """
        # checking for previous fits of the discretizer that could cause unwanted errors
        assert (
            not self.is_fitted
        ), " - [BaseDiscretizer] This Discretizer has already been fitted. Fitting it anew could break established orders. Please initialize a new one."

        # checking for input columns
        x_copy = X
        if X is not None:
            missing_columns = [feature for feature in self.features if feature not in X]
            assert (
                len(missing_columns) == 0
            ), f" - [BaseDiscretizer] Missing features from the provided DataFrame: {str(missing_columns)}"

            # copying X
            x_copy = X
            if self.copy:
                x_copy = X.copy()

        return x_copy

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

        # checking for unexpected values for each feature
        for feature in features:
            unexpected = [
                val for val in uniques[feature] if val not in self.values_orders[feature].values()
            ]
            assert (
                len(unexpected) == 0
            ), f" - [BaseDiscretizer] Unexpected value! The ordering for values: {str(list(unexpected))} of feature '{feature}' was not provided. There might be new values in your test/dev set. Consider taking a bigger test/dev set or dropping the column {feature}."

    def fit(self, X: DataFrame = None, y: Series = None) -> None:
        """Learns the labels associated to each value for each feature

        Parameters
        ----------
        X : DataFrame
            Contains columns named after ``BaseDiscretizer.features`` attribute, by default None
        y : Series, optional
            Model target, by default None
        """
        # checking that all features to discretize are in values_orders
        missing_features = [
            feature for feature in self.features if feature not in self.values_orders
        ]
        assert (
            len(missing_features) == 0
        ), f" - [BaseDiscretizer] Missing values_orders for following features {str(missing_features)}."

        # for each feature, getting label associated to each value
        self.labels_per_values = self._get_labels_per_values(self.output_dtype)

        # setting fitted as True to raise alerts
        self.is_fitted = True

        return self

    def transform(self, X: DataFrame, y: Series = None) -> DataFrame:
        """Applies discretization to a DataFrame's columns.

        * For each feature's modality, the associated group label is attributed as definid by ``values_orders``.
        * If ``output_dtype="float"``, converts labels into floats.
        * Data types are matched as ``input_dtypes=="str"`` for qualitative features and ``input_dtypes=="float"`` for quantitative ones.
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
        # copying dataframes
        x_copy = X
        if self.copy:
            x_copy = X.copy()

        # applying default discretization for concerned features
        for feature in self.features:
            known_values = self.values_orders[feature].values()
            if self.str_default in known_values:
                known_values_index = x_copy[feature].isin(known_values)
                nans = (x_copy[feature].isna()) | (x_copy[feature] == self.str_nan)
                x_copy.loc[(~known_values_index) & (~nans), feature] = self.str_default

        # transforming quantitative features
        if len(self.quantitative_features) > 0:
            if self.verbose:  # verbose if requested
                print(
                    f" - [BaseDiscretizer] Transform Quantitative {str(self.quantitative_features)}"
                )
            x_copy = self._transform_quantitative(x_copy, y)

        # transforming qualitative features
        if len(self.qualitative_features) > 0:
            if self.verbose:  # verbose if requested
                print(
                    f" - [BaseDiscretizer] Transform Qualitative {str(self.qualitative_features)}"
                )
            x_copy = self._transform_qualitative(x_copy, y)

        # reinstating nans
        if not self.dropna:
            for feature in self.features:
                label_per_value = self.labels_per_values[feature]
                if self.str_nan in label_per_value:  # checking for nans in the feature
                    x_copy[feature] = x_copy[feature].replace(label_per_value[self.str_nan], nan)

        return x_copy

    def _transform_quantitative(self, X: DataFrame, y: Series) -> DataFrame:
        """Applies discretization to a DataFrame's Quantitative columns.

        * Data types are matched as ``input_dtypes=="str"`` for qualitative features and ``input_dtypes=="float"`` for quantitative ones.
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
        # dataset length
        x_len = X.shape[0]

        # grouping each quantitative feature
        for feature in self.quantitative_features:
            # feature's labels associated to each quantile
            feature_values = self.values_orders[feature]

            # keeping track of nans
            nans = isna(X[feature])

            # converting nans to there corresponding quantile (if it was grouped to a quantile)
            if any(nans):
                assert feature_values.contains(
                    self.str_nan
                ), f" - [BaseDiscretizer] Unexpected value! Missing values found for feature '{feature}' at transform step but not during fit. There might be new values in your test/dev set. Consider taking a bigger test/dev set or dropping the column {feature}."
                nan_value = feature_values.get_group(self.str_nan)
                # checking that nans have been grouped to a quantile otherwise they are left as numpy.nan (for comparison purposes)
                if nan_value != self.str_nan:
                    X.loc[nans, feature] = nan_value

            # list of masks of values to replace with there respective group
            values_to_group = [
                X[feature] <= value for value in feature_values if value != self.str_nan
            ]

            # corressponding group for each value
            group_labels = [
                [self.labels_per_values[feature][value]] * x_len
                for value in feature_values
                if value != self.str_nan
            ]

            # checking for values to group
            if len(values_to_group) > 0:
                X[feature] = select(values_to_group, group_labels, default=X[feature])

            # converting nans to there value
            if any(nans):
                X.loc[nans, feature] = self.labels_per_values[feature].get(nan_value, self.str_nan)

        return X

    def _transform_qualitative(self, X: DataFrame, y: Series = None) -> DataFrame:
        """Applies discretization to a DataFrame's Qualitative columns.

        * Data types are matched as ``input_dtypes=="str"`` for qualitative features and ``input_dtypes=="float"`` for quantitative ones.
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
        # filling up nans from features that have some
        if self.str_nan:
            X[self.qualitative_features] = X[self.qualitative_features].fillna(self.str_nan)

        # checking that all unique values in X are in values_orders
        self._check_new_values(X, features=self.qualitative_features)

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
        """Converts the BaseDiscretizer's values_orders to .json format.

        To be used with ``json.dump``.

        Returns
        -------
        str
            JSON serialized BaseDiscretizer
        """
        # extracting content dictionnaries
        json_serialized_groupedlistdiscretizer = {
            "features": self.features,
            "values_orders": json_serialize_values_orders(self.values_orders),
            "input_dtypes": self.input_dtypes,
            "output_dtype": self.output_dtype,
            "str_nan": self.str_nan,
            "str_default": self.str_default,
            "dropna": self.dropna,
            "copy": self.copy,
        }

        # dumping as json
        return json_serialized_groupedlistdiscretizer

    # TODO: add crosstabs per feature for a provided X?
    # TODO: change str_nan for str(nan)
    def summary(self) -> DataFrame:
        """Summarizes the data discretization process.

        Returns
        -------
        DataFrame
            A summary of features' values per modalities.
        """
        # raw label per value with output_dtype 'str'
        raw_labels_per_values = self._get_labels_per_values(output_dtype="str")

        # initiating summaries
        summaries: list[dict[str, Any]] = []
        for feature in self.features:
            # adding each value/label
            for value, label in self.labels_per_values[feature].items():
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


def convert_to_labels(
    features: list[str],
    quantitative_features: list[str],
    values_orders: dict[str, GroupedList],
    str_nan: str,
    dropna: bool = True,
) -> dict[str, GroupedList]:
    """Converts a values_orders values (quantiles) to labels

    Parameters
    ----------
    features : list[str]
        _description_
    quantitative_features : list[str]
        _description_
    values_orders : dict[str, GroupedList]
        _description_
    str_nan : str
        _description_
    dropna : bool, optional
        _description_, by default True

    Returns
    -------
    dict[str, GroupedList]
        _description_
    """
    # copying values_orders without nans
    labels_orders = {
        feature: GroupedList([value for value in values_orders[feature] if value != str_nan])
        for feature in features
    }

    # for quantitative features getting labels per quantile
    if any(quantitative_features):
        # getting group "name" per quantile
        quantiles_labels = get_quantiles_labels(quantitative_features, values_orders, str_nan)

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
):
    """Converts a values_orders labels to values (quantiles)

    Parameters
    ----------
    features : list[str]
        _description_
    quantitative_features : list[str]
        _description_
    values_orders : dict[str, GroupedList]
        _description_
    label_orders : dict[str, GroupedList]
        _description_
    str_nan : str
        _description_

    Returns
    -------
    _type_
        _description_
    """
    # for quantitative features getting labels per quantile
    if any(quantitative_features):
        # getting quantile per group "name"
        labels_to_quantiles = get_labels_quantiles(quantitative_features, values_orders, str_nan)

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
                # case 0: keeping the largest value amongst the discarded (otherwise they wont be grouped)
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


def min_value_counts(
    x: Series,
    values_orders: dict[str, GroupedList] = None,
    dropna: bool = False,
    normalize: bool = True,
) -> float:
    """Minimum of modalities' frequencies.

    Parameters
    ----------
    x : Series
        _description_
    values_orders : dict[str, GroupedList], optional
        _description_, by default None
    dropna : bool, optional
        _description_, by default False
    normalize : bool, optional
        _description_, by default True

    Returns
    -------
    float
        _description_
    """
    # modality frequency
    values = x.value_counts(dropna=dropna, normalize=normalize)

    # setting indices
    order = values_orders.get(x.name)
    if order is not None:
        values = values.reindex(order).fillna(0)

    # minimal frequency
    min_values = values.values.min()

    return min_values


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


def nunique(x: Series, dropna=False) -> int:
    """Computes number of unique modalities

    Parameters
    ----------
    x : Series
        _description_
    dropna : bool, optional
        _description_, by default False

    Returns
    -------
    int
        _description_
    """

    uniques = unique(x)
    n = len(uniques)

    # removing nans
    if dropna:
        if any(isna(uniques)):
            n -= 1

    return n


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
    # filtering out nan for formatting
    if str_nan in quantiles:
        quantiles = [val for val in quantiles if val != str_nan]

    # filtering out inf for formatting
    if inf in quantiles:
        quantiles = [val for val in quantiles if isfinite(val)]

    # converting quantiles in string
    labels = format_quantiles(quantiles)

    return labels


def get_quantiles_labels(
    features: list[str], values_orders: dict[str, GroupedList], str_nan: str
) -> dict[str, GroupedList]:
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
    for feature in features:
        quantiles = list(values_orders[feature])
        labels = get_labels(quantiles, str_nan)
        # associates quantiles to their respective labels
        quantiles_to_labels.update(
            {feature: {quantile: alias for quantile, alias in zip(quantiles, labels)}}
        )

    return quantiles_to_labels


def get_labels_quantiles(
    features: list[str], values_orders: dict[str, GroupedList], str_nan: str
) -> dict[str, GroupedList]:
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
    labels_to_quantiles = {}
    for feature in features:
        quantiles = list(values_orders[feature])
        labels = get_labels(quantiles, str_nan)
        # associates quantiles to their respective labels
        labels_to_quantiles.update(
            {feature: {alias: quantile for quantile, alias in zip(quantiles, labels)}}
        )

    return labels_to_quantiles


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

    # finding the closest power of thousands for each element
    closest_powers = [
        next((k for k in range(-3, 4) if abs(elt) / 100 // 1000 ** (k) < 10)) for elt in a_list
    ]

    # rounding elements to the closest power of thousands
    rounded_to_powers = [elt / 1000 ** (k) for elt, k in zip(a_list, closest_powers)]

    # computing optimal decimal per unique power of thousands
    optimal_decimals: dict[str, int] = {}
    for power in unique(closest_powers):  # iterating over each power of thousands found
        # filtering on the specific power of thousands
        sub_array = array([elt for elt, p in zip(rounded_to_powers, closest_powers) if power == p])

        # number of distinct values
        n_uniques = sub_array.shape[0]

        # computing the first rounding decimal that allows for distinction of
        # each values when rounded, by default None (no rounding)
        optimal_decimal = next(
            (k for k in range(1, 10) if len(unique(sub_array.round(k))) == n_uniques),
            None,
        )

        # storing in the dict
        optimal_decimals.update({power: optimal_decimal})

    # rounding according to each optimal_decimal
    rounded_list: list[float] = []
    for elt, power in zip(rounded_to_powers, closest_powers):
        # rounding each element
        rounded = elt  # by default None (no rounding)
        optimal_decimal = optimal_decimals.get(power)
        if optimal_decimal:  # checking that the optimal decimal exists
            rounded = round(elt, optimal_decimal)

        # adding the rounded number to the list
        rounded_list += [rounded]

    # dict of suffixes per power of thousands
    suffixes = {-3: "n", -2: "mi", -1: "m", 0: "", 1: "K", 2: "M", 3: "B"}

    # adding the suffixes
    formatted_list = [
        f"{elt: 1.1f}{suffixes[power]}" for elt, power in zip(rounded_list, closest_powers)
    ]

    # keeping zeros
    formatted_list = [
        rounded if raw != 0 else f"{raw: 1.1f}" for rounded, raw in zip(formatted_list, a_list)
    ]

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
    # TODO: use this function whenever apply is used

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
        assert (
            len(unexpected) == 0
        ), f"Unexpected value! The ordering for values: {str(list(unexpected))} of feature '{feature}' was provided but there are not matching value in provided X. You should check 'values_orders['{feature}']' for unwanted values."
