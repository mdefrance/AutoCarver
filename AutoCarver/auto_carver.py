"""Tool to build optimized buckets out of Quantitative and Qualitative features
for a binary classification model.
"""

from typing import Any, Callable, Union

from numpy import add, array, mean, searchsorted, sqrt, unique, zeros
from pandas import DataFrame, Series, crosstab
from scipy.stats import chi2_contingency, kruskal
from tqdm import tqdm

from .discretizers import GroupedList
from .discretizers.discretizers import Discretizer
from .discretizers.utils.base_discretizers import (
    BaseDiscretizer,
    convert_to_labels,
    convert_to_values,
)
from .discretizers.utils.serialization import json_deserialize_values_orders

# trying to import extra dependencies
try:
    from IPython.display import display_html
except ImportError:
    _has_idisplay = False
else:
    _has_idisplay = True


class AutoCarver(BaseDiscretizer):
    """Automatic carving of continuous, discrete, categorical and ordinal
    features that maximizes association with a binary or continuous target.

    First fits a :ref:`Discretizer`. Raw data should be provided as input (not a result of ``Discretizer.transform()``).
    """

    def __init__(
        self,
        min_freq: float,
        sort_by: str,
        *,
        quantitative_features: list[str] = None,
        qualitative_features: list[str] = None,
        ordinal_features: list[str] = None,
        values_orders: dict[str, GroupedList] = None,
        max_n_mod: int = 5,
        # min_carved_freq: float = 0,  # TODO: update this parameter so that it is set according to frequency rather than number of groups
        output_dtype: str = "float",
        dropna: bool = True,
        unknown_handling: str = "raises",
        copy: bool = False,
        verbose: bool = False,
        pretty_print: bool = False,
        str_nan: str = "__NAN__",
        str_default: str = "__OTHER__",
    ) -> None:
        """Initiates an ``AutoCarver``.

        Parameters
        ----------
        min_freq : float
            Minimum frequency per grouped modalities.

            * Features whose most frequent modality is less frequent than ``min_freq`` will not be carved.
            * Sets the number of quantiles in which to discretize the continuous features.
            * Sets the minimum frequency of a quantitative feature's modality.

            **Tip**: should be set between 0.02 (slower, preciser, less robust) and 0.05 (faster, more robust)

        sort_by : str
            To be choosen amongst ``["tschuprowt", "cramerv", "kruskal"]``
            Metric to be used to perform association measure between features and target.

            * Binary target: use ``"tschuprowt"``, for Tschuprow's T.
            * Binary target: use ``"cramerv"``, for Cramér's V.
            * Continuous target: use ``"kruskal"``, for Kruskal-Wallis' H test statistic.

            **Tip**: use ``"tschuprowt"`` for more robust, or less output modalities,
            use ``"cramerv"`` for more output modalities.

        quantitative_features : list[str], optional
            List of column names of quantitative features (continuous and discrete) to be carved, by default ``None``

        qualitative_features : list[str], optional
            List of column names of qualitative features (non-ordinal) to be carved, by default ``None``

        ordinal_features : list[str], optional
            List of column names of ordinal features to be carved. For those features a list of
            values has to be provided in the ``values_orders`` dict, by default ``None``

        values_orders : dict[str, GroupedList], optional
            Dict of feature's column names and there associated ordering.
            If lists are passed, a GroupedList will automatically be initiated, by default ``None``

        max_n_mod : int, optional
            Maximum number of modality per feature, by default ``5``

            All combinations of modalities for groups of modalities of sizes from 1 to ``max_n_mod`` will be tested.
            The combination with the greatest association (as defined by ``sort_by``) will be the selected one.

            **Tip**: should be set between 4 (faster, more robust) and 7 (slower, preciser, less robust)

        output_dtype : str, optional
            To be choosen amongst ``["float", "str"]``, by default ``"float"``

            * ``"float"``, grouped modalities will be converted to there corresponding floating rank.
            * ``"str"``, a per-group modality will be set for all the modalities of a group.

        dropna : bool, optional
            * ``True``, ``AutoCarver`` will try to group ``numpy.nan`` with other modalities.
            * ``False``, ``AutoCarver`` all non-``numpy.nan`` will be grouped, by default ``True``

        copy : bool, optional
            If ``True``, feature processing at transform is applied to a copy of the provided DataFrame, by default ``False``

        verbose : bool, optional
            If ``True``, prints raw Discretizers Fit and Transform steps, as long as
            information on AutoCarver's processing and tables of target rates and frequencies for
            X, by default ``False``

        pretty_print : bool, optional
            If ``True``, adds to the verbose some HTML tables of target rates and frequencies for X and, if provided, X_dev.
            Overrides the value of ``verbose``, by default ``False``

        str_nan : str, optional
            String representation to input ``numpy.nan``. If ``dropna=False``, ``numpy.nan`` will be left unchanged, by default ``"__NAN__"``

        str_default : str, optional
            String representation for default qualitative values, i.e. values less frequent than ``min_freq``, by default ``"__OTHER__"``

        Examples
        --------
        See `AutoCarver examples <https://autocarver.readthedocs.io/en/latest/index.html>`_
        """
        # Lists of features
        if quantitative_features is None:
            quantitative_features = []
        if qualitative_features is None:
            qualitative_features = []
        if ordinal_features is None:
            ordinal_features = []
        assert (
            len(quantitative_features) > 0
            or len(qualitative_features) > 0
            or len(ordinal_features) > 0
        ), " - [AutoCarver] No feature passed as input. Pleased provided column names to Carver by setting quantitative_features, quantitative_features or ordinal_features."
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
            output_dtype=output_dtype,
            str_nan=str_nan,
            str_default=str_default,
            dropna=dropna,
            copy=copy,
            verbose=bool(max(verbose, pretty_print)),
        )

        # checking that qualitatitve and qualitative featues are distinct
        assert all(
            quali_feature not in self.quantitative_features
            for quali_feature in self.qualitative_features
        ), " - [AutoCarver] A feature of quantitative_features also is in qualitative_features or ordinal_features. Please, be carreful with your inputs!"
        assert all(
            quanti_feature not in self.qualitative_features
            for quanti_feature in self.quantitative_features
        ), " - [AutoCarver] A feature of qualitative_features or ordinal_features also is in quantitative_features. Please, be carreful with your inputs!"

        # class specific attributes
        self.min_freq = min_freq  # minimum frequency per base bucket
        self.max_n_mod = max_n_mod  # maximum number of modality per feature
        # self.min_carved_freq = min_carved_freq  # TODO
        self.min_group_size = 1
        if pretty_print:
            if _has_idisplay:  # checking for installed dependencies
                self.pretty_print = pretty_print
            else:
                self.verbose = True
                print(
                    "Package not found: ipython. Defaulting to verbose=True. Install extra dependencies with pip install autocarver[jupyter]"
                )

        measures = [
            "tschuprowt",
            "cramerv",
            "kruskal",
        ]  # association measure used to find the best groups
        assert (
            sort_by in measures
        ), f""" - [AutoCarver] Measure '{sort_by}' not yet implemented. Choose from: {str(measures)}."""
        self.sort_by = sort_by

        assert unknown_handling in [
            "drop",
            "raises",
            "best",
            "worst",
        ], " - [AutoCarver] Wrong value for attribute unknown_handling. Choose from ['drop', 'raises', 'best', 'worst']."
        self.unknown_handling = unknown_handling

    def _prepare_data(
        self,
        X: DataFrame,
        y: Series,
        X_dev: DataFrame = None,
        y_dev: Series = None,
    ) -> tuple[DataFrame, DataFrame, dict[str, Callable]]:
        """Validates format and content of X and y.

        Parameters
        ----------
        X : DataFrame
            Dataset used to discretize. Needs to have columns has specified in ``AutoCarver.features``.

        y : Series
            Binary target feature with wich the association is maximized.

        X_dev : DataFrame, optional
            Dataset to evalute the robustness of discretization, by default ``None``
            It should have the same distribution as X.

        y_dev : Series, optional
            Binary target feature with wich the robustness of discretization is evaluated, by default ``None``

        Returns
        -------
        tuple[DataFrame, DataFrame, dict[str, Callable]]
            Copies of (X, X_dev) and helpers to be used according to target type
        """
        # Checking for binary target and copying X
        x_copy = super()._prepare_data(X, y)
        x_dev_copy = super()._prepare_data(X_dev, y_dev)

        # checking for nans in the target
        assert not any(y.isna()), " - [AutoCarver] y should not contain numpy.nan"

        # checking for binary target
        y_values = unique(y)
        is_binary = len(y_values) == 2
        # case 0: binary target, checking values
        if is_binary:
            assert (0 in y_values) & (
                1 in y_values
            ), " - [AutoCarver] y must be a binary Series (int or float, not object)"
            assert (
                len(y_values) == 2
            ), " - [AutoCarver] y must be a binary Series (int or float, not object)"

            # setting up helper functions to be used in autocarver
            helpers = {
                "aggregator": get_xtabs,
                "grouper": xtab_grouper,
                "association_measure": association_xtab,
                "target_rate": xtab_target_rate,
                "combination_formatter": xtab_combination_formatter,
                "printer": pretty_xtab,
            }

            # checking for a corresponding sorting measure
            measures = ["tschuprowt", "cramerv"]
            assert (
                self.sort_by in measures
            ), f""" - [AutoCarver] Measure '{self.sort_by}' not implemented for a binary target. Choose from: {str(measures)}."""

        # case 1: continuous target, checking values
        else:
            not_numeric = str in y.apply(type).unique()
            assert (
                not not_numeric
            ), " - [AutoCarver] y must be a continuous Series (int or float, not object)"
            pct_y = len(y_values) / len(y)
            assert (
                pct_y > 0.01
            ), f" - [AutoCarver] y must be a continuous or binary Series (not implemented for multiclass, only {pct_y:2.2%} distinct values)"

            # setting up helper functions to be used in autocarver
            helpers = {
                "aggregator": get_yvals,
                "grouper": yval_grouper,
                "association_measure": association_yval,
                "target_rate": yval_target_rate,
                "combination_formatter": yval_combination_formatter,
                "printer": pretty_yval,
            }

            # checking for a corresponding sorting measure
            measures = ["kruskal"]
            assert (
                self.sort_by in measures
            ), f""" - [AutoCarver] Measure '{self.sort_by}' not implemented for a binary target. Choose from: {str(measures)}."""

        return x_copy, x_dev_copy, helpers

    def _remove_feature(self, feature: str) -> None:
        """Removes a feature from all ``AutoCarver.features`` attributes

        Parameters
        ----------
        feature : str
            Column name of the feature to remove
        """
        if feature in self.features:
            super()._remove_feature(feature)
            if feature in self.ordinal_features:
                self.ordinal_features.remove(feature)

    def fit(
        self,
        X: DataFrame,
        y: Series,
        *,
        X_dev: DataFrame = None,
        y_dev: Series = None,
    ) -> None:
        """Finds the combination of modalities of X that provides the best association with y.

        Parameters
        ----------
        X : DataFrame
            Dataset used to discretize. Needs to have columns has specified in ``AutoCarver.features``.

        y : Series
            Binary target feature with wich the association is maximized.

        X_dev : DataFrame, optional
            Dataset to evalute the robustness of discretization, by default None
            It should have the same distribution as X.

        y_dev : Series, optional
            Binary target feature with wich the robustness of discretization is evaluated, by default None
        """
        # preparing datasets and checking for wrong values
        x_copy, x_dev_copy, helpers = self._prepare_data(X, y, X_dev, y_dev)

        # discretizing all features
        discretizer = Discretizer(
            quantitative_features=self.quantitative_features,
            qualitative_features=self.qualitative_features,
            min_freq=self.min_freq,
            ordinal_features=self.ordinal_features,
            values_orders=self.values_orders,
            str_nan=self.str_nan,
            str_default=self.str_default,
            # unknwon_handling=self.unknown_handling,  # TODO
            copy=True,  # copying anyways, otherwise no discretization from start to finish
            verbose=self.verbose,
        )
        x_copy = discretizer.fit_transform(x_copy, y)
        if x_dev_copy is not None:
            x_dev_copy = discretizer.transform(x_dev_copy, y_dev)
        self.input_dtypes.update(discretizer.input_dtypes)  # saving data types

        # updating values_orders according to base bucketization
        self.values_orders.update(discretizer.values_orders)

        # removing dropped features
        removed_features = [
            feature for feature in self.features if feature not in discretizer.features
        ]
        for feature in removed_features:
            self._remove_feature(feature)

        # converting quantiles into there respective labels
        labels_orders = convert_to_labels(
            features=self.features,
            quantitative_features=self.quantitative_features,
            values_orders=self.values_orders,
            str_nan=self.str_nan,
            dropna=False,
        )

        # computing crosstabs for each feature on train/test
        xaggs = helpers["aggregator"](self.features, x_copy, y, labels_orders)
        xaggs_dev = helpers["aggregator"](self.features, x_dev_copy, y_dev, labels_orders)

        # optimal butcketization/carving of each feature
        all_features = self.features[:]  # (features are being removed from self.features)
        for n, feature in enumerate(all_features):
            if self.verbose:  # verbose if requested
                print(f"\n------\n[AutoCarver] Fit {feature} ({n+1}/{len(all_features)})\n---")

            # getting xtabs on train/test
            xagg = xaggs[feature]
            xagg_dev = xaggs_dev[feature]
            if self.verbose:  # verbose if requested
                print("\n - [AutoCarver] Raw feature distribution")
                # TODO: get the good labels
                print_xagg(xagg, xagg_dev=xagg_dev, pretty_print=self.pretty_print, **helpers)

            # ordering
            order = labels_orders[feature]

            # getting best combination
            best_combination = self._get_best_combination(order, xagg, helpers, xagg_dev=xagg_dev)

            # checking that a suitable combination has been found
            if best_combination is not None:
                order, xagg, xagg_dev = best_combination
                if self.verbose:  # verbose if requested
                    print("\n - [AutoCarver] Carved feature distribution")
                    # TODO: get the good labels
                    print_xagg(xagg, xagg_dev=xagg_dev, pretty_print=self.pretty_print, **helpers)

                # updating label_orders
                labels_orders.update({feature: order})

            # no suitable combination has been found -> removing feature
            else:
                print(
                    f" - [AutoCarver] No robust combination for feature '{feature}' could be found. It will be ignored. You might have to increase the size of your test sample (test sample not representative of test sample for this feature) or you should consider dropping this features."
                )
                self._remove_feature(feature)
                if feature in labels_orders:
                    labels_orders.pop(feature)

            if self.verbose:  # verbose if requested
                print("------\n")

        # converting potential labels into there respective values (quantiles)
        self.values_orders.update(
            convert_to_values(
                features=self.features,
                quantitative_features=self.quantitative_features,
                values_orders=self.values_orders,
                label_orders=labels_orders,
                str_nan=self.str_nan,
            )
        )

        # discretizing features based on each feature's values_order
        super().fit(X, y)

        return self

    def _get_best_combination(
        self,
        order: GroupedList,
        xagg: DataFrame,
        helpers: dict[str, Callable],
        *,
        xagg_dev: DataFrame = None,
    ) -> tuple[GroupedList, DataFrame, DataFrame]:
        """_summary_

        Parameters
        ----------
        order : GroupedList
            _description_
        xagg : DataFrame
            _description_
        xagg_dev : DataFrame, optional
            _description_, by default None

        Returns
        -------
        tuple[GroupedList, DataFrame, DataFrame]
            _description_
        """
        # raw ordering
        raw_order = GroupedList(order)
        if self.str_nan in raw_order:
            raw_order.remove(self.str_nan)

        # filtering out nans if requested from train/test crosstabs
        raw_xagg = filter_nan(xagg, self.str_nan)
        raw_xagg_dev = filter_nan(xagg_dev, self.str_nan)

        # checking for non-nan values
        best_association = None
        if raw_xagg.shape[0] > 1:
            # all possible consecutive combinations
            combinations = consecutive_combinations(
                raw_order, self.max_n_mod, min_group_size=self.min_group_size
            )

            # getting most associated combination
            best_association = get_best_association(
                raw_xagg,
                combinations,
                sort_by=self.sort_by,
                xagg_dev=raw_xagg_dev,
                verbose=self.verbose,
                **helpers,
            )

            # applying best_combination to order and xtabs
            if best_association is not None:
                order = order_apply_combination(order, best_association["combination"])
                xagg = xagg_apply_order(xagg, order)
                xagg_dev = xagg_apply_order(xagg_dev, order)

            # grouping NaNs if requested to drop them (dropna=True)
            if self.dropna and self.str_nan in order and best_association is not None:
                # raw ordering without nans
                raw_order = GroupedList(order)
                raw_order.remove(self.str_nan)

                # all possible consecutive combinations
                combinations = consecutive_combinations(
                    raw_order, self.max_n_mod, min_group_size=self.min_group_size
                )

                # adding combinations with NaNs
                nan_combinations = add_nan_in_combinations(
                    combinations, self.str_nan, self.max_n_mod
                )

                # getting most associated combination
                best_association = get_best_association(
                    xagg,
                    nan_combinations,
                    sort_by=self.sort_by,
                    xagg_dev=xagg_dev,
                    verbose=self.verbose,
                    **helpers,
                )

                # applying best_combination to order and xtab
                if best_association is not None:
                    order = order_apply_combination(order, best_association["combination"])
                    xagg = xagg_apply_order(xagg, order)
                    xagg_dev = xagg_apply_order(xagg_dev, order)

        # checking that a suitable combination has been found
        if best_association is not None:
            return order, xagg, xagg_dev


def xtab_target_rate(xtab: DataFrame) -> DataFrame:
    """Computes target rate per row for a binary target (column) in a crosstab

    Parameters
    ----------
    xtab : DataFrame
        _description_

    Returns
    -------
    DataFrame
        _description_
    """
    return xtab[1].divide(xtab.sum(axis=0)).sort_values()


def yval_target_rate(yval: Series) -> Series:
    """Computes target rate per row for a binary target (column) in a crosstab

    Parameters
    ----------
    yval : Series
        _description_

    Returns
    -------
    Series
        _description_
    """
    return yval.apply(mean).sort_values()


def filter_nan(xagg: Union[Series, DataFrame], str_nan: str) -> DataFrame:
    """Filters out nans from crosstab or y values

    Parameters
    ----------
    xagg : Union[Series, DataFrame]
        _description_
    str_nan : str
        _description_

    Returns
    -------
    Union[Series, DataFrame]
        _description_
    """
    # cehcking for values in crosstab
    filtered_xagg = None
    if xagg is not None:
        # filtering out nans if requested from train crosstab
        filtered_xagg = xagg.copy()
        if str_nan in xagg.index:
            filtered_xagg = xagg.drop(str_nan, axis=0)

    return filtered_xagg


def get_xtabs(
    features: list[str], X: DataFrame, y: Series, labels_orders: dict[str, GroupedList]
) -> dict[str, DataFrame]:
    """Computes crosstabs for specified features and ensures that the crosstab is ordered according to the known labels

    Parameters
    ----------
    features : list[str]
        _description_
    X : DataFrame
        _description_
    y : Series
        _description_
    labels_orders : dict[str, GroupedList]
        _description_

    Returns
    -------
    dict[str, DataFrame]
        _description_
    """
    # checking for empty datasets
    xtabs = {feature: None for feature in features}
    if X is not None:
        # crosstab for each feature
        for feature in features:
            # computing crosstab with str_nan
            xtab = crosstab(X[feature], y)

            # reordering according to known_order
            xtab = xtab.reindex(labels_orders[feature])

            # storing results
            xtabs.update({feature: xtab})

    return xtabs


def get_yvals(
    features: list[str], X: DataFrame, y: Series, labels_orders: dict[str, GroupedList]
) -> dict[str, DataFrame]:
    """Computes y values for modalities of specified features and ensures the ordering according to the known labels

    Parameters
    ----------
    features : list[str]
        _description_
    X : DataFrame
        _description_
    y : Series
        _description_
    labels_orders : dict[str, GroupedList]
        _description_

    Returns
    -------
    dict[str, DataFrame]
        _description_
    """
    # checking for empty datasets
    yvals = {feature: None for feature in features}
    if X is not None:
        # crosstab for each feature
        for feature in features:
            # computing crosstab with str_nan
            yval = y.groupby(X[feature]).apply(lambda u: list(u))

            # reordering according to known_order
            yval = yval.reindex(labels_orders[feature])

            # storing results
            yvals.update({feature: yval})

    return yvals


def association_xtab(xtab: DataFrame, n_obs: int) -> dict[str, float]:
    """Computes measures of association between feature and target by crosstab.

    Parameters
    ----------
    xtab : DataFrame
        Crosstab between feature and target.

    n_obs : int
        Sample total size.

    Returns
    -------
    dict[str, float]
        Cramér's V and Tschuprow's as a dict.
    """
    # number of values taken by the features
    n_mod_x = xtab.shape[0]

    # Chi2 statistic
    chi2 = chi2_contingency(xtab)[0]

    # Cramér's V
    cramerv = sqrt(chi2 / n_obs)

    # Tschuprow's T
    tschuprowt = cramerv / sqrt(sqrt(n_mod_x - 1))

    return {"cramerv": cramerv, "tschuprowt": tschuprowt}


def association_yval(yval: Series, **kwargs) -> dict[str, float]:
    """Computes measures of association between feature and quantitative target.

    Parameters
    ----------
    yval : DataFrame
        Values taken by y for each of x's modalities.

    Returns
    -------
    dict[str, float]
        Kruskal-Wallis' H as a dict.
    """
    # Kruskal-Wallis' H
    return {"kruskal": kruskal(*tuple(yval.values))[0]}


def yval_grouper(yval: Series, groupby: dict[str:str]) -> Series:
    grouped_yval = yval.groupby(groupby).sum()

    return grouped_yval


def xtab_grouper(xtab: DataFrame, groupby: list[str]) -> DataFrame:
    """Groups a crosstab by groupby and sums column values by groups (vectorized)

    Parameters
    ----------
    xtab : DataFrame
        _description_
    groupby : list[str]
        _description_

    Returns
    -------
    DataFrame
        _description_
    """
    # all indices that may be duplicated
    index_values = array(groupby)

    # all unique indices deduplicated
    unique_indices = unique(index_values)

    # initiating summed up array with zeros
    summed_values = zeros((len(unique_indices), len(xtab.columns)))

    # for each unique_index found in index_values sums xtab.Values at corresponding position in summed_values
    add.at(summed_values, searchsorted(unique_indices, index_values), xtab.values)

    # converting back to dataframe
    grouped_xtab = DataFrame(summed_values, index=unique_indices, columns=xtab.columns)

    return grouped_xtab


def combinations_at_index(
    start_idx: int, order: list[Any], nb_remaining_groups: int, min_group_size: int = 1
) -> tuple[list[Any], int, int]:
    """Gets all possible combinations of sizes up to the last element of a list

    Parameters
    ----------
    start_idx : int
        _description_
    order : list[Any]
        _description_
    nb_remaining_groups : int
        _description_
    min_group_size : int, optional
        _description_, by default 1

    Returns
    -------
    tuple[list[Any], int, int]
        _description_

    Yields
    ------
    Iterator[tuple[list[Any], int, int]]
        _description_
    """
    # iterating over each possible length of groups
    for size in range(min_group_size, len(order) + 1):
        next_idx = start_idx + size  # index from which to start the next group

        # checking that next index is not off the order list
        if next_idx < len(order) + 1:
            # checking that there are remaining groups or that it is the last group
            if (nb_remaining_groups > 1) | (next_idx == len(order)):
                combination = list(order[start_idx:next_idx])
                yield (combination, next_idx, nb_remaining_groups - 1)


def consecutive_combinations(
    raw_order: list[Any],
    max_group_size: int,
    min_group_size: int = 1,
    nb_remaining_group: int = None,
    current_combination: list[Any] = None,
    next_index: int = None,
    all_combinations: list[list[Any]] = None,
) -> list[list[Any]]:
    """Computes all possible combinations of values of order up to max_group_size.

    Parameters
    ----------
    raw_order : list[Any]
        _description_
    max_group_size : int
        _description_
    min_group_size : int, optional
        _description_, by default 1
    nb_remaining_group : int, optional
        _description_, by default None
    current_combination : list[Any], optional
        _description_, by default None
    next_index : int, optional
        _description_, by default None
    all_combinations : list[list[Any]], optional
        _description_, by default None

    Returns
    -------
    list[list[Any]]
        _description_
    """
    # initiating recursive attributes
    if current_combination is None:
        current_combination = []
    if next_index is None:
        next_index = 0
    if nb_remaining_group is None:
        nb_remaining_group = max_group_size
    if all_combinations is None:
        all_combinations = []

    # getting combinations for next index
    next_combinations = [
        elt
        for elt in combinations_at_index(next_index, raw_order, nb_remaining_group, min_group_size)
    ]

    # stop case: no next_combinations possible -> adding to all_combinations
    if len(next_combinations) == 0 and min_group_size < len(current_combination) <= max_group_size:
        # saving up combination
        all_combinations += [current_combination]

        # resetting remaining number of groups
        nb_remaining_group = max_group_size

    # otherwise: adding all next_combinations to the current_combination
    for combination, next_index, current_nb_remaining_group in next_combinations:
        # going a rank further in the raw_xtab
        consecutive_combinations(
            raw_order,
            max_group_size,
            min_group_size=min_group_size,
            nb_remaining_group=current_nb_remaining_group,
            current_combination=current_combination + [combination],
            next_index=next_index,
            all_combinations=all_combinations,
        )

    return all_combinations


def xtab_combination_formatter(combination: list[list[str]]) -> list[str]:
    formatted_combination = [
        value for values in ([group[0]] * len(group) for group in combination) for value in values
    ]

    return formatted_combination


def yval_combination_formatter(combination: list[list[str]]) -> dict[str, str]:
    formatted_combination = {modal: group[0] for group in combination for modal in group}

    return formatted_combination


def get_best_association(
    xagg: Union[Series, DataFrame],
    combinations: list[list[str]],
    sort_by: str,
    *,
    xagg_dev: Union[Series, DataFrame] = None,
    verbose: bool = False,
    grouper: Callable = xtab_grouper,
    combination_formatter: Callable = xtab_combination_formatter,
    association_measure: Callable = association_xtab,
    target_rate: Callable = xtab_target_rate,
    **kwargs,
) -> dict[str, Any]:
    """Computes associations of the tab for each combination

    Parameters
    ----------
    xagg : Union[Series, DataFrame]
        _description_
    combinations : list[list[str]]
        _description_
    sort_by : str
        _description_
    xagg_dev : Union[Series, DataFrame], optional
        _description_, by default None
    verbose : bool, optional
        _description_, by default False

    Returns
    -------
    dict[str, Any]
        _description_
    """
    # values to groupby indices with
    indices_to_groupby = [combination_formatter(combination) for combination in combinations]

    # grouping tab by its indices
    grouped_xaggs = [
        grouper(xagg, index_to_groupby)
        for index_to_groupby in tqdm(
            indices_to_groupby, disable=not verbose, desc="Grouping modalities   "
        )
    ]

    # computing associations for each tabs
    n_obs = xagg.apply(sum).sum()  # number of observations for xtabs
    associations_xagg = [
        association_measure(grouped_xagg, n_obs=n_obs)
        for grouped_xagg in tqdm(grouped_xaggs, disable=not verbose, desc="Computing associations")
    ]

    # adding corresponding combination to the association
    for combination, index_to_groupby, association, grouped_xagg in zip(
        combinations, indices_to_groupby, associations_xagg, grouped_xaggs
    ):
        association.update(
            {"combination": combination, "index_to_groupby": index_to_groupby, "xagg": grouped_xagg}
        )

    # sorting associations according to specified metric
    associations_xagg = (
        DataFrame(associations_xagg).sort_values(sort_by, ascending=False).to_dict(orient="records")
    )

    # case 0: no test sample provided -> not testing for robustness
    if xagg_dev is None:
        return associations_xagg[0]

    # case 1: testing viability on provided test sample
    for association in tqdm(associations_xagg, disable=not verbose, desc="Testing robustness    "):
        # needed parameters
        index_to_groupby, xagg = (
            association["index_to_groupby"],
            association["xagg"],
        )

        # grouping rows of the test crosstab
        grouped_xagg_dev = grouper(xagg_dev, index_to_groupby)

        # computing target rate ranks per value
        train_ranks = target_rate(xagg).index
        test_ranks = target_rate(grouped_xagg_dev).index

        # viable on test sample: grouped values have the same ranks in train/test
        if all(train_ranks == test_ranks):
            return association


def add_nan_in_combinations(
    combinations: list[list[str]], str_nan: str, max_n_mod: int
) -> list[list[str]]:
    """Adds nan to each possible group and a last group only with nan if the max_n_mod is not reached by the combination

    Parameters
    ----------
    combinations : list[list[str]]
        _description_
    str_nan : str
        _description_
    max_n_mod : int
        _description_

    Returns
    -------
    list[list[str]]
        _description_
    """
    # iterating over each combination
    nan_combinations = []
    for combination in combinations:
        # adding nan to each group of the combination
        nan_combination = []
        for n in range(len(combination)):
            # copying input combination
            new_combination = combination[:]
            # adding nan to the nth group
            new_combination[n] = new_combination[n] + [str_nan]
            # storing updated combination with attributed group to nan
            nan_combination += [new_combination]

        # if max_n_mod is not reached adding a combination with nans alone
        if len(combination) < max_n_mod:
            # copying input combination
            new_combination = combination[:]
            # adding a group for nans only
            nan_combination += [new_combination + [[str_nan]]]

        nan_combinations += nan_combination

    return nan_combinations


def order_apply_combination(order: GroupedList, combination: list[list[Any]]) -> GroupedList:
    """Converts a list of combination to a GroupedList

    Parameters
    ----------
    order : GroupedList
        _description_
    combination : list[list[Any]]
        _description_

    Returns
    -------
    GroupedList
        _description_
    """
    order_copy = GroupedList(order)
    for combi in combination:
        order_copy.group_list(combi, combi[0])

    return order_copy


def xagg_apply_order(xagg: DataFrame, order: GroupedList) -> DataFrame:
    """Applies an order (combination) to a crosstab

    Parameters
    ----------
    xagg : DataFrame
        Crosstab
    order : GroupedList
        Combination of index to apply to the crosstab

    Returns
    -------
    dict[str, Any]
        Orderd crosstab.
    """
    # checking for input values
    combi_xagg = None
    if xagg is not None:
        # grouping modalities in the crosstab
        groups = list(map(order.get_group, xagg.index))
        combi_xagg = xagg.groupby(groups, dropna=False, sort=False).sum()

    return combi_xagg


def load_carver(auto_carver_json: dict) -> BaseDiscretizer:
    """Allows one to load an AutoCarver saved as a .json file.

    The AutoCarver has to be saved with ``json.dump(f, AutoCarver.to_json())``, otherwise there
    can be no guarantee for it to be restored.

    Parameters
    ----------
    auto_carver_json : str
        Loaded .json file using ``json.load(f)``.

    Returns
    -------
    BaseDiscretizer
        A fitted AutoCarver.
    """
    # deserializing values_orders
    values_orders = json_deserialize_values_orders(auto_carver_json["values_orders"])

    # updating auto_carver attributes
    auto_carver_json.update({"values_orders": values_orders})

    # initiating BaseDiscretizer
    auto_carver = BaseDiscretizer(**auto_carver_json)
    auto_carver.fit()

    return auto_carver


def pretty_xtab(xtab: DataFrame = None) -> DataFrame:
    """Prints a binary xtab's statistics

    Parameters
    ----------
    xtab : Dataframe
        A crosstab, by default None

    Returns
    -------
    DataFrame
        Target rate and frequency per modality
    """
    # checking for an xtab
    stats = None
    if xtab is not None:
        # target rate and frequency statistics per modality
        stats = DataFrame(
            {
                # target rate per modality
                "target_rate": xtab[1].divide(xtab.sum(axis=1)),
                # frequency per modality
                "frequency": xtab.sum(axis=1) / xtab.sum().sum(),
            }
        )

        # rounding up stats
        stats = stats.round(3)

    return stats


def pretty_yval(yval: Series = None) -> DataFrame:
    """Prints a continuous yval's statistics

    Parameters
    ----------
    yval : Series
        A series of values of y by modalities of x, by default None

    Returns
    -------
    DataFrame
        Target rate and frequency per modality
    """
    # checking for an xtab
    stats = None
    if yval is not None:
        # target rate and frequency statistics per modality
        stats = DataFrame(
            {
                # target rate per modality
                "target_rate": yval.apply(mean),
                # frequency per modality
                "frequency": yval.apply(len) / yval.apply(len).sum(),
            }
        )

        # rounding up stats
        stats = stats.round(3)

    return stats


def prettier_xagg(
    nice_xagg: DataFrame = None,
    caption: str = None,
) -> str:
    """Converts a crosstab to the HTML format, adding nice colors

    Parameters
    ----------
    nice_xagg : DataFrame, optional
        Target rate and frequency per modality, by default None
    caption : str, optional
        Title of the HTML table, by default None

    Returns
    -------
    str
        HTML format of the crosstab
    """
    """Pretty display of frequency and target rate per modality on the same line."""
    # checking for a provided xtab
    nicer_xagg = ""
    if nice_xagg is not None:
        # adding coolwarn color gradient
        nicer_xagg = nice_xagg.style.background_gradient(cmap="coolwarm")

        # printing inline notebook
        nicer_xagg = nicer_xagg.set_table_attributes("style='display:inline'")

        # adding custom caption/title
        if caption is not None:
            nicer_xagg = nicer_xagg.set_caption(caption)

        # converting to html
        nicer_xagg = nicer_xagg._repr_html_()

    return nicer_xagg


def print_xagg(
    xagg: DataFrame,
    printer: Callable,
    xagg_dev: DataFrame = None,
    pretty_print: bool = False,
    **kwargs,
) -> None:
    """Prints crosstabs' target rates and frequencies per modality, in raw or html format

    Parameters
    ----------
    xagg : DataFrame
        Train crosstab
    xagg_dev : DataFrame
        Dev crosstab, by default None
    pretty_print : bool, optional
        Whether to output html or not, by default False
    """
    # getting pretty xtabs
    nice_xagg = printer(xagg)
    nice_xagg_dev = printer(xagg_dev)

    # case 0: no pretty hmtl printing
    if not pretty_print:
        print(nice_xagg, "\n")

    # case 1: pretty html printing
    else:
        # getting prettier xtabs
        nicer_xagg = prettier_xagg(nice_xagg, caption="X distribution")
        nicer_xagg_dev = prettier_xagg(nice_xagg_dev, caption="X_dev distribution")

        # merging outputs
        nicer_xaggs = nicer_xagg + "    " + nicer_xagg_dev

        # displaying html of colored DataFrame
        display_html(nicer_xaggs, raw=True)
