"""Tool to build optimized buckets out of Quantitative and Qualitative features
for any task.
"""

from typing import Any, Callable, Union
from warnings import warn

from pandas import DataFrame, Series
from tqdm import tqdm

from ..discretizers import GroupedList
from ..discretizers.discretizers import Discretizer
from ..discretizers.utils.base_discretizers import (
    BaseDiscretizer,
    convert_to_labels,
    convert_to_values,
)
from ..discretizers.utils.serialization import json_deserialize_values_orders

# trying to import extra dependencies
try:
    from IPython.display import display_html
except ImportError:
    _has_idisplay = False
else:
    _has_idisplay = True


class BaseCarver(BaseDiscretizer):
    """Automatic carving of continuous, discrete, categorical and ordinal
    features that maximizes association with a binary or continuous target.

    First fits a :class:`Discretizer`. Raw data should be provided as input (not a result of ``Discretizer.transform()``).
    """

    def __init__(
        self,
        sort_by: str,
        min_freq: float,
        *,
        quantitative_features: list[str] = None,
        qualitative_features: list[str] = None,
        ordinal_features: list[str] = None,
        values_orders: dict[str, GroupedList] = None,
        max_n_mod: int = 5,
        min_freq_mod: float = None,
        output_dtype: str = "float",
        dropna: bool = True,
        copy: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """
        min_freq : float
            Minimum frequency per grouped modalities.

            * Features whose most frequent modality is less frequent than ``min_freq`` will not be carved.
            * Sets the number of quantiles in which to discretize the continuous features.
            * Sets the minimum frequency of a quantitative feature's modality.

            **Tip**: should be set between 0.02 (slower, preciser, less robust) and 0.05 (faster, more robust)

        quantitative_features : list[str], optional
            List of column names of quantitative features (continuous and discrete) to be carved, by default ``None``

        qualitative_features : list[str], optional
            List of column names of qualitative features (non-ordinal) to be carved, by default ``None``

        ordinal_features : list[str], optional
            List of column names of ordinal features to be carved. For those features a list of
            values has to be provided in the ``values_orders`` dict, by default ``None``

        values_orders : dict[str, GroupedList], optional
            Dict of feature's column names and there associated ordering.
            If lists are passed, a :class:`GroupedList` will automatically be initiated, by default ``None``

        max_n_mod : int, optional
            Maximum number of modality per feature, by default ``5``

            All combinations of modalities for groups of modalities of sizes from 1 to ``max_n_mod`` will be tested.
            The combination with the best association will be selected.

            **Tip**: should be set between 4 (faster, more robust) and 7 (slower, preciser, less robust)

        min_freq_mod : float, optional
            Minimum frequency per final modality, by default ``None`` for min_freq

        output_dtype : str, optional
            To be choosen amongst ``["float", "str"]``, by default ``"float"``

            * ``"float"``, grouped modalities will be converted to there corresponding floating rank.
            * ``"str"``, a per-group modality will be set for all the modalities of a group.

        dropna : bool, optional
            * ``True``, try to group ``numpy.nan`` with other modalities.
            * ``False``, all non-``numpy.nan`` will be grouped, by default ``True``

        copy : bool, optional
            If ``True``, feature processing at transform is applied to a copy of the provided DataFrame, by default ``False``

        verbose : bool, optional
            * ``True``, without IPython installed: prints raw Discretizers and AutoCarver Fit steps for X, by default ``False``
            * ``True``, with IPython installed: adds HTML tables of target rates and frequencies for X and X_dev.

            **Tip**: IPython displaying can be turned off by setting ``pretty_print=False``.

        **kwargs
            Pass values for ``str_default`` and ``str_nan`` (default string values), as long as ``pretty_print`` to turn off IPython.

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
        ), (
            " - [BaseCarver] No feature passed as input. Pleased provided column names to Carver "
            "by setting quantitative_features, quantitative_features or ordinal_features."
        )
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
            str_nan=kwargs.get("str_nan", "__NAN__"),
            str_default=kwargs.get("str_default", "__OTHER__"),
            dropna=dropna,
            copy=copy,
            verbose=bool(max(verbose, kwargs.get("pretty_print", False))),
        )

        # checking that qualitatitve and quantitative features are distinct
        assert all(
            quali_feature not in self.quantitative_features
            for quali_feature in self.qualitative_features
        ), (
            " - [BaseCarver] One of quantitative_features is also in qualitative_features "
            "or ordinal_features. Please, be carreful with your inputs!"
        )
        assert all(
            quanti_feature not in self.qualitative_features
            for quanti_feature in self.quantitative_features
        ), (
            " - [BaseCarver] One of qualitative_features or ordinal_features is also in "
            "quantitative_features. Please, be carreful with your inputs!"
        )

        # class specific attributes
        self.min_freq = min_freq  # minimum frequency per base bucket
        self.max_n_mod = max_n_mod  # maximum number of modality per feature
        if min_freq_mod is None:
            min_freq_mod = min_freq
        self.min_freq_mod = min_freq_mod  # minimum frequency per final bucket
        self.sort_by = sort_by
        self.pretty_print = False
        if self.verbose and kwargs.get("pretty_print", True):
            if _has_idisplay:  # checking for installed dependencies
                self.pretty_print = True
            else:
                warn(
                    "Package not found: IPython. Defaulting to raw verbose. "
                    "Install extra dependencies with pip install autocarver[jupyter]",
                    UserWarning,
                )

    def _prepare_data(
        self,
        X: DataFrame,
        y: Series,
        X_dev: DataFrame = None,
        y_dev: Series = None,
    ) -> tuple[DataFrame, DataFrame]:
        """Validates format and content of X and y.

        Parameters
        ----------
        X : DataFrame
            Dataset used to discretize. Needs to have columns has specified in ``BaseCarver.features``.

        y : Series
            Binary target feature with wich the association is maximized.

        X_dev : DataFrame, optional
            Dataset to evalute the robustness of discretization, by default ``None``
            It should have the same distribution as X.

        y_dev : Series, optional
            Binary target feature with wich the robustness of discretization is evaluated, by default ``None``

        Returns
        -------
        tuple[DataFrame, DataFrame]
            Copies of (X, X_dev)
        """
        # Checking for binary target and copying X
        x_copy = super()._prepare_data(X, y)
        x_dev_copy = super()._prepare_data(X_dev, y_dev)

        # checking for not provided y
        assert y is not None, f" - [BaseCarver] y must be provided {y}"

        return x_copy, x_dev_copy

    def _aggregator():
        """HELPER: get_xtabs or get_yvals"""
        pass

    def _grouper():
        """HELPER: xtab_grouper or yval_grouper"""
        pass

    def _association_measure():
        """HELPER: association_xtab or association_yval"""
        pass

    def _target_rate():
        """HELPER: xtab_target_rate or yval_target_rate"""
        pass

    def _combination_formatter(self, combination: list[list[str]]) -> dict[str, str]:
        """Attributes the first element of a group to all elements of a group

        Parameters
        ----------
        combination : list[list[str]]
            _description_

        Returns
        -------
        dict[str, str]
            _description_
        """
        return {modal: group[0] for group in combination for modal in group}

    def _printer():
        """HELPER: pretty_xtab or pretty_yval"""
        pass

    def _remove_feature(self, feature: str) -> None:
        """Removes a feature from all ``BaseCarver.features`` attributes

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
            Training dataset, to determine features' optimal carving.
            Needs to have columns has specified in ``features`` attribute.

        y : Series
            Target with wich the association is maximized.

        X_dev : DataFrame, optional
            Development dataset, to evaluate robustness of carved features, by default ``None``
            Should have the same distribution as X.

        y_dev : Series, optional
            Target of the development dataset, by default ``None``
            Should have the same distribution as y.
        """
        # preparing datasets and checking for wrong values
        x_copy, x_dev_copy = self._prepare_data(X, y, X_dev, y_dev)

        # discretizing all features
        discretizer = Discretizer(
            quantitative_features=self.quantitative_features,
            qualitative_features=self.qualitative_features,
            min_freq=self.min_freq,
            ordinal_features=self.ordinal_features,
            values_orders=self.values_orders,
            str_nan=self.str_nan,
            str_default=self.str_default,
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
        xaggs = self._aggregator(self.features, x_copy, y, labels_orders)
        xaggs_dev = self._aggregator(self.features, x_dev_copy, y_dev, labels_orders)

        # optimal butcketization/carving of each feature
        all_features = self.features[:]  # (features are being removed from self.features)
        for n, feature in enumerate(all_features):
            if self.verbose:  # verbose if requested
                print(f"\n------\n[BaseCarver] Fit {feature} ({n+1}/{len(all_features)})\n---")

            # getting xtabs on train/test
            xagg = xaggs[feature]
            xagg_dev = xaggs_dev[feature]
            if self.verbose:  # verbose if requested
                print("\n - [BaseCarver] Raw feature distribution")
                # TODO: get the good labels
                print_xagg(
                    xagg,
                    xagg_dev=xagg_dev,
                    pretty_print=self.pretty_print,
                    printer=self._printer,
                )

            # ordering
            order = labels_orders[feature]

            # getting best combination
            best_combination = self._get_best_combination(order, xagg, xagg_dev=xagg_dev)

            # checking that a suitable combination has been found
            if best_combination is not None:
                order, xagg, xagg_dev = best_combination
                if self.verbose:  # verbose if requested
                    print("\n - [BaseCarver] Carved feature distribution")
                    # TODO: get the good labels
                    print_xagg(
                        xagg,
                        xagg_dev=xagg_dev,
                        pretty_print=self.pretty_print,
                        printer=self._printer,
                    )

                # updating label_orders
                labels_orders.update({feature: order})

            # no suitable combination has been found -> removing feature
            else:
                warn(
                    f" - [BaseCarver] No robust combination for feature '{feature}' could be found"
                    ". It will be ignored. You might have to increase the size of your dev sample"
                    " (dev sample not representative of dev sample for this feature) or you"
                    " should consider dropping this features.",
                    UserWarning,
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
            combinations = consecutive_combinations(raw_order, self.max_n_mod, min_group_size=1)

            # getting most associated combination
            best_association = self.get_best_association(
                raw_xagg,
                combinations,
                xagg_dev=raw_xagg_dev,
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
                combinations = consecutive_combinations(raw_order, self.max_n_mod, min_group_size=1)

                # adding combinations with NaNs
                nan_combinations = add_nan_in_combinations(
                    combinations, self.str_nan, self.max_n_mod
                )

                # getting most associated combination
                best_association = self.get_best_association(
                    xagg,
                    nan_combinations,
                    xagg_dev=xagg_dev,
                )

                # applying best_combination to order and xtab
                if best_association is not None:
                    order = order_apply_combination(order, best_association["combination"])
                    xagg = xagg_apply_order(xagg, order)
                    xagg_dev = xagg_apply_order(xagg_dev, order)

        # checking that a suitable combination has been found
        if best_association is not None:
            return order, xagg, xagg_dev

    def get_best_association(
        self,
        xagg: Union[Series, DataFrame],
        combinations: list[list[str]],
        *,
        xagg_dev: Union[Series, DataFrame] = None,
    ) -> dict[str, Any]:
        """Computes associations of the tab for each combination

        Parameters
        ----------
        xagg : Union[Series, DataFrame]
            _description_
        combinations : list[list[str]]
            _description_
        xagg_dev : Union[Series, DataFrame], optional
            _description_, by default None

        Returns
        -------
        dict[str, Any]
            _description_
        """
        # values to groupby indices with  # TODO: !!!!!!!!!! not working as expected with nans
        indices_to_groupby = [
            self._combination_formatter(combination) for combination in combinations
        ]

        # grouping tab by its indices
        grouped_xaggs = [
            self._grouper(xagg, index_to_groupby)
            for index_to_groupby in tqdm(
                indices_to_groupby, disable=not self.verbose, desc="Grouping modalities   "
            )
        ]

        # computing associations for each tabs
        n_obs = xagg.apply(sum).sum()  # number of observations for xtabs
        associations_xagg = [
            self._association_measure(grouped_xagg, n_obs=n_obs)
            for grouped_xagg in tqdm(
                grouped_xaggs, disable=not self.verbose, desc="Computing associations"
            )
        ]

        # adding corresponding combination to the association
        for combination, index_to_groupby, association, grouped_xagg in zip(
            combinations, indices_to_groupby, associations_xagg, grouped_xaggs
        ):
            association.update(
                {
                    "combination": combination,
                    "index_to_groupby": index_to_groupby,
                    "xagg": grouped_xagg,
                }
            )

        # sorting associations according to specified metric
        associations_xagg = (
            DataFrame(associations_xagg)
            .sort_values(self.sort_by, ascending=False)
            .to_dict(orient="records")
        )

        # testing viability
        for association in tqdm(
            associations_xagg, disable=not self.verbose, desc="Testing robustness    "
        ):
            # needed parameters
            index_to_groupby, grouped_xagg = (
                association["index_to_groupby"],
                association["xagg"],
            )

            # computing target rate and frequency per value
            train_rates = self._printer(grouped_xagg)

            # viable on train sample:
            # - target rates are distinct for all modalities
            # - minimum frequency is reached for all modalities
            if (
                all(train_rates["frequency"] >= self.min_freq_mod)
                and len(train_rates) == train_rates["target_rate"].nunique()
            ):
                # case 0: no test sample provided -> not testing for robustness
                if xagg_dev is None:
                    return association

                # grouping the dev sample per modality
                grouped_xagg_dev = self._grouper(xagg_dev, index_to_groupby)

                # computing target rate and frequency per modality
                dev_rates = self._printer(grouped_xagg_dev)

                # case 1: testing viability on provided dev sample
                # - grouped values have the same ranks in train/test
                # - target rates are distinct for all modalities
                # - minimum frequency is reached for all modalities
                if (
                    all(
                        train_rates.sort_values("target_rate").index
                        == dev_rates.sort_values("target_rate").index
                    )
                    and all(dev_rates["frequency"] >= self.min_freq_mod)
                    and len(dev_rates) == dev_rates["target_rate"].nunique()
                ):
                    return association


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
        # adding coolwarm color gradient
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
        if xagg_dev is not None:
            print("X_dev distribution\n", nice_xagg_dev, "\n")

    # case 1: pretty html printing
    else:
        # getting prettier xtabs
        nicer_xagg = prettier_xagg(nice_xagg, caption="X distribution")
        nicer_xagg_dev = prettier_xagg(nice_xagg_dev, caption="X_dev distribution")

        # merging outputs
        nicer_xaggs = nicer_xagg + "    " + nicer_xagg_dev

        # displaying html of colored DataFrame
        display_html(nicer_xaggs, raw=True)
