"""Tool to build optimized buckets out of Quantitative and Qualitative features
for any task.
"""

from typing import Any, Union
from warnings import warn

from numpy import isclose
from pandas import DataFrame, Series
from tqdm import tqdm

from ..discretizers import GroupedList
from ..discretizers.discretizers import Discretizer
from ..discretizers.utils.base_discretizers import (
    BaseDiscretizer,
    convert_to_labels,
    convert_to_values,
    load_discretizer,
)

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
        **kwargs: dict,
    ) -> None:
        """
        min_freq : float
            Minimum frequency per grouped modalities.

            * Features whose most frequent modality is less frequent than ``min_freq`` will not be carved.
            * Sets the number of quantiles in which to discretize the continuous features.
            * Sets the minimum frequency of a quantitative feature's modality.

            **Tip**: should be set between ``0.01`` (slower, preciser, less robust) and ``0.2`` (faster, more robust)

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

            **Tip**: should be set between ``3`` (faster, more robust) and ``7`` (slower, preciser, less robust)

        min_freq_mod : float, optional
            Minimum frequency per final modality, by default ``None`` for ``min_freq/2``

        output_dtype : str, optional
            To be choosen amongst ``["float", "str"]``, by default ``"float"``

            * ``"float"``, sets the rank of modalities as label.
            * ``"str"``, sets one modality of group as label.

        dropna : bool, optional
            * ``True``, try to group ``numpy.nan`` with other modalities.
            * ``False``, ``numpy.nan`` are ignored (not grouped), by default ``True``

        copy : bool, optional
            If ``True``, feature processing at transform is applied to a copy of the provided DataFrame, by default ``False``

        verbose : bool, optional
            * ``True``, without IPython installed: prints raw Discretizers and AutoCarver Fit steps for X, by default ``False``
            * ``True``, with IPython installed: adds HTML tables of target rates and frequencies for X and X_dev

            **Tip**: IPython displaying can be turned off by setting ``pretty_print=False``

        **kwargs: dict
            Pass values for ``str_default`` and ``str_nan`` (default string values),
            as long as ``pretty_print`` to turn off IPython
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
            " - [AutoCarver] No feature passed as input. Pleased provided column names to Carver "
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
            " - [AutoCarver] One of quantitative_features is also in qualitative_features "
            "or ordinal_features. Please, be carreful with your inputs!"
        )
        assert all(
            quanti_feature not in self.qualitative_features
            for quanti_feature in self.quantitative_features
        ), (
            " - [AutoCarver] One of qualitative_features or ordinal_features is also in "
            "quantitative_features. Please, be carreful with your inputs!"
        )

        # class specific attributes
        self.min_freq = min_freq  # minimum frequency per base bucket
        self.max_n_mod = max_n_mod  # maximum number of modality per feature
        if min_freq_mod is None:
            min_freq_mod = min_freq / 2
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

        # historizing everything
        self._history = {feature: [] for feature in self.features}

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
        assert y is not None, f" - [AutoCarver] y must be provided {y}"

        return x_copy, x_dev_copy

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
            if feature in self._history:
                self._history[feature] += [{"removed": True}]

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
                print(f"\n------\n[AutoCarver] Fit {feature} ({n+1}/{len(all_features)})\n---")

            # carving the feature
            labels_orders = self._carve_feature(feature, xaggs, xaggs_dev, labels_orders)

            if self.verbose:  # verbose if requested
                print("------\n")

        # discretizing features based on each feature's values_order
        super().fit(X, y)

        return self

    def _carve_feature(
        self,
        feature: str,
        xaggs: dict[str, Union[Series, DataFrame]],
        xaggs_dev: dict[str, Union[Series, DataFrame]],
        labels_orders: dict[str, GroupedList],
    ) -> dict[str, GroupedList]:
        """Carves a feature into buckets that maximize association with the target

        Parameters
        ----------
        feature : str
            _description_
        xaggs : dict[str, Union[Series, DataFrame]]
            _description_
        xaggs_dev : dict[str, Union[Series, DataFrame]]
            _description_
        labels_orders : dict[str, GroupedList]
            _description_

        Returns
        -------
        dict[str, GroupedList]
            _description_
        """
        # getting xtabs on train/test
        xagg = xaggs[feature]
        xagg_dev = xaggs_dev[feature]

        # checking that the feature has at least 2 modalities
        best_combination = None
        if len(xagg.index) > 1:
            # ordering
            order = labels_orders[feature]

            # historizing raw combination
            raw_association = {
                "index_to_groupby": {modality: modality for modality in xagg.index},
                self.sort_by: self._association_measure(xagg, n_obs=xagg.apply(sum).sum())[
                    self.sort_by
                ],
            }
            self._historize_viability_test(feature, raw_association, order)

            # verbose
            self._print_xagg(
                xagg=self._index_mapper(feature, labels_orders, xagg),
                xagg_dev=self._index_mapper(feature, labels_orders, xagg_dev),
                message="Raw distribution",
            )

            # getting best combination
            best_combination = self._get_best_combination(feature, order, xagg, xagg_dev=xagg_dev)

        # checking that a suitable combination has been found
        if best_combination is not None:
            order, xagg, xagg_dev = best_combination
            # updating orders accordingly
            labels_orders = self._update_orders(feature, order, labels_orders)

            # verbose
            self._print_xagg(
                xagg=self._index_mapper(feature, labels_orders, xagg),
                xagg_dev=self._index_mapper(feature, labels_orders, xagg_dev),
                message="Carved distribution",
            )

        # no suitable combination has been found -> removing feature
        else:
            warn(
                f" - [AutoCarver] No robust combination for feature '{feature}' could be found"
                ". It will be ignored. You might have to increase the size of your dev sample"
                " (dev sample not representative of dev sample for this feature) or you"
                " should consider dropping this features.",
                UserWarning,
            )
            self._remove_feature(feature)
            if feature in labels_orders:
                labels_orders.pop(feature)

        return labels_orders

    def _update_orders(
        self, feature: str, new_order: GroupedList, labels_orders: dict[str, GroupedList]
    ) -> dict[str, GroupedList]:
        """updates values_orders and labels_orders accoding to the new order for specified feature"""

        # updating label_orders
        labels_orders.update({feature: new_order})

        # updating values_orders
        self.values_orders.update(
            convert_to_values(
                features=[feature],
                quantitative_features=[feature] if feature in self.quantitative_features else [],
                values_orders=self.values_orders,
                label_orders=labels_orders,
                str_nan=self.str_nan,
            )
        )
        # updating labels
        labels_orders = convert_to_labels(
            features=self.features,
            quantitative_features=self.quantitative_features,
            values_orders=self.values_orders,
            str_nan=self.str_nan,
            dropna=False,
        )

        return labels_orders

    def _get_best_combination(
        self,
        feature: str,
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
            best_association, order = self._get_best_association(
                feature,
                order,
                raw_xagg,
                combinations,
                xagg_dev=raw_xagg_dev,
            )

            # applying best_combination to order and xtabs
            if best_association is not None:
                xagg = xagg_apply_order(xagg, order)
                xagg_dev = xagg_apply_order(xagg_dev, order)

            # grouping NaNs if requested to drop them (dropna=True)
            if self.dropna and self.str_nan in order and best_association is not None:
                # raw ordering without nans
                raw_order = GroupedList(order)
                raw_order.remove(self.str_nan)

                # adding combinations with NaNs
                combinations = nan_combinations(raw_order, self.str_nan, self.max_n_mod, 1)

                # getting most associated combination
                best_association, order = self._get_best_association(
                    feature,
                    order,
                    xagg,
                    combinations,
                    xagg_dev=xagg_dev,
                    dropna=True,
                )

                # applying best_combination to order and xtab
                if best_association is not None:
                    xagg = xagg_apply_order(xagg, order)
                    xagg_dev = xagg_apply_order(xagg_dev, order)

        # checking that a suitable combination has been found
        if best_association is not None:
            return order, xagg, xagg_dev

    def _get_best_association(
        self,
        feature: str,
        order: GroupedList,
        xagg: Union[Series, DataFrame],
        combinations: list[list[str]],
        *,
        xagg_dev: Union[Series, DataFrame] = None,
        dropna: bool = False,
    ) -> tuple[dict[str, Any], GroupedList]:
        """Computes associations of the tab for each combination

        Parameters
        ----------
        feature : str
            feature to measure association with
        xagg : Union[Series, DataFrame]
            _description_
        order : GroupedList
            order of the feature
        combinations : list[list[str]]
            _description_
        xagg_dev : Union[Series, DataFrame], optional
            _description_, by default None

        Returns
        -------
        tuple[dict[str, Any], GroupedList]
            best viable association and associated modality order
        """
        # values to groupby indices with
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

        # testing viability of combination
        best_association = self._test_viability(feature, order, associations_xagg, xagg_dev, dropna)

        # applying best_combination to order and xtab
        if best_association is not None:
            order = order_apply_combination(order, best_association["combination"])

        return best_association, order

    def _test_viability(
        self,
        feature: str,
        order: GroupedList,
        associations_xagg: list[dict[str, Any]],
        xagg_dev: Union[Series, DataFrame],
        dropna: bool,
    ) -> dict[str, Any]:
        """Tests the viability of all possible combinations onto xagg_dev

        Parameters
        ----------
        feature : str
            _description_
        order : GroupedList
            order of the feature
        associations_xagg : list[dict[str, Any]]
            _description_
        xagg_dev : Union[Series, DataFrame]
            _description_

        Returns
        -------
        dict[str, Any]
            _description_
        """
        # testing viability of all combinations
        best_association, train_viable, dev_viable = (None,) * 3
        test_results: dict[str, bool] = {}
        for n_combination, association in tqdm(
            enumerate(associations_xagg),
            total=len(associations_xagg),
            disable=not self.verbose,
            desc="Testing robustness    ",
        ):
            # computing target rate and frequency per value
            train_rates = self._printer(association["xagg"])

            # viability on train sample:
            # - target rates are distinct for consecutive modalities
            distinct_rates_train = not any(
                isclose(train_rates["target_rate"][1:], train_rates["target_rate"].shift(1)[1:])
            )
            # - minimum frequency is reached for all modalities
            min_freq_train = all(train_rates["frequency"] >= self.min_freq_mod)

            # checking for viability on train
            train_viable = min_freq_train and distinct_rates_train
            test_results.update(
                {
                    "train_viable": train_viable,
                    "min_freq_train": min_freq_train,
                    "distinct_rates_train": distinct_rates_train,
                }
            )
            if train_viable:
                # case 0: no test sample provided -> not testing for robustness
                if xagg_dev is None:
                    best_association = association  # found best viable combination

                # case 1: test sample provided -> testing robustness
                else:
                    # grouping the dev sample per modality
                    grouped_xagg_dev = self._grouper(xagg_dev, association["index_to_groupby"])

                    # computing target rate and frequency per modality
                    dev_rates = self._printer(grouped_xagg_dev)

                    # viability on dev sample:
                    # - grouped values have the same ranks in train/test
                    ranks_train_dev = all(
                        train_rates.sort_values("target_rate").index
                        == dev_rates.sort_values("target_rate").index
                    )
                    # - minimum frequency is reached for all modalities
                    min_freq_dev = all(dev_rates["frequency"] >= self.min_freq_mod)
                    # - target rates are distinct for all modalities
                    distinct_rates_dev = not any(
                        isclose(dev_rates["target_rate"][1:], dev_rates["target_rate"].shift(1)[1:])
                    )

                    # checking for viability on dev
                    dev_viable = ranks_train_dev and min_freq_dev and distinct_rates_dev
                    test_results.update(
                        {
                            "dev_viable": dev_viable,
                            "min_freq_dev": min_freq_dev,
                            "ranks_train_dev": ranks_train_dev,
                            "distinct_rates_dev": distinct_rates_dev,
                        }
                    )
                    if dev_viable:
                        best_association = association  # found best viable combination

            # historizing combinations and tests
            self._historize_viability_test(
                feature=feature,
                association=association,
                order=order,
                n_combination=n_combination,
                associations_xagg=associations_xagg,
                dropna=dropna,
                **test_results,
            )

            # best combination found: breaking the loop on combinations
            if best_association is not None:
                break

        return best_association

    def _historize_viability_test(
        self,
        feature: str,
        association: dict[Any],
        order: GroupedList,
        n_combination: int = None,
        associations_xagg: list[dict[str, Any]] = None,
        train_viable: bool = None,
        dev_viable: bool = None,
        dropna: bool = False,
        **viability_msg_params,
    ) -> None:
        """historizes the viability tests results for specified feature

        Parameters
        ----------
        feature : str
            feature for which to historize the combination
        viability : bool
            result of viability test
        order : GroupedList
            order of the modalities and there respective groups
        association : dict[Any]
            index_to_groupby and self.sort_by values
        viability_msg_params : dict
            kwargs to determine the viability message
        """
        # Messages associated to each failed viability test
        messages = []
        if not viability_msg_params.get("ranks_train_dev", True):
            messages += ["X_dev: inversion of target rates per modality"]
        if not viability_msg_params.get("min_freq_dev", True):
            messages += [
                f"X_dev: non-representative modality (min_freq_mod={self.min_freq_mod:2.2%})"
            ]
        if not viability_msg_params.get("distinct_rates_dev", True):
            messages += ["X_dev: non-distinct target rates per consecutive modalities"]
        if not viability_msg_params.get("min_freq_train", True):
            messages += [f"X: non-representative modality (min_freq_mod={self.min_freq_mod:2.2%})"]
        if not viability_msg_params.get("distinct_rates_train", True):
            messages += ["X: non-distinct target rates per consecutive modalities"]

        # viability has been checked on train
        viability = None
        if train_viable is not None:
            # viability on train
            viability = train_viable
            if train_viable:
                # viability has been checked on dev
                if dev_viable is not None:
                    # viability on dev
                    viability = dev_viable
                    if dev_viable:
                        messages = ["Combination robust between X and X_dev"]
                else:  # no x_dev provided
                    messages = ["Combination viable on X"]
        else:
            messages = ["Raw X distribution"]

        # viability not checked for following less associated combinations
        associations_not_checked = []
        if viability:
            associations_not_checked = associations_xagg[n_combination + 1 :]

        # storing combination and adding not tested combinations to the set to be historized
        associations_to_historize = [association] + associations_not_checked
        messages_to_historize = [messages] + [["Not checked"]] * len(associations_not_checked)
        viability_to_historize = [viability] + [None] * len(associations_not_checked)

        # historizing all combinations
        for asso, msg, viab in zip(
            associations_to_historize, messages_to_historize, viability_to_historize
        ):
            # Formats a combination for historization
            combi = asso["index_to_groupby"]
            formatted_combi = [
                [
                    value
                    for modality in combi.keys()
                    for group_modality in order.get(modality, modality)
                    for value in self.values_orders[feature].get(group_modality, group_modality)
                    if combi[modality] == final_group
                ]
                for final_group in Series(combi.values()).unique()
            ]

            # historizing test results
            self._history[feature] += [
                {
                    "combination": formatted_combi,
                    self.sort_by: asso[self.sort_by],
                    "viability": viab,
                    "viability_message": msg,
                    "grouping_nan": dropna,
                }
            ]

    def history(self, feature: str = None) -> DataFrame:
        """Historic of tested combinations and there association with the target.

        By default:

            * Modality ``str_default="__OTHER__"`` is generated for features that contain non-representative modalities.
            * Modality ``str_nan="__NAN__"`` is generated for features that contain ``numpy.nan``.
            * Whatever the value of ``dropna``, the association is computed for non-missing values.

        Parameters
        ----------
        feature : str, optional
            Specify for which feature to return the history, by default ``None``

        Returns
        -------
        DataFrame
            Historic of features' tested combinations.
        """
        # getting feature's history
        if feature is not None:
            assert (
                feature in self._history.keys()
            ), f"Carving of feature {feature} was not requested."
            histo = self._history[feature]

        # getting all features' history
        else:
            histo = []
            for feature in self._history.keys():
                feature_histories = self._history[feature]
                for feature_history in feature_histories:
                    feature_history.update({"feature": feature})
                histo += feature_histories

        # formatting combinations
        # history["combination"] = history["combination"].apply(format_for_history)

        return DataFrame(histo)

    def _index_mapper(
        self, feature: str, labels_orders: dict[str, GroupedList], xtab: DataFrame = None
    ) -> DataFrame:
        """Prints a binary xtab's statistics

        Parameters
        ----------
        order_get : Callable
            Ordering of modalities used to map indices
        xtab : Dataframe
            A crosstab, by default None

        Returns
        -------
        DataFrame
            Target rate and frequency per modality
        """
        # checking for an xtab
        mapped_xtab = None
        if xtab is not None:
            # copying initial xtab
            mapped_xtab = xtab.copy()

            # for qualitative features -> mapping with values_orders.content
            if feature in self.qualitative_features:
                mapped_index = [
                    self.values_orders[feature].get(idx, idx) for idx in mapped_xtab.index
                ]
                # removing str_default
                mapped_index = [
                    [str(idx) for idx in mapped_idx if idx != self.str_default]
                    for mapped_idx in mapped_index
                ]
                mapped_index = [
                    mapped_idx[-1] + " to " + mapped_idx[0]
                    if len(mapped_idx) > 2
                    else mapped_idx[0]
                    if len(mapped_idx) == 0
                    else ", ".join(mapped_idx)
                    for mapped_idx in mapped_index
                ]
            # for quantitative features -> mapping with labels_orders.keys
            else:
                mapped_index = labels_orders[feature][:]

            # modifying indices based on provided order
            mapped_xtab.index = mapped_index

        return mapped_xtab

    def _print_xagg(
        self,
        xagg: DataFrame,
        message: str,
        *,
        xagg_dev: DataFrame = None,
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
        if self.verbose:  # verbose if requested
            print(f"\n - [AutoCarver] {message}")

            # getting pretty xtabs
            nice_xagg = self._printer(xagg)
            nice_xagg_dev = self._printer(xagg_dev)

            # case 0: no pretty hmtl printing
            if not self.pretty_print:
                print(nice_xagg, "\n")
                if xagg_dev is not None:
                    print("X_dev distribution\n", nice_xagg_dev, "\n")

            # case 1: pretty html printing
            else:
                # getting prettier xtabs
                nicer_xagg = prettier_xagg(nice_xagg, caption="X distribution")
                nicer_xagg_dev = prettier_xagg(
                    nice_xagg_dev, caption="X_dev distribution", hide_index=True
                )

                # merging outputs
                nicer_xaggs = nicer_xagg + "          " + nicer_xagg_dev

                # displaying html of colored DataFrame
                display_html(nicer_xaggs, raw=True)


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


def nan_combinations(
    raw_order: GroupedList,
    str_nan: str,
    max_n_mod: int,
    min_group_size: int,
) -> list[list[str]]:
    """All consecutive combinatios of non-nans with added nan to each possible group and a last
      group only with nan if the max_n_mod is not reached by the combination

    Parameters
    ----------
    raw_order : GroupedList
        _description_
    str_nan : str
        _description_
    max_n_mod : int
        _description_
    min_group_size : int
        _description_

    Returns
    -------
    list[list[str]]
        _description_
    """
    # all possible consecutive combinations
    combinations = consecutive_combinations(raw_order, max_n_mod, min_group_size=1)
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
    return load_discretizer(auto_carver_json)


def prettier_xagg(
    nice_xagg: DataFrame = None,
    caption: str = None,
    hide_index: bool = False,
) -> str:
    """Converts a crosstab to the HTML format, adding nice colors

    Parameters
    ----------
    nice_xagg : DataFrame, optional
        Target rate and frequency per modality, by default None
    caption : str, optional
        Title of the HTML table, by default None
    hide_index : bool, optional
        Whether or not to hide the index (for dev distribution)

    Returns
    -------
    str
        HTML format of the crosstab
    """
    """Pretty display of frequency and target rate per modality on the same line."""
    # checking for a provided xtab
    nicer_xagg = ""
    if nice_xagg is not None:
        # checking for non unique indices
        if any(nice_xagg.index.duplicated()):
            nice_xagg.reset_index(inplace=True)

        # adding coolwarm color gradient
        nicer_xagg = nice_xagg.style.background_gradient(cmap="coolwarm")

        # printing inline notebook
        nicer_xagg = nicer_xagg.set_table_attributes("style='display:inline'")

        # lower precision
        nicer_xagg = nicer_xagg.format(precision=4)

        # adding custom caption/title
        if caption is not None:
            nicer_xagg = nicer_xagg.set_caption(caption)

        # hiding index for dev
        if hide_index:
            nicer_xagg.hide(axis="index")

        # converting to html
        nicer_xagg = nicer_xagg._repr_html_()

    return nicer_xagg
