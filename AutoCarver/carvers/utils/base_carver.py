"""Tool to build optimized buckets out of Quantitative and Qualitative features
for any task.
"""

from functools import partial
from typing import Any, Union
from warnings import warn

from numpy import isclose
from pandas import DataFrame, Series
from tqdm.autonotebook import tqdm

from .combinations import (
    nan_combinations,
    consecutive_combinations,
    xagg_apply_combination,
    order_apply_combination,
)
from .pretty_print import prettier_xagg, index_mapper

from ...discretizers.discretizers import Discretizer
from ...discretizers.utils.base_discretizers import BaseDiscretizer, load_discretizer
from ...discretizers.utils.serialization import json_serialize_history
from ...features import GroupedList, Features, BaseFeature

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

    First fits a :class:`Discretizer`. Raw data should be provided as input (not a result of
    ``Discretizer.transform()``).
    """

    __name__ = "AutoCarver"

    def __init__(
        self,
        sort_by: str,
        min_freq: float,
        features: Features,
        *,
        max_n_mod: int = 5,
        min_freq_mod: float = None,
        output_dtype: str = "float",
        dropna: bool = True,
        # copy: bool = False,
        # verbose: bool = False,
        # n_jobs: int = 1,
        **kwargs: dict,
    ) -> None:
        """
        min_freq : float
            Minimum frequency per grouped modalities.

            * Features whose most frequent modality is less than ``min_freq`` will not be carved.
            * Sets the number of quantiles in which to discretize the continuous features.
            * Sets the minimum frequency of a quantitative feature's modality.

            **Tip**: should be set between ``0.01`` (slower, preciser, less robust) and ``0.2``
            (faster, more robust)

        quantitative_features : list[str], optional
            List of column names of quantitative features (continuous and discrete) to be carved,
            by default ``None``

        qualitative_features : list[str], optional
            List of column names of qualitative features (non-ordinal) to be carved,
            by default ``None``

        ordinal_features : list[str], optional
            List of column names of ordinal features to be carved. For those features a list of
            values has to be provided in the ``values_orders`` dict, by default ``None``

        values_orders : dict[str, GroupedList], optional
            Dict of feature's column names and there associated ordering.
            If lists are passed, a :class:`GroupedList` will automatically be initiated,
            by default ``None``

        max_n_mod : int, optional
            Maximum number of modality per feature, by default ``5``

            All combinations of modalities for groups of modalities of sizes from 1 to
            ``max_n_mod`` will be tested.
            The combination with the best association will be selected.

            **Tip**: set between ``3`` (faster, more robust) and ``7`` (slower, less robust)

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
            If ``True``, feature processing at transform is applied to a copy of the provided
            DataFrame, by default ``False``

        verbose : bool, optional
            * ``True``, without IPython: prints raw steps for X, by default ``False``
            * ``True``, with IPython: adds HTML tables of target rates for X and X_dev

            **Tip**: IPython displaying can be turned off by setting ``pretty_print=False``

        n_jobs : int, optional
            Number of processes used by multiprocessing, by default ``1``

        **kwargs: dict
            Pass values for ``str_default`` and ``str_nan`` (default string values),
            as long as ``pretty_print`` to turn off IPython
        """

        # adding correct verbose to kwargs
        kwargs.update(
            {"verbose": bool(max(kwargs.get("verbose", False), kwargs.get("pretty_print", False)))}
        )

        # Initiating BaseDiscretizer
        super().__init__(features=features, output_dtype=output_dtype, dropna=dropna, **kwargs)

        # class specific attributes
        self.min_freq = min_freq  # minimum frequency per base bucket
        self.max_n_mod = max_n_mod  # maximum number of modality per feature
        if min_freq_mod is None:
            min_freq_mod = min_freq / 2
        self.min_freq_mod = min_freq_mod  # minimum frequency per final bucket
        self.sort_by = sort_by  # metric used to sort feature combinations

        # pretty printing if requested
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
        # progress bar if requested
        self.tqdm = partial(tqdm, disable=not self.verbose)

        # historizing everything
        self._history = {feature.name: [] for feature in self.features}

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
            Dataset used to discretize. Needs to have columns has specified in
            ``BaseCarver.features``.

        y : Series
            Binary target feature with wich the association is maximized.

        X_dev : DataFrame, optional
            Dataset to evalute the robustness of discretization, by default ``None``
            It should have the same distribution as X.

        y_dev : Series, optional
            Binary target feature with wich the robustness of discretization is evaluated,
            by default ``None``

        Returns
        -------
        tuple[DataFrame, DataFrame]
            Copies of (X, X_dev)
        """
        # checking for not provided y
        if y is None:
            raise ValueError(f" - [{self.__name__}] y must be provided, got {y}")

        # Checking for binary target and copying X
        x_copy = super()._prepare_data(X, y)
        x_dev_copy = super()._prepare_data(X_dev, y_dev)

        # discretizing all features, always copying, to keep discretization from start to finish
        discretizer = Discretizer(
            self.min_freq, self.features, **dict(self.kwargs, copy=True, output_dtype="str")
        )
        x_copy = discretizer.fit_transform(x_copy, y)
        if x_dev_copy is not None:  # applying on x_dev
            x_dev_copy = discretizer.transform(x_dev_copy, y_dev)

        # removing dropped features
        self.features.keep_features(discretizer.features.get_names())

        # setting up features to convert nans to feature.nan (drop nans)
        self.features.set_dropna(True)

        # filling up nans
        x_copy = self.features.fillna(x_copy)
        x_dev_copy = self.features.fillna(x_dev_copy)

        return x_copy, x_dev_copy

    def _combination_formatter(self, combination: list[list[str]]) -> dict[str, str]:
        """Attributes the first element of a group to all elements of a group"""
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
            if feature in self._history:
                self._history[feature] += [{"removed": True}]

    def fit(  # pylint: disable=W0222
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

        # computing crosstabs for each feature on train/test
        xaggs = self._aggregator(x_copy, y)
        xaggs_dev = self._aggregator(x_dev_copy, y_dev)

        # optimal butcketization/carving of each feature
        all_features = self.features[:]  # (features are being removed from self.features)
        for n, feature in enumerate(all_features):
            if self.verbose:  # verbose if requested
                print(f"\n------\n[{self.__name__}] Fit {feature} ({n+1}/{len(all_features)})\n---")

            # carving the feature
            self._carve_feature(feature, xaggs, xaggs_dev)

            if self.verbose:  # verbose if requested
                print("------\n")

            # TODO
            # self.features.set_dropna(False)

        # discretizing features based on each feature's values_order
        super().fit(X, y)

        return self

    def _aggregator(self, X: DataFrame, y: Series) -> Union[Series, DataFrame]:
        """Helper that aggregates X by y into crosstab or means (carver specific)"""
        _, _ = X, y  # unused attributes
        return True

    def _association_measure(self, xagg: DataFrame, n_obs: int) -> Union[Series, DataFrame]:
        """Helper to measure association between X and y (carver specific)"""
        _, _ = xagg, n_obs  # unused attributes
        return True

    def _grouper(self, xagg: DataFrame, groupby: list[str]) -> DataFrame:
        """Helper to group XAGG's values by groupby (carver specific)"""
        _, _ = xagg, groupby  # unused attributes
        return True

    def _printer(self, xagg: DataFrame = None) -> DataFrame:
        """helper to print an XAGG (carver specific)"""
        _ = xagg  # unused attribute
        return True

    def _carve_feature(
        self,
        feature: BaseFeature,
        xaggs: dict[str, Union[Series, DataFrame]],
        xaggs_dev: dict[str, Union[Series, DataFrame]],
    ) -> dict[str, GroupedList]:
        """Carves a feature into buckets that maximize association with the target"""
        # getting xtabs on train/test
        xagg = xaggs[feature.name]
        xagg_dev = xaggs_dev[feature.name]

        # historizing raw combination TODO
        raw_association = {
            "index_to_groupby": {modality: modality for modality in xagg.index},
            self.sort_by: self._association_measure(
                xagg.dropna(), n_obs=sum(xagg.dropna().apply(sum))
            )[self.sort_by],
        }
        self._historize_viability_test(feature, raw_association, feature.labels)

        # verbose if requested
        self._print_xagg(feature, xagg=xagg, xagg_dev=xagg_dev, message="Raw distribution")

        # getting best combination
        best_combination = self._get_best_combination(feature, xagg, xagg_dev=xagg_dev)

        # checking that a suitable combination has been found
        if best_combination is not None:
            xagg, xagg_dev = best_combination  # unpacking

            # verbose if requested
            self._print_xagg(feature, xagg=xagg, xagg_dev=xagg_dev, message="Carved distribution")

        # no suitable combination has been found -> removing feature
        else:
            warn(
                f"\n - [{self.__name__}] No robust combination for '{feature}' could be found"
                ".\nConsider increasing the size of dev sample or dropping the feature"
                " (train sample not representative of dev sample for this feature).",
                UserWarning,
            )
            self.features.remove(feature.name)

    def _get_best_combination(
        self,
        feature: BaseFeature,
        xagg: DataFrame,
        *,
        xagg_dev: DataFrame = None,
    ) -> tuple[GroupedList, DataFrame, DataFrame]:
        """ """
        # raw ordering
        raw_labels = GroupedList(feature.labels[:])
        if feature.has_nan:  # removing nans for combination of non-nans
            raw_labels.remove(feature.nan)

        # filtering out nans from train/test crosstabs
        raw_xagg = filter_nan(xagg, feature.nan)
        raw_xagg_dev = filter_nan(xagg_dev, feature.nan)

        # checking for non-nan values
        best_association = None
        if raw_xagg.shape[0] > 1:
            # all possible consecutive combinations
            combinations = consecutive_combinations(raw_labels, self.max_n_mod)

            # getting most associated combination
            best_association = self._get_best_association(
                feature, raw_xagg, combinations, xagg_dev=raw_xagg_dev, dropna=False
            )

            # grouping NaNs if requested to drop them (dropna=True)
            if self.dropna and feature.has_nan and best_association is not None:
                xagg, xagg_dev = best_association  # unpacking

                # raw ordering without nans
                raw_labels = GroupedList(feature.labels[:])
                raw_labels.remove(feature.nan)  # nans are added within nan_combinations

                # adding combinations with NaNs
                combinations = nan_combinations(raw_labels, feature.nan, self.max_n_mod)

                # getting most associated combination
                best_association = self._get_best_association(
                    feature, xagg, combinations, xagg_dev=xagg_dev, dropna=True
                )

        # checking that a suitable combination has been found
        if best_association is not None:
            xagg, xagg_dev = best_association  # unpacking

            return xagg, xagg_dev

    def _get_best_association(
        self,
        feature: BaseFeature,
        xagg: Union[Series, DataFrame],
        combinations: list[list[str]],
        *,
        xagg_dev: Union[Series, DataFrame] = None,
        dropna: bool = False,
    ) -> tuple[DataFrame, DataFrame]:
        """Computes associations of the tab for each combination

        Parameters
        ----------
        feature : str
            feature to measure association with
        xagg : Union[Series, DataFrame]
            _description_
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
            for index_to_groupby in self.tqdm(indices_to_groupby, desc="Grouping modalities   ")
        ]

        # computing associations for each tabs
        n_obs = xagg.apply(sum).sum()  # number of observations for xtabs
        associations_xagg = [
            self._association_measure(grouped_xagg, n_obs=n_obs)
            for grouped_xagg in self.tqdm(grouped_xaggs, desc="Computing associations")
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
        best_association = self._test_viability(feature, associations_xagg, xagg_dev, dropna)

        # applying best_combination to feature labels and xtab
        if best_association is not None:
            labels = order_apply_combination(feature.labels, best_association["combination"])

            # applying best_combination to xtabs
            xagg = xagg_apply_combination(xagg, labels)
            xagg_dev = xagg_apply_combination(xagg_dev, labels)

            # updating feature's values accordingly
            feature.update(labels, convert_labels=True)

            return xagg, xagg_dev

    def _test_viability(
        self,
        feature: BaseFeature,
        associations_xagg: list[dict[str, Any]],
        xagg_dev: Union[Series, DataFrame],
        dropna: bool,
    ) -> dict[str, Any]:
        """Tests the viability of all possible combinations onto xagg_dev"""

        # testing viability of all combinations
        best_association, train_viable, dev_viable = (None,) * 3
        test_results: dict[str, bool] = {}
        for n_combination, association in self.tqdm(
            enumerate(associations_xagg),
            total=len(associations_xagg),
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
                n_combination=n_combination,
                associations_xagg=associations_xagg,
                dropna=dropna,
                verbose=self.verbose,
                **test_results,
            )

            # best combination found: breaking the loop on combinations
            if best_association is not None:
                break

        return best_association

    def _historize_viability_test(
        self,
        feature: BaseFeature,
        association: dict[Any],
        n_combination: int = None,
        associations_xagg: list[dict[str, Any]] = None,
        train_viable: bool = None,
        dev_viable: bool = None,
        dropna: bool = False,
        verbose: bool = False,
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

        # historizing test results: list comprehension for faster processing (large number of combi)
        self._history[feature.name] += [
            {
                # Formats a combination for historization
                "combination": [
                    [
                        value
                        for modality in asso["index_to_groupby"].keys()
                        for group_modality in feature.labels.get(modality, modality)
                        for value in feature.values.get(group_modality, group_modality)
                        if asso["index_to_groupby"][modality] == final_group
                    ]
                    for final_group in Series(asso["index_to_groupby"].values()).unique()
                ],
                self.sort_by: asso[self.sort_by],
                "viability": viab,
                "viability_message": msg,
                "grouping_nan": dropna,
            }
            # historizing all combinations
            for asso, msg, viab in zip(
                associations_to_historize, messages_to_historize, viability_to_historize
            )
            # tqdm(total=len(associations_to_historize),
            # disable=verbose,
            # desc="Logging combinations  ",)
        ]

    def _print_xagg(
        self,
        feature: BaseFeature,
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
            print(f"\n - [{self.__name__}] {message}")

            # formatting XAGG
            formatted_xagg = index_mapper(feature, xagg)
            formatted_xagg_dev = index_mapper(feature, xagg_dev)

            # getting pretty xtabs
            nice_xagg = self._printer(formatted_xagg)
            nice_xagg_dev = self._printer(formatted_xagg_dev)

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

    def to_json(self) -> str:
        """Converts to .json format.

        To be used with ``json.dump``.

        Returns
        -------
        str
            JSON serialized object
        """
        # extracting content dictionnaries
        json_serialized_base_discretizer = super().to_json()

        # adding carver history and summary
        json_serialized_base_discretizer.update({"_history": json_serialize_history(self._history)})

        # dumping as json
        return json_serialized_base_discretizer


def filter_nan(xagg: Union[Series, DataFrame], str_nan: str) -> DataFrame:
    """Filters out nans from crosstab or y values"""
    # cehcking for values in crosstab
    filtered_xagg = None
    if xagg is not None:
        # filtering out nans if requested from train crosstab
        filtered_xagg = xagg.copy()
        if str_nan in xagg.index:
            filtered_xagg = xagg.drop(str_nan, axis=0)

    return filtered_xagg


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
    # removing _history if it exists
    _history = auto_carver_json.pop("_history", None)

    # loading discretizer
    loaded_discretizer = load_discretizer(auto_carver_json)

    # adding back _history
    loaded_discretizer._history = _history  # pylint: disable=W0212

    return loaded_discretizer
