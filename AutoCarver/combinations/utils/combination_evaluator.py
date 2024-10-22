"""CombinationEvaluator class to evaluate
the best combination of modalities for a feature."""

import json
from abc import ABC, abstractmethod
from typing import Union
from dataclasses import dataclass

from pandas import DataFrame, Series
from tqdm.autonotebook import tqdm
from ...utils import get_attribute, get_bool_attribute

from ...features import BaseFeature, GroupedList
from .combinations import (
    consecutive_combinations,
    format_combinations,
    nan_combinations,
    order_apply_combination,
    xagg_apply_combination,
)
from .testing import _test_viability, is_viable


@dataclass
class AggregatedSample:
    """Sample class to store aggregated samples

    Attributes
    ----------
    xagg : DataFrame
        Aggregated sample
    raw : DataFrame
        Raw aggregated sample
    """

    xagg: Union[DataFrame, Series]
    _raw: Union[DataFrame, Series] = None

    def __post_init__(self):
        """Post initialization"""
        # setting xtab_dev to xtab if not provided
        if self._raw is None and self.xagg is not None:
            self._raw = self.xagg.copy()

    @property
    def raw(self) -> DataFrame:
        """Returns the raw value of the xagg"""
        return self._raw

    @raw.setter
    def raw(self, value: Union[DataFrame, Series]) -> None:
        """Sets the raw value of the xagg"""

        # setting raw value
        self._raw = value

        # copying xagg
        if value is not None:
            self.xagg = value.copy()

    @property
    def shape(self) -> tuple[int, int]:
        """Returns the shape of the xagg"""
        return self.xagg.shape

    @property
    def index(self) -> list[str]:
        """Returns the index of the xagg"""
        return self.xagg.index

    @property
    def columns(self) -> list[str]:
        """Returns the columns of the xagg"""
        return self.xagg.columns

    @property
    def values(self) -> DataFrame:
        """Returns the values of the xagg"""
        return self.xagg.values

    def groupby(self, *args, **kwargs) -> DataFrame:
        """Groups the xagg by the specified indices"""
        return self.xagg.groupby(*args, **kwargs)


@dataclass
class AggregatedSamples:
    """stores train and dev samples"""

    train: AggregatedSample = AggregatedSample(None)
    dev: AggregatedSample = AggregatedSample(None)


class CombinationEvaluator(ABC):
    """CombinationEvaluator class to evaluate
    the best combination of modalities for a feature."""

    __name__ = "CombinationEvaluator"

    is_y_binary = False
    is_y_continuous = False
    sort_by = None

    def __init__(
        self,
        max_n_mod: int = 5,
        **kwargs,
    ) -> None:
        self.verbose = get_bool_attribute(kwargs, "verbose", False)
        self.dropna = get_bool_attribute(kwargs, "verbose", False)
        self.max_n_mod = max_n_mod
        self.min_freq = get_attribute(kwargs, "min_freq")

        # attributes to be set by get_best_combination
        self.feature: BaseFeature = None
        self.samples: AggregatedSamples = AggregatedSamples()

    def __repr__(self) -> str:
        return f"{self.__name__}(max_n_mod={self.max_n_mod})"

    def _group_xagg_by_combinations(self, combinations: list[list]) -> list[dict]:
        """groups xagg by combinations of indices"""

        # values to groupby indices with
        indices_to_groupby = format_combinations(combinations)

        # grouping tab by its indices
        return [
            {
                "xagg": self._grouper(self.samples.train, index_to_groupby),
                "combination": combination,
                "index_to_groupby": index_to_groupby,
            }
            for combination, index_to_groupby in zip(
                combinations,
                tqdm(indices_to_groupby, desc="Grouping modalities", disable=not self.verbose),
            )
        ]

    def _compute_associations(self, grouped_xaggs: list[dict]) -> list[dict]:
        """computes associations for each grouped xagg"""

        # number of observations (only used for crosstabs)
        n_obs = self.samples.train.xagg.apply(sum).sum()

        # computing associations for each crosstab
        associations = [
            {**grouped_xagg, **self._association_measure(grouped_xagg["xagg"], n_obs=n_obs)}
            for grouped_xagg in tqdm(
                grouped_xaggs, desc="Computing associations", disable=not self.verbose
            )
        ]

        # sorting associations according to specified metric
        return (
            DataFrame(associations)
            .sort_values(self.sort_by, ascending=False)
            .to_dict(orient="records")
        )

    def _get_best_association(self, combinations: list[list[str]]) -> dict:
        """Computes associations of the tab for each combination

        Returns
        -------
        tuple[dict[str, Any], GroupedList]
            best viable association and associated modality order
        """
        # grouping tab by its indices
        grouped_xaggs = self._group_xagg_by_combinations(combinations)

        # computing associations for each tabs
        associations = self._compute_associations(grouped_xaggs)

        # testing viability of combination
        best_combination = self._get_viable_combination(associations)

        # applying best combination to feature labels and xtab
        self._apply_best_combination(best_combination)

        return best_combination

    def _apply_best_combination(self, best_association: dict) -> None:
        """Applies best combination to feature labels and xtab"""

        # checking that a combination was found
        if best_association is not None:
            # applying best_combination to feature labels
            labels = order_apply_combination(self.feature.labels, best_association["combination"])

            # updating feature's values and xagg indices accordingly
            self.feature.update(labels, convert_labels=True)

            # applying best_combination to raw xagg and xagg_dev
            self.samples.train.raw = xagg_apply_combination(
                self.samples.train.raw, labels, self.feature
            )
            self.samples.dev.raw = xagg_apply_combination(
                self.samples.dev.raw, labels, self.feature
            )

    def _get_best_combination_non_nan(self) -> dict:
        """Computes associations of the tab for each combination of non-nans

        - dropna has to be set to True
        """

        # raw ordering without nans
        raw_labels = GroupedList(self.feature.labels[:])

        # removing nans if any
        if self.feature.has_nan:
            # removing nans for combination of non-nans
            if self.feature.dropna:
                raw_labels.remove(self.feature.nan)

            # removing nans from crosstabs
            self.samples.dev.xagg = filter_nan(self.samples.dev.raw, self.feature.nan)
            self.samples.train.xagg = filter_nan(self.samples.train.raw, self.feature.nan)

        # checking for non-nan values
        if self.samples.train.shape[0] > 1:
            # all possible consecutive combinations
            combinations = consecutive_combinations(raw_labels, self.max_n_mod)

            # getting most associated combination
            return self._get_best_association(combinations)

        return None

    def _get_best_combination_with_nan(self, best_combination: dict) -> dict:
        """Computes associations of the tab for each combination with nans"""

        # grouping NaNs if requested to drop them (dropna=True)
        if self.dropna and self.feature.has_nan and best_combination is not None:
            # verbose if requested
            if self.verbose:
                print(f"[{self.__name__}] Grouping NaNs")

            # adding combinations with NaNs
            combinations = nan_combinations(self.feature, self.max_n_mod)

            # getting most associated combination
            best_combination = self._get_best_association(combinations)

        return best_combination

    def get_best_combination(
        self, feature: BaseFeature, xagg: DataFrame, xagg_dev: DataFrame = None
    ) -> dict:
        """Computes best combination of modalities for the feature"""

        # checking for min_freq
        if self.min_freq is None:
            raise ValueError("min_freq has to be set before calling get_best_combination")

        # setting feature and xtab
        self.feature = feature
        self.samples.train = AggregatedSample(xagg)
        self.samples.dev = AggregatedSample(xagg_dev)

        # setting dropna to user-requested value
        self.feature.dropna = self.dropna

        # historizing raw combination
        self._historize_raw_combination()

        # getting best combination without NaNs
        best_combination = self._get_best_combination_non_nan()

        # grouping NaNs if requested to drop them (dropna=True)
        return self._get_best_combination_with_nan(best_combination)

    def _test_viability_train(self, combination: dict) -> dict:
        """testing the viability of the combination on xagg_train"""

        # computing target rate and frequency per value
        train_rates = self._compute_target_rates(combination["xagg"])

        # viability on train sample:
        return _test_viability(train_rates, self.min_freq)

    def _test_viability_dev(self, test_results: dict, combination: dict) -> dict:
        """testing the viability of the combination on xagg_dev"""

        # case 0: not viable on train or no test sample -> not testing for robustness
        if not test_results.get("train").get("viable") or self.samples.dev.xagg is None:
            return {**test_results, "dev": {"viable": None}}
        # case 1: test sample provided -> testing robustness

        # getting train target rates
        train_target_rate = test_results.pop("train_rates")["target_rate"]

        # grouping the dev sample per modality
        grouped_xagg_dev = self._grouper(self.samples.dev, combination["index_to_groupby"])

        # computing target rate and frequency per modality
        dev_rates = self._compute_target_rates(grouped_xagg_dev)

        # viability on dev sample:
        return {**test_results, **_test_viability(dev_rates, self.min_freq, train_target_rate)}

    def _get_viable_combination(self, associations: list[dict]) -> dict:
        """Tests the viability of all possible combinations onto xagg_dev"""

        # testing viability of all combinations
        viable_combination = None
        for n_combination, combination in tqdm(
            enumerate(associations),
            total=len(associations),
            desc="Testing robustness    ",
            disable=not self.verbose,
        ):
            # testing combination viability on train sample
            test_results = self._test_viability_train(combination)

            # testing combination viability on dev sample
            test_results = self._test_viability_dev(test_results, combination)

            # historizing combinations and tests
            self._historize()

            # best combination found: breaking the loop on combinations
            if is_viable(test_results):
                viable_combination = combination
                break

        if self.verbose:  # verbose if requested
            print("\n")

        return viable_combination

    @abstractmethod
    def _grouper(self, xagg: AggregatedSample, groupby: dict[str, str]) -> Union[Series, DataFrame]:
        """Helper to group XAGG's values by groupby (carver specific)"""

    @abstractmethod
    def _association_measure(
        self, xagg: AggregatedSample, n_obs: int = None, tol: float = 1e-10
    ) -> dict[str, float]:
        """Helper to measure association between X and y (carver specific)"""

    @abstractmethod
    def _compute_target_rates(self, xagg: Union[Series, DataFrame]) -> DataFrame:
        """helper to print an XAGG (carver specific)"""

    def _historize_raw_combination(self):
        # historizing raw combination TODO
        # raw_association = {
        #     "index_to_groupby": {modality: modality for modality in self.samples.train.xagg.index},
        #     self.sort_by: self._association_measure(
        #         self.samples.train.xagg.dropna(),
        #         n_obs=sum(self.samples.train.xagg.dropna().apply(sum)),
        #     )[self.sort_by],
        # }
        # self._historize(self.feature, raw_association, self.feature.labels)
        pass

    def to_json(self) -> dict:
        """Converts to JSON format.

        To be used with ``json.dump``.

        Returns
        -------
        dict
            JSON serialized object
        """
        return {
            "sort_by": self.sort_by,
            "max_n_mod": self.max_n_mod,
            "dropna": self.dropna,
            "min_freq": self.min_freq,
            "verbose": self.verbose,
        }

    def save(self, file_name: str) -> None:
        """Saves pipeline to .json file.

        Parameters
        ----------
        file_name : str
            String of .json file name
        """
        # checking for input
        if file_name.endswith(".json"):
            with open(file_name, "w", encoding="utf-8") as json_file:
                json.dump(self.to_json(), json_file)
        # raising for non-json file name
        else:
            raise ValueError(f"[{self.__name__}] Provide a file_name that ends with .json.")

    @classmethod
    def load(cls, file: Union[str, dict]) -> "CombinationEvaluator":
        """Allows one to load a CombinationEvaluator saved as a .json file.

        The CombinationEvaluator has to be saved with ``CombinationEvaluator.save()``, otherwise
        there can be no guarantee for it to be restored.

        Parameters
        ----------
        file_name : str | dict
            String of saved CombinationEvaluator's .json file name or content of the file.

        Returns
        -------
        CombinationEvaluator
            A ready-to-use CombinationEvaluator
        """
        # reading file
        if isinstance(file, str):
            with open(file, "r", encoding="utf-8") as json_file:
                combinations_json = json.load(json_file)
        elif isinstance(file, dict):
            combinations_json = file
        else:
            raise ValueError(f"[{cls.__name__}] Provide a file_name or a dict.")

        # checking for sort_by
        if combinations_json.get("sort_by") is None:
            if cls.sort_by is not None:
                raise ValueError(f"[{cls.__name__}] sort_by has to be {cls.sort_by}")
        elif combinations_json.get("sort_by") != cls.sort_by:
            raise ValueError(f"[{cls.__name__}] sort_by has to be {cls.sort_by}")

        # initiating BaseDiscretizer
        return cls(**combinations_json)

    def _historize(self, *args, **kwargs) -> None:
        pass

    # def _historize(
    #     self,
    #     feature: BaseFeature,
    #     association: dict[Any],
    #     n_combination: int = None,
    #     associations_xagg: list[dict[str, Any]] = None,
    #     train_viable: bool = None,
    #     dev_viable: bool = None,
    #     dropna: bool = False,
    #     **viability_msg_params,
    # ) -> None:
    #     """historizes the viability tests results for specified feature

    #     Parameters
    #     ----------
    #     feature : str
    #         feature for which to historize the combination
    #     viability : bool
    #         result of viability test
    #     order : GroupedList
    #         order of the modalities and there respective groups
    #     association : dict[Any]
    #         index_to_groupby and self.sort_by values
    #     viability_msg_params : dict
    #         kwargs to determine the viability message
    #     """
    #     # Messages associated to each failed viability test
    #     messages = []
    #     if not viability_msg_params.get("ranks_train_dev", True):
    #         messages += ["X_dev: inversion of target rates per modality"]
    #     if not viability_msg_params.get("min_freq_dev", True):
    #         messages += [f"X_dev: non-representative modality (min_freq={self.min_freq:2.2%})"]
    #     if not viability_msg_params.get("distinct_rates_dev", True):
    #         messages += ["X_dev: non-distinct target rates per consecutive modalities"]
    #     if not viability_msg_params.get("min_freq_train", True):
    #         messages += [f"X: non-representative modality (min_freq={self.min_freq:2.2%})"]
    #     if not viability_msg_params.get("distinct_rates_train", True):
    #         messages += ["X: non-distinct target rates per consecutive modalities"]

    #     # viability has been checked on train
    #     viability = None
    #     if train_viable is not None:
    #         # viability on train
    #         viability = train_viable
    #         if train_viable:
    #             # viability has been checked on dev
    #             if dev_viable is not None:
    #                 # viability on dev
    #                 viability = dev_viable
    #                 if dev_viable:
    #                     messages = ["Combination robust between X and X_dev"]
    #             else:  # no x_dev provided
    #                 messages = ["Combination viable on X"]
    #     else:
    #         messages = ["Raw X distribution"]

    #     # viability not checked for following less associated combinations
    #     associations_not_checked = []
    #     if viability:
    #         associations_not_checked = associations_xagg[n_combination + 1 :]

    #     # storing combination and adding not tested combinations to the set to be historized
    #     associations_to_historize = [association] + associations_not_checked
    #     messages_to_historize = [messages] + [["Not checked"]] * len(associations_not_checked)
    #     viability_to_historize = [viability] + [None] * len(associations_not_checked)

    #     # historizing test results: list comprehension for faster processing (large number of
    # combi)
    #     feature.history += [
    #         {
    #             # Formats a combination for historization
    #             "combination": [
    #                 [
    #                     value
    #                     for modality in asso["index_to_groupby"].keys()
    #                     for group_modality in feature.label_per_value.get(modality, modality)
    #                     for value in feature.content.get(group_modality, group_modality)
    #                     if asso["index_to_groupby"][modality] == final_group
    #                 ]
    #                 for final_group in Series(asso["index_to_groupby"].values()).unique()
    #             ],
    #             self.sort_by: asso[self.sort_by],
    #             "viability": viab,
    #             "viability_message": msg,
    #             "grouping_nan": dropna,
    #         }
    #         # historizing all combinations
    #         for asso, msg, viab in zip(
    #             associations_to_historize, messages_to_historize, viability_to_historize
    #         )
    #     ]


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
