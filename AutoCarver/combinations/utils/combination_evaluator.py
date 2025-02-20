"""CombinationEvaluator class to evaluate
the best combination of modalities for a feature."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Union

from pandas import DataFrame, Series
from tqdm import tqdm

from ...features import BaseFeature, GroupedList
from ...utils import get_attribute, get_bool_attribute
from .combinations import (
    consecutive_combinations,
    format_combinations,
    nan_combinations,
    order_apply_combination,
    xagg_apply_combination,
)
from .target_rate import TargetRate
from .testing import TestKeys, is_viable, test_viability


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

    train: AggregatedSample = field(default_factory=lambda: AggregatedSample(None))
    dev: AggregatedSample = field(default_factory=lambda: AggregatedSample(None))

    def set(self, train: DataFrame, dev: DataFrame = None) -> None:
        """Sets the train and dev samples"""

        # setting train and dev samples
        self.train = AggregatedSample(train)
        self.dev = AggregatedSample(dev)

    def dropna(self, feature_nan: str) -> None:
        """Removes nans from the samples"""

        # removing nans from crosstabs
        self.dev.xagg = filter_nan(self.dev.raw, feature_nan)
        self.train.xagg = filter_nan(self.train.raw, feature_nan)

    def set_indices_to_values(self, feature: BaseFeature) -> None:
        """Resets the index of the samples"""

        # renaming index to feature values
        if self.train.raw is not None:
            self.train.raw.rename(index=feature.value_per_label, inplace=True)
        if self.dev.raw is not None:
            self.dev.raw.rename(index=feature.value_per_label, inplace=True)

    def __repr__(self) -> str:
        return f"AggregatedSamples(train={self.train.shape}, dev={self.dev.shape})"

    def apply_combination(self, feature: BaseFeature) -> None:
        """Applies best combination to xagg"""

        # applying best_combination to xaggs
        self.train.raw = xagg_apply_combination(self.train.raw, feature)
        self.dev.raw = xagg_apply_combination(self.dev.raw, feature)


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
        """
        Parameters
        ----------
        max_n_mod : int, optional
            Maximum number of modalities per feature, by default ``5``

            * The combination with the best association will be selected.
            * All combinations of sizes from 1 to :attr:`max_n_mod` are tested out.

            .. tip::
                Set between ``3`` (faster, more robust) and ``7`` (slower, less robust)

        Keyword Arguments
        -----------------
        min_freq : float, optional
            Minimum frequency per modality per feature, by default ``None``

            * Features need at least one modality more frequent than :attr:`min_freq`
            * Defines number of quantiles of continuous features
            * Minimum frequency of modality of quantitative features

            .. tip::
                Set between ``0.01`` (slower, less robust) and ``0.2`` (faster, more robust)

        dropna : bool, optional
            * ``True``, try to group ``nan`` with other modalities.
            * ``False``, ``nan`` are ignored (not grouped), by default ``False``

        verbose : bool, optional
            * ``True``, without ``IPython``: prints raw statitics
            * ``True``, with ``IPython``: prints HTML statistics, by default ``False``
        """

        # setting attributes
        self.verbose = get_bool_attribute(kwargs, "verbose", False)
        self.dropna = get_bool_attribute(kwargs, "verbose", False)
        self.max_n_mod = max_n_mod
        self.min_freq = get_attribute(kwargs, "min_freq")

        # attributes to be set by get_best_combination
        self.feature: BaseFeature = None
        self.samples: AggregatedSamples = AggregatedSamples()
        self._statistics_cache = None
        self._init_target_rate(get_attribute(kwargs, "target_rate"))

    @abstractmethod
    def _init_target_rate(self, target_rate: TargetRate) -> None:
        """Initializes target rate."""
        self.target_rate = target_rate

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
                tqdm(indices_to_groupby, desc="Grouping modalities   ", disable=not self.verbose),
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
            # resetting samples' indices to values
            self.samples.set_indices_to_values(self.feature)

            # applying best_combination to feature labels
            labels = order_apply_combination(self.feature.labels, best_association["combination"])

            # updating feature's values according to combination
            self.feature.update(labels, convert_labels=True)

            # applying best_combination to raw xagg and xagg_dev
            self.samples.apply_combination(self.feature)

            # udpating statistics
            self.feature.statistics = self.target_rate.compute(self.samples.train.raw)

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
            self.samples.dropna(self.feature.nan)

        # checking for non-nan values
        if self.samples.train.shape[0] > 1:
            # historizing raw combination
            self._historize_raw_combination()

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

        # setting dropna to user-requested value
        self.feature = feature
        self.feature.dropna = self.dropna

        # setting samples
        self.samples.set(train=xagg, dev=xagg_dev)

        # getting best combination without NaNs
        best_combination = self._get_best_combination_non_nan()

        # grouping NaNs if requested to drop them (dropna=True)
        return self._get_best_combination_with_nan(best_combination)

    def _test_viability_train(self, combination: dict) -> dict:
        """testing the viability of the combination on xagg_train"""

        # computing target rate and frequency per value
        train_rates = self.target_rate.compute(combination["xagg"])

        # viability on train sample:
        result = test_viability(train_rates, self.min_freq, self.target_rate.__name__)

        return result

    def _test_viability_dev(self, test_results: dict, combination: dict) -> dict:
        """testing the viability of the combination on xagg_dev"""

        # case 0: not viable on train or no test sample -> not testing for robustness
        if not test_results[TestKeys.VIABLE.value] or self.samples.dev.xagg is None:
            return {**test_results, "dev": {TestKeys.VIABLE.value: None}}

        # case 1: test sample provided and viable on train -> testing robustness
        # getting train target rates
        train_target_rate = test_results["train_rates"][self.target_rate.__name__]

        # grouping the dev sample per modality
        grouped_xagg_dev = self._grouper(self.samples.dev, combination["index_to_groupby"])

        # computing target rate and frequency per modality
        dev_rates = self.target_rate.compute(grouped_xagg_dev)

        # viability on dev sample:
        dev_results = test_viability(
            dev_rates, self.min_freq, self.target_rate.__name__, train_target_rate
        )
        test_results = {**test_results, **dev_results}

        # checking for viability on both samples
        test_results[TestKeys.VIABLE.value] = is_viable(test_results)

        return test_results

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
            self._historize_combination(combination, test_results)

            # best combination found: breaking the loop on combinations
            if test_results[TestKeys.VIABLE.value]:
                viable_combination = combination

                # historizing remaining combinations/not tested
                self._historize_remaining_combinations(associations, n_combination)
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

    def _historize_remaining_combinations(
        self, associations: list[dict], n_combination: int
    ) -> None:
        """historizes the remaining combinations that have not been tested"""

        # historizing all remaining combinations
        for combination in associations[n_combination + 1 :]:
            # historizing not tested combination
            self.feature.history = [
                {**clean_combination(combination, self.feature), "info": "Not checked"}
            ]

    def _historize_combination(self, combination: dict, test_results: dict) -> None:
        """historizes the test results of the combination"""

        # keeping only relevant information
        test_results.update(clean_combination(combination, self.feature))

        # checking for viability
        if test_results[TestKeys.VIABLE.value]:
            # setting feature's statistics (selected combination)
            # self.feature.statistics = test_results.pop("train_rates")
            self.feature.statistics = test_results.pop("train_rates")

            # adding info
            info = f"Best for {self.sort_by} and max_n_mod={self.max_n_mod}"

            # adding dropna info if its the case
            if test_results.get("dropna"):
                info += " (dropna=True)"

        # not viable on train or dev
        else:
            info = "Not viable"

            # removing train rates if still in there
            test_results.pop("train_rates", None)

        # saving info
        test_results.update({"info": info})

        # historizing test results
        self.feature.history = [test_results]

    def _historize_raw_combination(self):
        """historizes the raw combination"""

        # setting feature's statistics
        self.feature.statistics = self.target_rate.compute(self.samples.train.raw)

        # computing association of sample
        raw_association = self._association_measure(
            self.samples.train.raw, n_obs=sum(self.samples.train.raw.apply(sum))
        )

        # computing number of modalities
        n_mod = self.samples.train.raw.shape[0]

        # creating info message
        info = "Raw distribution"

        # adding info if n_mod > max_n_mod
        if n_mod > self.max_n_mod:
            info += f" (n_mod={n_mod}>max_n_mod={self.max_n_mod})"

        # historizing raw combination
        combination = {
            "info": info,
            **raw_association,
            "combination": {modality: modality for modality in self.samples.train.raw.index},
        }

        # historizing within feature
        self.feature.history = [{**combination, "n_mod": n_mod, "dropna": False}]

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
        """Saves :class:`CombinationEvaluator` to .json file.

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
        """Allows one to load a :class:`CombinationEvaluator` saved as a .json file.

        Parameters
        ----------
        file : str | dict
            String of .json file name or content of the file.

        Returns
        -------
        CombinationEvaluator
            A ready-to-use :class:`CombinationEvaluator`
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


def clean_combination(
    combination: dict, feature: BaseFeature, remove_train_rates: bool = False
) -> dict:
    """Cleans a combination to remove unwanted keys"""

    # removing unwanted keys
    index_to_groupby = combination.pop("index_to_groupby")

    # checking if dropna was used
    dropna = feature.nan in index_to_groupby

    # computing number of modalities
    n_mod = len(combination["combination"])

    # listing unwanted keys
    unwanted_keys = ["xagg", "combination"]
    if not remove_train_rates:
        unwanted_keys.append("train_rates")

    # filtering unwanted keys
    filtered = {k: v for k, v in combination.items() if k not in unwanted_keys}

    return {**filtered, "n_mod": n_mod, "dropna": dropna, "combination": index_to_groupby}
