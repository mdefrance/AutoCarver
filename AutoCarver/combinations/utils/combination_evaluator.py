"""CombinationEvaluator class to evaluate
the best combination of modalities for a feature."""

import json
import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Generic

import pandas as pd
from tqdm import tqdm

from AutoCarver.combinations.utils.combinations import (
    combination_formatter,
    consecutive_combinations,
    nan_combinations,
    order_apply_combination,
    xagg_apply_combination,
)
from AutoCarver.combinations.utils.target_rate import TargetRate, XAgg
from AutoCarver.combinations.utils.testing import Keys, is_viable, test_viability
from AutoCarver.features import BaseFeature, GroupedList


class AggregatedSample:
    """Sample class to store aggregated samples.

    The public ``xagg`` is typed as mandatory ``pd.Series | pd.DataFrame``. The
    constructor accepts ``None`` so that placeholders (default factories, optional
    dev samples) are expressible, but reading ``.xagg`` on an unset sample raises.
    Use :attr:`has_xagg` to check presence without triggering.
    """

    def __init__(
        self,
        xagg: pd.Series | pd.DataFrame | None = None,
        _raw: pd.Series | pd.DataFrame | None = None,
    ) -> None:
        self._xagg: pd.Series | pd.DataFrame | None = xagg
        self._raw: pd.Series | pd.DataFrame | None = _raw
        # setting xtab_dev to xtab if not provided
        if self._raw is None and self._xagg is not None:
            self._raw = self._xagg.copy()

    @property
    def xagg(self) -> pd.Series | pd.DataFrame:
        """Returns the aggregated sample, or raises if not set."""
        if self._xagg is None:
            raise RuntimeError("[AggregatedSample] xagg is not set")
        return self._xagg

    @xagg.setter
    def xagg(self, value: pd.Series | pd.DataFrame) -> None:
        self._xagg = value

    @property
    def has_xagg(self) -> bool:
        """Whether xagg is set."""
        return self._xagg is not None

    @property
    def raw(self) -> pd.Series | pd.DataFrame | None:
        """Returns the raw value of the xagg"""
        return self._raw

    @raw.setter
    def raw(self, value: pd.Series | pd.DataFrame | None) -> None:
        """Sets the raw value of the xagg"""

        # setting raw value
        self._raw = value

        # copying xagg
        if value is not None:
            self.xagg = value.copy()

    @property
    def shape(self) -> tuple[int, ...]:
        """Returns the shape of the xagg"""
        return self.xagg.shape

    @property
    def index(self) -> pd.Index:
        """Returns the index of the xagg"""
        return self.xagg.index

    @property
    def columns(self) -> pd.Index:
        """Returns the columns of the xagg"""
        return self.xagg.columns

    @property
    def values(self):
        """Returns the values of the xagg"""
        return self.xagg.values

    def groupby(self, *args, **kwargs):
        """Groups the xagg by the specified indices"""
        return self.xagg.groupby(*args, **kwargs)


@dataclass
class AggregatedSamples:
    """stores train and dev samples"""

    train: AggregatedSample = field(default_factory=AggregatedSample)
    dev: AggregatedSample = field(default_factory=AggregatedSample)

    def set(self, train: pd.Series | pd.DataFrame | None, dev: pd.Series | pd.DataFrame | None = None) -> None:
        """Sets the train and dev samples"""

        # setting train and dev samples
        self.train = AggregatedSample(train)
        self.dev = AggregatedSample(dev)

    def dropna(self, feature_nan: str) -> None:
        """Removes nans from the samples"""

        # removing nans from crosstabs (filter_nan returns None when raw is None;
        # only assign when filtering actually produced a frame)
        dev_filtered = filter_nan(self.dev.raw, feature_nan)
        if dev_filtered is not None:
            self.dev.xagg = dev_filtered
        train_filtered = filter_nan(self.train.raw, feature_nan)
        if train_filtered is not None:
            self.train.xagg = train_filtered

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


class CombinationEvaluator(ABC, Generic[XAgg]):
    """CombinationEvaluator class to evaluate
    the best combination of modalities for a feature."""

    __name__ = "CombinationEvaluator"

    is_y_binary = False
    is_y_continuous = False
    sort_by = None

    def __init__(
        self,
        *,
        verbose: bool = False,
        target_rate: TargetRate[XAgg] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        verbose : bool, optional
            Whether to print progress / statistics, by default ``False``.

        target_rate : TargetRate, optional
            Target rate strategy. If ``None``, each evaluator subclass picks its own
            default in :meth:`_init_target_rate`.
        """
        self.verbose: bool = verbose

        # attributes to be set by get_best_combination
        self._feature: BaseFeature | None = None
        self.samples: AggregatedSamples = AggregatedSamples()
        self._statistics_cache = None
        self.target_rate: TargetRate[XAgg] = self._init_target_rate(target_rate)

    @abstractmethod
    def _init_target_rate(self, target_rate: TargetRate[XAgg] | None) -> TargetRate[XAgg]:
        """Initializes target rate."""
        if target_rate is None:
            raise NotImplementedError("Subclasses must implement _init_target_rate to provide a default target rate")
        return target_rate

    @property
    def feature(self) -> BaseFeature:
        """Current feature under evaluation (set by ``get_best_combination``)."""
        if self._feature is None:
            raise RuntimeError(f"[{self.__name__}] feature is not set; call get_best_combination first")
        return self._feature

    @feature.setter
    def feature(self, value: BaseFeature) -> None:
        self._feature = value

    def __repr__(self) -> str:
        return f"{self.__name__}(target_rate={self.target_rate.__name__})"

    def _group_xagg_by_combinations(self, combinations: Iterable[list]) -> Iterator[dict]:
        """Streams ``{xagg, combination, index_to_groupby}`` for each combination.

        Yields one entry at a time so the caller can fuse this with scoring
        without ever materialising the full 92k-entry list (the dominant
        source of peak RAM in the continuous case).
        """
        for combination in tqdm(combinations, desc="Grouping modalities   ", disable=not self.verbose):
            index_to_groupby = combination_formatter(combination)
            yield {
                "xagg": self._grouper(self.samples.train, index_to_groupby),
                "combination": combination,
                "index_to_groupby": index_to_groupby,
            }

    def _compute_associations(self, grouped_xaggs: Iterable[dict]) -> Iterator[dict]:
        """Streams light association dicts ``{combination, index_to_groupby, <metric>}``.

        The heavy ``xagg`` is consumed for scoring and then **dropped** — only
        the lightweight identifying fields and the association metric are
        carried downstream so peak memory does not grow with the number of
        combinations. Output is in arrival order; the caller is expected to
        sort by the configured metric.
        """
        n_obs: int = self.samples.train.xagg.apply(sum).sum()  # type: ignore
        for grouped_xagg in tqdm(grouped_xaggs, desc="Computing associations", disable=not self.verbose):
            measure = self._association_measure(grouped_xagg["xagg"], n_obs=n_obs)
            yield {
                "combination": grouped_xagg["combination"],
                "index_to_groupby": grouped_xagg["index_to_groupby"],
                **measure,
            }

    def _get_best_association(self, combinations: Iterable[list[list[str]]]) -> dict | None:
        """Streams grouping → scoring → viability in one pass.

        - ``combinations`` is consumed lazily (generator-friendly).
        - Grouping and scoring are fused: each combination is grouped, scored,
          and the heavy xagg is dropped before moving to the next one.
        - The lightweight associations are collected (each ~ a few hundred
          bytes) and sorted by ``self.sort_by``; the viability walk then
          lazily rebuilds the xagg only for combinations it actually tests
          (see :meth:`_test_viability_train`).

        Returns
        -------
        dict | None
            Best viable association (with grouped xagg rebuilt on demand),
            or ``None`` if no combination satisfied the viability checks.
        """
        # fused stream: combinations -> grouped -> scored (light, no xagg)
        association_stream = self._compute_associations(self._group_xagg_by_combinations(combinations))

        # collect & sort by metric (NaN / None last, matching prior pandas behaviour)
        associations = list(association_stream)
        sort_key = self.sort_by

        def _key(assoc: dict) -> float:
            value = assoc.get(sort_key)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return float("-inf")
            return float(value)

        associations.sort(key=_key, reverse=True)

        # testing viability of combination (lazy-rebuilds xagg per candidate)
        best_combination = self._get_viable_combination(associations)

        # applying best combination to feature labels and xtab
        self._apply_best_combination(best_combination)

        return best_combination

    def _apply_best_combination(self, best_association: dict | None) -> None:
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

            # udpating statistics — `apply_combination` always populates
            # `samples.train.raw`; narrow Optional so the overloaded
            # compute() returns DataFrame (not DataFrame | None).
            raw = self.samples.train.raw
            if raw is None:
                raise RuntimeError(f"[{self.__name__}] samples.train.raw is not populated after apply_combination")
            self.feature.statistics = self.target_rate.compute(raw)

    def _get_best_combination_non_nan(self) -> dict | None:
        """Computes associations of the tab for each combination of non-nans

        - dropna has to be set to True
        """

        # raw ordering without nans (labels are populated before this method runs)
        feature_labels = self.feature.labels
        if feature_labels is None:
            raise RuntimeError(f"[{self.__name__}] feature labels are not populated")
        raw_labels = GroupedList(feature_labels[:])

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

    def _get_best_combination_with_nan(self, best_combination: dict | None) -> dict | None:
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
        self,
        feature: BaseFeature,
        xagg: pd.Series | pd.DataFrame | None,
        xagg_dev: pd.Series | pd.DataFrame | None = None,
        *,
        max_n_mod: int,
        min_freq: float,
        dropna: bool,
    ) -> dict | None:
        """Computes best combination of modalities for the feature"""

        self.max_n_mod = max_n_mod
        self.min_freq = min_freq
        self.dropna = dropna

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
        """Testing the viability of the combination on xagg_train.

        When the combination comes from the streaming pipeline the heavy
        grouped xagg was dropped after scoring, so we rebuild it lazily here
        — once, and only for combinations actually selected for viability
        checks (typically the top handful). The rebuilt xagg is cached on the
        combination dict so subsequent dev/apply steps reuse it.
        """
        xagg = combination.get("xagg")
        if xagg is None:
            xagg = self._grouper(self.samples.train, combination["index_to_groupby"])
            combination["xagg"] = xagg

        # computing target rate and frequency per value
        train_rates = self.target_rate.compute(xagg)

        # viability on train sample:
        result = test_viability(train_rates, self.min_freq, self.target_rate.__name__)

        return result

    def _test_viability_dev(self, test_results: dict, combination: dict) -> dict:
        """testing the viability of the combination on xagg_dev"""

        # case 0: not viable on train or no test sample -> not testing for robustness
        if not test_results[Keys.VIABLE.value] or not self.samples.dev.has_xagg:
            return {**test_results, "dev": {Keys.VIABLE.value: None}}

        # case 1: test sample provided and viable on train -> testing robustness
        # getting train target rates
        train_target_rate = test_results["train_rates"][self.target_rate.__name__]

        # grouping the dev sample per modality
        grouped_xagg_dev = self._grouper(self.samples.dev, combination["index_to_groupby"])

        # computing target rate and frequency per modality
        dev_rates = self.target_rate.compute(grouped_xagg_dev)

        # viability on dev sample:
        dev_results = test_viability(dev_rates, self.min_freq, self.target_rate.__name__, train_target_rate)
        test_results = {**test_results, **dev_results}

        # checking for viability on both samples
        test_results[Keys.VIABLE.value] = is_viable(test_results)

        return test_results

    def _get_viable_combination(self, associations: list[dict]) -> dict | None:
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
            if test_results[Keys.VIABLE.value]:
                viable_combination = combination

                # historizing remaining combinations/not tested
                self._historize_remaining_combinations(associations, n_combination)
                break

        if self.verbose:  # verbose if requested
            print("\n")

        return viable_combination

    @abstractmethod
    def _grouper(self, xagg: AggregatedSample, groupby: dict[str, str]) -> XAgg:
        """Helper to group XAGG's values by groupby (carver specific)"""

    @abstractmethod
    def _association_measure(
        self,
        xagg: AggregatedSample | pd.Series | pd.DataFrame,
        n_obs: int | None = None,
        tol: float = 1e-10,
    ) -> dict[str, float | None]:
        """Helper to measure association between X and y (carver specific).

        Return value type is widened to ``float | None`` so the continuous
        Kruskal–Wallis path (which returns ``None`` when scipy raises) can
        share the base signature with the binary chi² path.
        """

    def _historize_remaining_combinations(self, associations: list[dict], n_combination: int) -> None:
        """historizes the remaining combinations that have not been tested"""

        # historizing all remaining combinations
        for combination in associations[n_combination + 1 :]:
            # historizing not tested combination
            self.feature.historize({**clean_combination(combination, self.feature), "info": "Not checked"})

    def _historize_combination(self, combination: dict, test_results: dict) -> None:
        """historizes the test results of the combination"""

        # keeping only relevant information
        test_results.update(clean_combination(combination, self.feature))

        # checking for viability
        if test_results[Keys.VIABLE.value]:
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
        self.feature.historize(test_results)

    def _historize_raw_combination(self):
        """historizes the raw combination"""

        # narrow Optional: this method is only called after samples.set() has populated raw
        raw = self.samples.train.raw
        if raw is None:
            raise RuntimeError(f"[{self.__name__}] samples.train.raw is not populated")

        # setting feature's statistics
        self.feature.statistics = self.target_rate.compute(raw)

        # computing association of sample
        raw_association = self._association_measure(raw, n_obs=sum(raw.apply(sum)))

        # computing number of modalities
        n_mod = raw.shape[0]

        # creating info message
        info = "Raw distribution"

        # adding info if n_mod > max_n_mod
        if n_mod > self.max_n_mod:
            info += f" (n_mod={n_mod}>max_n_mod={self.max_n_mod})"

        # historizing raw combination
        combination = {
            "info": info,
            **raw_association,
            "combination": {modality: modality for modality in raw.index},
        }

        # historizing within feature
        self.feature.historize({**combination, "n_mod": n_mod, "dropna": False})

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
            "target_rate": self.target_rate.__name__,
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
    def load(cls, file: str | dict) -> "CombinationEvaluator":
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
            with open(file, encoding="utf-8") as json_file:
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

        # resolve target_rate name → instance using the subclass registry
        target_rate_name = combinations_json.pop("target_rate", None)
        target_rate = None
        registry = {tr().__name__: tr for tr in getattr(cls, "_target_rate_classes", [])}
        if target_rate_name in registry:
            target_rate = registry[target_rate_name]()

        # strip non-constructor fields before passing remaining kwargs
        combinations_json.pop("sort_by", None)
        return cls(target_rate=target_rate, **combinations_json)


def filter_nan(xagg: pd.Series | pd.DataFrame | None, str_nan: str) -> pd.Series | pd.DataFrame | None:
    """Filters out nans from crosstab or y values"""

    # cehcking for values in crosstab
    filtered_xagg = None
    if xagg is not None:
        # filtering out nans if requested from train crosstab
        filtered_xagg = xagg.copy()
        if str_nan in xagg.index:
            filtered_xagg = xagg.drop(str_nan, axis=0)

    return filtered_xagg


def clean_combination(combination: dict, feature: BaseFeature, remove_train_rates: bool = False) -> dict:
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
