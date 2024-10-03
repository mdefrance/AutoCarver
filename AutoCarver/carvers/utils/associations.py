from typing import Union

from .testing import is_viable, _test_viability
from pandas import Series, DataFrame
from ...features import BaseFeature, GroupedList
from .combinations import (
    nan_combinations,
    consecutive_combinations,
    format_combinations,
    order_apply_combination,
    xagg_apply_combination,
)
from tqdm.autonotebook import tqdm

from abc import abstractmethod, ABC


from numpy import add, array, searchsorted, sqrt, unique, zeros
from pandas import DataFrame, Series, crosstab
from scipy.stats import chi2_contingency

from numpy import mean, unique
from pandas import DataFrame, Series
from scipy.stats import kruskal


class PartialCombinationEvaluator:
    def __init__(self, max_n_mod, min_freq, sort_by):
        self.max_n_mod = max_n_mod
        self.min_freq = min_freq
        self.sort_by = sort_by

    def __call__(
        self,
        feature: BaseFeature,
        xagg: DataFrame,
        xagg_dev: DataFrame = None,
        dropna: bool = False,
        verbose: bool = False,
    ):
        return CombinationEvaluator(
            feature=feature,
            xagg=xagg,
            sort_by=self.sort_by,
            max_n_mod=self.max_n_mod,
            min_freq=self.min_freq,
            xagg_dev=xagg_dev,
            dropna=dropna,
            verbose=verbose,
        )


class CombinationEvaluator(ABC):

    __name__ = "CombinationEvaluator"

    is_y_binary = False
    is_y_continuous = False
    sort_by = None

    def __init__(
        self,
        feature: BaseFeature,
        xagg: DataFrame,
        max_n_mod: int,
        min_freq: float,
        xagg_dev: DataFrame = None,
        dropna: bool = False,
        verbose: bool = False,
    ) -> None:
        self.verbose = verbose
        self.dropna = dropna
        self.feature = feature
        self.max_n_mod = max_n_mod
        self.min_freq = min_freq
        self.raw_xagg = xagg.copy()
        self.raw_xagg_dev = None
        if xagg_dev is not None:
            self.raw_xagg_dev = xagg_dev.copy()

        self.xagg = xagg
        self.xagg_dev = xagg_dev

        # historizing raw combination
        self._historize_raw_combination()

    @abstractmethod
    def _grouper(
        self, xagg: Union[Series, DataFrame], groupby: dict[str, str]
    ) -> Union[Series, DataFrame]:
        """"""

    @abstractmethod
    def _association_measure(self, xagg: Union[Series, DataFrame], n_obs: int) -> dict[str, float]:
        """"""

    @abstractmethod
    def _compute_target_rates(self, xagg: Union[Series, DataFrame]) -> DataFrame:
        """ """

    def _historize_raw_combination(self):
        # historizing raw combination TODO
        raw_association = {
            "index_to_groupby": {modality: modality for modality in self.xagg.index},
            self.sort_by: self._association_measure(
                self.xagg.dropna(), n_obs=sum(self.xagg.dropna().apply(sum))
            )[self.sort_by],
        }
        self._historize(self.feature, raw_association, self.feature.labels)

    def _historize(self, *args, **kwargs) -> None:
        pass

    def _group_xagg_by_combinations(self, combinations: list[list]) -> list[dict]:
        """groups xagg by combinations of indices"""

        # values to groupby indices with
        indices_to_groupby = format_combinations(combinations)

        # grouping tab by its indices
        return [
            {
                "xagg": self._grouper(self.xagg, index_to_groupby),
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
        n_obs = self.xagg.apply(sum).sum()

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

    def _test_viability_train(self, combination: dict) -> dict:
        """testing the viability of the combination on xagg_train"""

        # computing target rate and frequency per value
        train_rates = self._compute_target_rates(combination["xagg"])

        # viability on train sample:
        return _test_viability(train_rates, self.min_freq)

    def _test_viability_dev(self, test_results: dict, combination: dict) -> dict:
        """testing the viability of the combination on xagg_dev"""

        # case 0: not viable on train or no test sample -> not testing for robustness
        if not test_results.get("train_viable") or self.xagg_dev is None:
            return {**test_results, "dev": {"viable": None}}
        # case 1: test sample provided -> testing robustness

        # getting train target rates
        train_target_rate = test_results.pop("train_rates")["target_rate"]

        # grouping the dev sample per modality
        grouped_xagg_dev = self._grouper(self.xagg_dev, combination["index_to_groupby"])

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
            self._historize(
                feature=self.feature,
                association=combination,
                n_combination=n_combination,
                associations_xagg=associations,
                dropna=self.dropna,
                verbose=self.verbose,
                **test_results,
            )

            # best combination found: breaking the loop on combinations
            if is_viable(test_results):
                viable_combination = combination
                break

        if self.verbose:  # verbose if requested
            print("\n")

        return viable_combination

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

            # applying best_combination to raw xagg and xagg_dev
            self.xagg = xagg_apply_combination(self.raw_xagg, labels)
            self.xagg_dev = xagg_apply_combination(self.raw_xagg_dev, labels)

            # updating feature's values and xagg indices accordingly
            self.feature.update(labels, convert_labels=True)
            print(
                "TODO",
                self.xagg.index,
                self.feature.labels,
                labels,
                best_association["combination"],
            )
            self.xagg.index = self.feature.labels  # TODO: check if this is necessary
            self.xagg_dev.index = self.feature.labels

    def _get_best_combination_non_nan(self) -> DataFrame:
        """Computes associations of the tab for each combination of non-nans"""

        # raw ordering without nans
        raw_labels = GroupedList(self.feature.labels[:])

        # removing nans if any
        if self.feature.has_nan:

            # removing nans for combination of non-nans
            raw_labels.remove(self.feature.nan)

            # removing nans from crosstabs
            self.xagg_dev = filter_nan(self.raw_xagg_dev, self.feature.nan)
            self.xagg = filter_nan(self.raw_xagg, self.feature.nan)

        # checking for non-nan values
        if len(self.xagg.shape[0]) > 1:

            # all possible consecutive combinations
            combinations = consecutive_combinations(raw_labels, self.max_n_mod)

            # getting most associated combination
            return self._get_best_association(combinations)

        return None

    def _get_best_combination_with_nan(self, best_combination: dict) -> DataFrame:
        """Computes associations of the tab for each combination with nans"""

        # setting dropna to user-requested value
        self.feature.dropna = self.dropna

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

    def get_best_combination(self) -> tuple[GroupedList, DataFrame, DataFrame]:
        """Computes best combination of modalities for the feature"""

        # getting best combination without NaNs
        best_combination = self._get_best_combination_non_nan()

        # grouping NaNs if requested to drop them (dropna=True)
        return self._get_best_combination_with_nan(best_combination)


class BinaryCombinationEvaluator(CombinationEvaluator, ABC):

    is_y_binary = True

    def _compute_target_rates(self, xagg: DataFrame = None) -> DataFrame:
        """Prints a binary xtab's statistics

        - there should bnot be nans in xagg

        Parameters
        ----------
        xagg : Dataframe
            A crosstab, by default None

        Returns
        -------
        DataFrame
            Target rate and frequency per modality
        """
        # checking for an xtab
        stats = None
        if xagg is not None:
            # target rate and frequency statistics per modality
            stats = DataFrame(
                {
                    # target rate per modality
                    "target_rate": xagg[1].divide(xagg.sum(axis=1)),
                    # frequency per modality
                    "frequency": xagg.sum(axis=1) / xagg.sum().sum(),
                }
            )

        return stats

    def _association_measure(self, xagg: DataFrame, n_obs: int) -> dict[str, float]:
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
        n_mod_x = xagg.shape[0]

        # Chi2 statistic
        chi2 = chi2_contingency(xagg)[0]

        # Cramér's V
        cramerv = sqrt(chi2 / n_obs)

        # Tschuprow's T
        tschuprowt = cramerv / sqrt(sqrt(n_mod_x - 1))

        return {"cramerv": cramerv, "tschuprowt": tschuprowt}

    def _grouper(self, xagg: DataFrame, groupby: dict) -> DataFrame:
        """Groups a crosstab by groupby and sums column values by groups (vectorized)

        Parameters
        ----------
        xagg : DataFrame
            crosstab between X and y
        groupby : list[str]
            indices to group by

        Returns
        -------
        DataFrame
            Crosstab grouped by indices
        """
        # all indices that may be duplicated
        index_values = array([groupby.get(index_value, index_value) for index_value in xagg.index])

        # all unique indices deduplicated
        unique_indices = unique(index_values)

        # initiating summed up array with zeros
        summed_values = zeros((len(unique_indices), len(xagg.columns)))

        # for each unique_index found in index_values sums xtab.Values at corresponding position
        # in summed_values
        add.at(summed_values, searchsorted(unique_indices, index_values), xagg.values)

        # converting back to dataframe
        return DataFrame(summed_values, index=unique_indices, columns=xagg.columns)


class TschuprowtCombinations(BinaryCombinationEvaluator):

    sort_by = "tschuprowt"


class CramervCombinations(BinaryCombinationEvaluator):

    sort_by = "cramerv"


class ContinuousCombinationEvaluator(CombinationEvaluator, ABC):
    is_y_continuous = True

    def _grouper(self, xagg: Series, groupby: dict[str:str]) -> Series:
        """Groups values of y

        Parameters
        ----------
        yval : Series
            _description_
        groupby : _type_
            _description_

        Returns
        -------
        Series
            _description_
        """
        # TODO: convert this to the vectorial version like BinaryCarver
        return xagg.groupby(groupby).sum()

    def _association_measure(self, xagg: Series, n_obs: int) -> dict[str, float]:
        """Computes measures of association between feature and quantitative target.

        Parameters
        ----------
        xagg : DataFrame
            Values taken by y for each of x's modalities.

        Returns
        -------
        dict[str, float]
            Kruskal-Wallis' H as a dict.
        """
        _ = n_obs  # unused attribute

        # Kruskal-Wallis' H
        return {"kruskal": kruskal(*tuple(xagg.values))[0]}

    def _printer(self, xagg: Series = None) -> DataFrame:
        """Prints a continuous yval's statistics

        Parameters
        ----------
        xagg : Series
            A series of values of y by modalities of x, by default None

        Returns
        -------
        DataFrame
            Target rate and frequency per modality
        """
        # checking for an xtab
        stats = None
        if xagg is not None:
            # target rate and frequency statistics per modality
            stats = DataFrame(
                {
                    # target rate per modality
                    "target_rate": xagg.apply(mean),
                    # frequency per modality
                    "frequency": xagg.apply(len) / xagg.apply(len).sum(),
                }
            )

        return stats


class KruksalCombinations(ContinuousCombinationEvaluator):

    sort_by = "kruskal"


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
