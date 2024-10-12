""" Module for binary combination evaluators. """

from abc import ABC
from numpy import add, array, searchsorted, sqrt, unique, zeros
from pandas import DataFrame
from scipy.stats import chi2_contingency
from ..utils.combination_evaluator import CombinationEvaluator, AggregatedSample

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

    def _association_measure(self, xagg: DataFrame, n_obs: int = None) -> dict[str, float]:
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

    def _grouper(self, xagg: AggregatedSample, groupby: dict) -> DataFrame:
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
