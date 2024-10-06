""" Module for continuous combination evaluators. """

from abc import ABC
from numpy import mean
from pandas import DataFrame, Series
from scipy.stats import kruskal
from ..utils.combination_evaluator import CombinationEvaluator


class ContinuousCombinationEvaluator(CombinationEvaluator, ABC):
    is_y_continuous = True

    def _association_measure(self, xagg: Series, n_obs: int = None) -> dict[str, float]:
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

    def _compute_target_rates(self, xagg: Series = None) -> DataFrame:
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
