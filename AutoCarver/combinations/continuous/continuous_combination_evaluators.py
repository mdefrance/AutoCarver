""" Module for continuous combination evaluators. """

from abc import ABC

from pandas import Series
from scipy.stats import kruskal

from ..utils.combination_evaluator import AggregatedSample, CombinationEvaluator
from .continuous_target_rates import ContinuousTargetRate, TargetMean


class ContinuousCombinationEvaluator(CombinationEvaluator, ABC):
    """Continuous combination evaluator class."""

    is_y_continuous = True

    def _init_target_rate(self, target_rate: ContinuousTargetRate) -> None:
        """Initializes target rate."""
        if target_rate is None:
            self.target_rate = TargetMean()
        elif not isinstance(target_rate, ContinuousTargetRate):
            raise ValueError("target_rate must be a ContinuousTargetRate")
        else:
            self.target_rate = target_rate

    def _association_measure(
        self, xagg: AggregatedSample, n_obs: int = None, tol: float = 1e-10
    ) -> dict[str, float]:
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
        _, _ = n_obs, tol  # unused attribute

        # Kruskal-Wallis' H
        return {"kruskal": kruskal(*tuple(xagg.values))[0]}

    def _grouper(self, xagg: AggregatedSample, groupby: dict[str:str]) -> Series:
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


class KruskalCombinations(ContinuousCombinationEvaluator):
    """Kruskal-Wallis' H based combination evaluation toolkit"""

    sort_by = "kruskal"
