"""Module for continuous combination evaluators."""

from abc import ABC

import pandas as pd
from scipy.stats import kruskal

from AutoCarver.combinations.continuous.continuous_target_rates import ContinuousTargetRate, TargetMean, TargetMedian
from AutoCarver.combinations.utils.combination_evaluator import AggregatedSample, CombinationEvaluator


class ContinuousCombinationEvaluator(CombinationEvaluator, ABC):
    """Continuous combination evaluator class."""

    is_y_continuous = True
    _target_rate_classes: list[type[ContinuousTargetRate]] = [TargetMean, TargetMedian]

    def _init_target_rate(self, target_rate: ContinuousTargetRate | None) -> ContinuousTargetRate:
        """Initializes target rate."""
        if target_rate is None:
            return TargetMean()
        elif not isinstance(target_rate, ContinuousTargetRate):
            raise ValueError("target_rate must be a ContinuousTargetRate")
        return target_rate

    def _association_measure(
        self, xagg: AggregatedSample, n_obs: int | None = None, tol: float = 1e-10
    ) -> dict[str, float | None]:
        """Computes measures of association between feature and quantitative target.

        Parameters
        ----------
        xagg : pd.DataFrame
            Values taken by y for each of x's modalities.

        Returns
        -------
        dict[str, float]
            Kruskal-Wallis' H as a dict.
        """
        _, _ = n_obs, tol  # unused attribute

        # Kruskal-Wallis' H
        try:
            return {"kruskal": kruskal(*tuple(xagg.values))[0]}
        except (ValueError, IndexError):
            return {"kruskal": None}

    def _grouper(self, xagg: AggregatedSample, groupby: dict[str, str]) -> pd.Series:
        """Groups values of y

        Parameters
        ----------
        yval : pd.Series
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
