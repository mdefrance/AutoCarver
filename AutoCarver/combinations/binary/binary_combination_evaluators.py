"""Module for binary combination evaluators."""

from abc import ABC

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from AutoCarver.combinations.binary.binary_target_rates import BinaryTargetRate, OddsRatio, TargetMean, Woe
from AutoCarver.combinations.utils.combination_evaluator import (
    AggregatedSample,
    CombinationEvaluator,
)


class BinaryCombinationEvaluator(CombinationEvaluator, ABC):
    """Binary combination evaluator class."""

    is_y_binary = True
    _target_rate_classes: list[type[BinaryTargetRate]] = [TargetMean, OddsRatio, Woe]

    def _init_target_rate(self, target_rate: BinaryTargetRate) -> None:
        """Initializes target rate."""
        if target_rate is None:
            self.target_rate = TargetMean()
        elif not isinstance(target_rate, BinaryTargetRate):
            raise ValueError("target_rate must be a BinaryTargetRate")
        else:
            self.target_rate = target_rate

    def _association_measure(
        self, xagg: AggregatedSample, n_obs: int | None = None, tol: float = 1e-10
    ) -> dict[str, float]:
        """Computes measures of association between feature and target by crosstab.

        Parameters
        ----------
        xtab : pd.DataFrame
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
        chi2 = chi2_contingency(xagg.values + tol)[0]

        # Cramér's V
        cramerv = np.sqrt(chi2 / n_obs)
        if pd.notna(cramerv):
            cramerv = round(cramerv / tol) * tol

        # Tschuprow's T
        tschuprowt = cramerv / np.sqrt(np.sqrt(n_mod_x - 1))
        if pd.notna(tschuprowt):
            tschuprowt = round(tschuprowt / tol) * tol

        return {"cramerv": cramerv, "tschuprowt": tschuprowt}

    def _grouper(self, xagg: AggregatedSample, groupby: dict) -> pd.DataFrame:
        """Groups a crosstab by groupby and sums column values by groups (vectorized)

        Parameters
        ----------
        xagg : pd.DataFrame
            crosstab between X and y
        groupby : list[str]
            indices to group by

        Returns
        -------
        DataFrame
            Crosstab grouped by indices
        """
        # all indices that may be duplicated
        index_values = np.array([groupby.get(index_value, index_value) for index_value in xagg.index])

        # all unique indices deduplicated
        unique_indices = np.unique(index_values)

        # initiating summed up array with zeros
        summed_values = np.zeros((len(unique_indices), len(xagg.columns)))

        # for each unique_index found in index_values sums xtab.Values at corresponding position
        # in summed_values
        np.add.at(summed_values, np.searchsorted(unique_indices, index_values), xagg.values)

        # converting back to dataframe
        return pd.DataFrame(summed_values, index=unique_indices, columns=xagg.columns)


class TschuprowtCombinations(BinaryCombinationEvaluator):
    """Tschuprow's T based combination evaluation toolkit"""

    sort_by = "tschuprowt"


class CramervCombinations(BinaryCombinationEvaluator):
    """Cramér's V based combination evaluation toolkit"""

    sort_by = "cramerv"
