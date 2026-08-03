"""set of target rates for binary classification"""

from abc import ABC

import numpy as np
import pandas as pd

from AutoCarver.combinations.utils import TargetRate


class BinaryTargetRate(TargetRate[pd.DataFrame], ABC):
    """Binary target rate class."""

    __name__ = "binary_target_rate"


class TargetMean(BinaryTargetRate):
    """Mean target rate class."""

    __name__ = "target_mean"

    def _compute(self, xagg: pd.DataFrame) -> pd.Series:
        """Computes the mean target rate.

        Parameters
        ----------
        xagg : pd.DataFrame
            A crosstab.

        Returns
        -------
        Series
            Mean target rate.
        """
        return xagg[1].divide(xagg.sum(axis=1))


class OddsRatio(TargetMean):
    """Odds ratio."""

    __name__ = "odds_ratio"

    def _compute(self, xagg: pd.DataFrame) -> pd.Series:
        """Computes the mean target rate.

        Parameters
        ----------
        xagg : pd.DataFrame
            A crosstab.

        Returns
        -------
        Series
            Mean target rate.
        """
        target_rate = super()._compute(xagg)
        return target_rate / (1 - target_rate)


# class LogsOddsRatio(OddsRatio):
#     """Logs Odds ratio. same as WOE"""

#     __name__ = "log_odds_ratio"

#     def _compute(self, xagg: pd.DataFrame) -> pd.Series:
#         """Computes the mean target rate.

#         Parameters
#         ----------
#         xagg : pd.DataFrame
#             A crosstab.

#         Returns
#         -------
#         Series
#             Mean target rate.
#         """
#         return log(super()._compute(xagg))


# class GiniCoefficient(BinaryTargetRate):
#     """Gini coefficient class."""

#     __name__ = "gini_coefficient"

#     def _compute(self, xagg: pd.DataFrame) -> pd.Series:
#         """Computes the Gini coefficient.

#         Parameters
#         ----------
#         xagg : pd.DataFrame
#             A crosstab.

#         Returns
#         -------
#         Series
#             Gini coefficient.
#         """
#         sum_f = xagg.sum(axis=1)
#         squared = xagg.divide(sum_f, axis=0) ** 2
#         gini = 1 - squared.sum(axis=1)
#         return gini


class Woe(BinaryTargetRate):
    """Weight of evidence class."""

    __name__ = "woe"

    def _compute(self, xagg: pd.DataFrame) -> pd.Series:
        """Computes the Weight of evidence."""
        sum_f = xagg.sum(axis=1)
        means = xagg.divide(sum_f, axis=0)
        woe = np.log(means[1] / means[0])
        return woe


# class IV(Woe):
#     """Information Value coefficient class. TODO use for feature selection"""

#     __name__ = "iv"

#     def _compute(self, xagg: pd.DataFrame) -> pd.Series:
#         """Computes the Information Value ."""
#         sum_f = xagg.sum(axis=1)
#         means = xagg.divide(sum_f, axis=0)
#         woe = log(means[1] / means[0])
#         iv = (means[1] - means[0]) * woe
#         return iv
