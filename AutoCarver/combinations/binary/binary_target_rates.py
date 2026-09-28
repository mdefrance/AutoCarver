"""set of target rates for binary classification"""

from abc import ABC

import numpy as np
import pandas as pd

from AutoCarver.combinations.utils import TargetRate


class BinaryTargetRate(TargetRate[pd.DataFrame], ABC):
    """Binary target rate class."""

    __name__ = "binary_target_rate"

    def fit_reference(self, raw_xagg: pd.DataFrame) -> None:
        """No-op hook fixing a train reference before any candidate is scored.

        :class:`Woe` overrides it to fix the train class ratio.
        """


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
    """Scorecard weight of evidence per modality.

    ``woe = ln((n1_i / N1) / (n0_i / N0)) = logit(p_i) - ln(N1 / N0)``, where
    ``N1 / N0`` is the class ratio of the feature's raw **train** crosstab,
    fixed once per feature (:meth:`fit_reference`). Every later call — a train
    candidate grouping, a dev grouping or a production sample — applies that
    same train ratio, as a scorecard applies its train WoE table.
    """

    __name__ = "woe"

    def __init__(self) -> None:
        self._log_ratio: float | None = None

    @property
    def log_ratio(self) -> float:
        """The fixed train ``ln(N1 / N0)`` (raises until :meth:`fit_reference` runs)."""
        if self._log_ratio is None:
            raise RuntimeError(f"[{self.__name__}] reference is not fit; call fit_reference(raw_xagg) first")
        return self._log_ratio

    def fit_reference(self, raw_xagg: pd.DataFrame) -> None:
        """Fixes the train class ratio ``ln(N1 / N0)`` from the feature's raw train crosstab."""
        self._log_ratio = float(np.log(raw_xagg[1].sum() / raw_xagg[0].sum()))

    def reference_to_json(self) -> dict | None:
        """Snapshots the fitted train class ratio."""
        if self._log_ratio is None:
            return None
        return {"log_ratio": self._log_ratio}

    def load_reference(self, payload: dict | None) -> None:
        """Restores the train class ratio snapshotted by :meth:`reference_to_json`."""
        if payload is not None:
            self._log_ratio = payload["log_ratio"]

    def _compute(self, xagg: pd.DataFrame) -> pd.Series:
        """Computes the weight of evidence against the fixed train class ratio."""
        return np.log(xagg[1] / xagg[0]) - self.log_ratio


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
