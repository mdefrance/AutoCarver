"""set of target rates for ordinal targets"""

from abc import ABC
from typing import overload

import numpy as np
import pandas as pd

from AutoCarver.combinations.utils import TargetRate


class OrdinalTargetRate(TargetRate[pd.DataFrame], ABC):
    """Ordinal target rate class.

    Operates on an ordered contingency table ``feature-groups (rows) ×
    ordinal-target-levels (cols)`` — the same crosstab shape the binary target
    rates consume, only with one column per ordinal level instead of two.
    """

    __name__ = "ordinal_target_rate"

    @overload
    def compute(self, xagg: pd.Series | pd.DataFrame) -> pd.DataFrame: ...
    @overload
    def compute(self, xagg: None) -> None: ...
    def compute(self, xagg: pd.Series | pd.DataFrame | None) -> pd.DataFrame | None:
        """Computes the target rate.

        Parameters
        ----------
        xagg : pd.DataFrame
            A crosstab (feature groups × ordinal target levels).

        Returns
        -------
        pd.DataFrame
            Per-group target rate, ``frequency`` and ``count``.
        """
        # checking for an xtab
        if xagg is not None:
            # count + frequency per modality (count carried for CI-based viability tests)
            count = xagg.sum(axis=1)
            frequency = count / count.sum()

            # computing target rate. `_compute` expects pd.DataFrame (Generic
            # XAgg=DataFrame); compute()'s wide signature is for LSP matching,
            # callers always pass a crosstab here.
            return pd.DataFrame(
                {self.__name__: self._compute(xagg), "frequency": frequency, "count": count}  # type: ignore
            )
        return None


class TargetMeanLevel(OrdinalTargetRate):
    """Mean ordinal level per modality.

    The per-group rate is ``Σ_j level_j · n_gj / n_g+`` where ``level_j`` is read
    from the (integer) crosstab columns. It is monotone in the target's order, so
    it drives both the ``min_freq`` viability test and the train/dev
    rank-preservation veto exactly like the binary/continuous target means.
    """

    __name__ = "target_mean_level"

    def _compute(self, xagg: pd.DataFrame) -> pd.Series:
        """Computes the mean ordinal level per modality."""
        levels = np.asarray(xagg.columns, dtype=float)
        counts = xagg.to_numpy(dtype=float)
        totals = counts.sum(axis=1)
        # empty modalities legitimately yield NaN: silence numpy's divide warnings
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_level = (counts * levels).sum(axis=1) / totals
        return pd.Series(mean_level, index=xagg.index)
