"""set of target rates for multiclass (unordered) targets"""

from abc import ABC

import numpy as np
import pandas as pd

from AutoCarver.combinations.utils import TargetRate
from AutoCarver.stats.correspondence_analysis import CAAxis, ca_row_scores, fit_ca_axis


class MulticlassTargetRate(TargetRate[pd.DataFrame], ABC):
    """Multiclass target rate class.

    Operates on a crosstab ``feature-groups (rows) × unordered target classes
    (cols)`` — the same shape :class:`~AutoCarver.combinations.ordinal.ordinal_target_rates.OrdinalTargetRate`
    consumes, generalised to an unordered K-class target. The per-group "rate"
    is a scalar projection of the group's row profile onto a fixed
    correspondence-analysis axis (:class:`CAAxis`, see
    :mod:`AutoCarver.stats.correspondence_analysis`): the owning
    :class:`~AutoCarver.combinations.multiclass.multiclass_combination_evaluators.MulticlassCombinationEvaluator`
    fits that axis once, from the feature's raw (un-grouped) train crosstab
    (:meth:`fit_axis`), and every later call — a train candidate grouping, or a
    dev-sample grouping — projects onto that same fixed axis. That single
    scalar is what lets the existing min-freq / distinct-rates /
    train-dev-rank-order viability machinery (designed around one scalar rate)
    work unchanged for a K-class target.
    """

    __name__ = "multiclass_target_rate"

    def __init__(self) -> None:
        self._axis: CAAxis | None = None

    @property
    def axis(self) -> CAAxis:
        """The fixed correspondence-analysis axis (raises until :meth:`fit_axis` runs)."""
        if self._axis is None:
            raise RuntimeError(f"[{self.__name__}] CA axis is not fit; call fit_axis(raw_xagg) first")
        return self._axis

    def fit_axis(self, raw_xagg: pd.DataFrame) -> None:
        """Fits (and fixes) the CA axis from the feature's raw train crosstab.

        Must be called once per feature, before any candidate grouping is
        scored — every subsequent :meth:`compute` call (train candidate, dev
        candidate) then projects onto this same axis (the CA transition
        formula in :func:`~AutoCarver.stats.correspondence_analysis.ca_row_scores`
        needs only the row's own profile and this fixed axis).
        """
        self._axis = fit_ca_axis(raw_xagg)

    def reference_to_json(self) -> dict | None:
        """Snapshots the fitted CA axis (:class:`CAAxis`) as plain lists."""
        if self._axis is None:
            return None
        return {
            "col_mass": self._axis.col_mass.tolist(),
            "v1": self._axis.v1.tolist(),
            "degenerate": bool(self._axis.degenerate),
        }

    def load_reference(self, payload: dict | None) -> None:
        """Restores the CA axis snapshotted by :meth:`reference_to_json`."""
        if payload is not None:
            self._axis = CAAxis(
                col_mass=np.asarray(payload["col_mass"], dtype=float),
                v1=np.asarray(payload["v1"], dtype=float),
                degenerate=payload["degenerate"],
            )


class CAScoreRate(MulticlassTargetRate):
    """Correspondence-analysis first-axis score per modality.

    The chi²-optimal 1-D embedding of a group's row profile, projected onto
    the fixed train axis (see :class:`MulticlassTargetRate`). Monotone along
    the axis by construction, so it drives both the ``min_freq`` viability
    test and the train/dev rank-preservation veto exactly like the
    binary/ordinal target rates.
    """

    __name__ = "ca_score"

    def _compute(self, xagg: pd.DataFrame) -> pd.Series:
        """Computes the CA first-axis score per modality."""
        return ca_row_scores(xagg, self.axis)
