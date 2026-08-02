"""set of target rates for ordinal targets"""

from abc import ABC

import numpy as np
import pandas as pd

from AutoCarver.combinations.utils import TargetRate
from AutoCarver.discretizers.utils.ridits import ridit_scores_for_levels


class OrdinalTargetRate(TargetRate[pd.DataFrame], ABC):
    """Ordinal target rate class.

    Operates on an ordered contingency table ``feature-groups (rows) ×
    ordinal-target-levels (cols)`` — the same crosstab shape the binary target
    rates consume, only with one column per ordinal level instead of two.
    """

    __name__ = "ordinal_target_rate"

    def fit_reference(self, raw_xagg: pd.DataFrame) -> None:
        """No-op hook fixing a train reference before any candidate is scored.

        Rates needing one (:class:`TargetMeanRidit`) override it; mirrors
        :meth:`~AutoCarver.combinations.multiclass.multiclass_target_rates.MulticlassTargetRate.fit_axis`.
        """


class TargetMeanRidit(OrdinalTargetRate):
    """Mean train-ridit per modality (the ordinal default).

    The per-group rate is the count-weighted mean of the **train** ridits of
    the crosstab's columns (see :mod:`AutoCarver.discretizers.utils.ridits`):
    the owning evaluator fixes the reference marginal once, from the feature's
    raw (un-grouped) train crosstab (:meth:`fit_reference`), and every later
    call — a train candidate grouping, or a dev-sample grouping — scores
    against that same reference (levels unseen in train get the natural CDF
    extension). Invariant under any strictly increasing re-encoding of the
    target levels, bounded in ``[0, 1]``, and monotone in the target's order,
    so it drives both the ``min_freq`` viability test and the train/dev
    rank-preservation veto exactly like the binary/continuous target means.
    """

    __name__ = "target_mean_ridit"

    def __init__(self) -> None:
        self._reference: pd.Series | None = None

    @property
    def reference(self) -> pd.Series:
        """The fixed train count-marginal (raises until :meth:`fit_reference` runs)."""
        if self._reference is None:
            raise RuntimeError(f"[{self.__name__}] reference is not fit; call fit_reference(raw_xagg) first")
        return self._reference

    def fit_reference(self, raw_xagg: pd.DataFrame) -> None:
        """Fits (and fixes) the reference marginal from the feature's raw train crosstab.

        Must be called once per feature, before any candidate grouping is
        scored — every subsequent :meth:`compute` call (train candidate, dev
        candidate) then scores its columns' ridits against this marginal.
        """
        self._reference = raw_xagg.sum(axis=0)

    def reference_to_json(self) -> dict | None:
        """Snapshots the fitted reference marginal as parallel lists.

        Levels can be integers, which JSON object keys would stringify — hence
        two lists rather than a ``{level: count}`` mapping.
        """
        if self._reference is None:
            return None
        # levels come off a crosstab's columns: numpy scalars, which json.dump rejects
        levels = [level.item() if hasattr(level, "item") else level for level in self._reference.index]
        return {"levels": levels, "counts": [float(count) for count in self._reference]}

    def load_reference(self, payload: dict | None) -> None:
        """Restores the reference marginal snapshotted by :meth:`reference_to_json`."""
        if payload is not None:
            self._reference = pd.Series(payload["counts"], index=payload["levels"])

    def _compute(self, xagg: pd.DataFrame) -> pd.Series:
        """Computes the mean train-ridit per modality."""
        ridits = ridit_scores_for_levels(xagg.columns, self.reference)
        counts = xagg.to_numpy(dtype=float)
        totals = counts.sum(axis=1)
        # empty modalities legitimately yield NaN: silence numpy's divide warnings
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_ridit = (counts * ridits).sum(axis=1) / totals
        return pd.Series(mean_ridit, index=xagg.index)


class TargetMeanLevel(OrdinalTargetRate):
    """Mean ordinal level per modality.

    The per-group rate is ``Σ_j level_j · n_gj / n_g+`` where ``level_j`` is read
    from the (integer) crosstab columns — or, when ``level_values`` is given,
    from the user's per-level representative values. It is monotone in the
    target's order, so it drives both the ``min_freq`` viability test and the
    train/dev rank-preservation veto exactly like the binary/continuous target
    means.
    """

    __name__ = "target_mean_level"

    def __init__(self, level_values: dict | None = None) -> None:
        """
        Parameters
        ----------
        level_values : dict, optional
            ``{level: value}`` representative value per target level (e.g. a
            calibrated default probability per rating grade). Values must be
            strictly increasing when levels are sorted ascending. ``None``
            (default) reads the levels themselves from the crosstab columns.
        """
        if level_values is not None:
            values = [level_values[level] for level in sorted(level_values)]
            if any(nxt <= prev for prev, nxt in zip(values, values[1:])):
                raise ValueError(
                    f"[{self.__name__}] level_values must be strictly increasing in the level order, got {level_values}"
                )
        self.level_values = dict(level_values) if level_values is not None else None

    def _compute(self, xagg: pd.DataFrame) -> pd.Series:
        """Computes the mean ordinal level per modality."""
        if self.level_values is not None:
            missing = [column for column in xagg.columns if column not in self.level_values]
            if len(missing) > 0:
                raise ValueError(f"[{self.__name__}] levels {missing} are missing from level_values")
            levels = np.array([self.level_values[column] for column in xagg.columns], dtype=float)
        else:
            levels = np.asarray(xagg.columns, dtype=float)
        counts = xagg.to_numpy(dtype=float)
        totals = counts.sum(axis=1)
        # empty modalities legitimately yield NaN: silence numpy's divide warnings
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_level = (counts * levels).sum(axis=1) / totals
        return pd.Series(mean_level, index=xagg.index)
