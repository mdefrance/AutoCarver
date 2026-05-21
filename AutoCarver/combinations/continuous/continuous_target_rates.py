"""set of target rates for binary classification"""

from abc import ABC

import numpy as np
import pandas as pd

from AutoCarver.combinations.utils import TargetRate


class ContinuousTargetRate(TargetRate, ABC):
    """Continuous target rate class."""

    __name__ = "continuous_target_rate"

    def compute(self, xagg: pd.Series) -> pd.DataFrame:
        """Computes the target rate.

        Parameters
        ----------
        xagg : pd.DataFrame
            A crosstab.

        Returns
        -------
        Series
            Target rate.
        """
        # checking for an xtab
        if xagg is not None:
            # frequency per modality
            frequency = xagg.apply(len) / xagg.apply(len).sum()

            # computing target rate
            return pd.DataFrame({self.__name__: self._compute(xagg), "frequency": frequency})
        return None

    def compute_from_stats(self, *, stats: dict, index_to_groupby: dict) -> pd.DataFrame | None:
        """Closed-form viability path.

        Subclasses opt in by overriding this method to compute
        ``(target_rate, frequency)`` per group from per-raw-modality
        ``(n, sum_y, …)`` aggregates instead of from the heavy
        ``Series.groupby(...).sum()`` over lists of y values.

        The default returns ``None`` to signal "no closed-form path";
        callers then fall back to the legacy ``_grouper`` + :meth:`compute`
        pair. ``TargetMedian`` keeps this default because the median is not
        decomposable into per-modality sums.
        """
        _ = stats, index_to_groupby  # unused
        return None


class TargetMean(ContinuousTargetRate):
    """Mean target rate class."""

    __name__ = "target_mean"

    def _compute(self, xagg: pd.Series | pd.DataFrame) -> pd.Series:
        """Computes the mean target rate."""
        return xagg.apply(np.mean)

    def compute_from_stats(self, *, stats: dict, index_to_groupby: dict) -> pd.DataFrame | None:
        """Closed-form ``(target_mean, frequency)`` per group from per-modality stats.

        Vectorised counterpart of :meth:`compute` that bypasses the
        ``Series.groupby(...).sum()`` over Python lists of y values. Given
        precomputed per-raw-modality ``(n, sum_y)`` arrays and an
        ``index_to_groupby`` mapping, returns the same DataFrame
        :meth:`compute` would produce — same columns, same row order — but
        with O(k) FLOPs per combination instead of O(N) list concatenation.

        ``stats`` must carry:

        * ``n_per_mod`` (float ndarray, shape ``(n_mod,)``);
        * ``sum_y_per_mod`` (float ndarray, shape ``(n_mod,)``);
        * ``mod_to_pos`` (``{modality: position}`` dict);
        * ``n_mod`` (int).

        Returns ``None`` when ``index_to_groupby`` doesn't cover every raw
        modality (or references a modality the stats don't know about). The
        caller falls back to the legacy ``_grouper`` + ``compute`` path so
        invalid-state fixtures keep their previous behaviour.
        """
        n_per_mod = stats["n_per_mod"]
        sum_y_per_mod = stats["sum_y_per_mod"]
        mod_to_pos = stats["mod_to_pos"]
        n_mod = stats["n_mod"]

        leader_to_grp: dict = {}
        leader_labels: list = []
        assign = np.full(n_mod, -1, dtype=np.intp)
        for mod, leader in index_to_groupby.items():
            pos = mod_to_pos.get(mod)
            if pos is None:
                return None
            gid = leader_to_grp.get(leader)
            if gid is None:
                gid = len(leader_to_grp)
                leader_to_grp[leader] = gid
                leader_labels.append(leader)
            assign[pos] = gid
        if (assign < 0).any():
            return None

        n_groups = len(leader_to_grp)
        n_g = np.bincount(assign, weights=n_per_mod, minlength=n_groups)
        sum_y_g = np.bincount(assign, weights=sum_y_per_mod, minlength=n_groups)
        n_total = float(n_g.sum())

        with np.errstate(invalid="ignore", divide="ignore"):
            mean_g = sum_y_g / n_g
        freq_g = n_g / n_total if n_total > 0 else np.zeros_like(n_g)

        df = pd.DataFrame(
            {self.__name__: mean_g, "frequency": freq_g},
            index=pd.Index(leader_labels),
        )
        # Legacy `xagg.groupby(...).sum()` sorts groups by leader label;
        # mirror that so downstream `_test_distinct_target_rates_between_modalities`
        # (which is order-sensitive) sees the same sequence.
        return df.sort_index()


class TargetMedian(ContinuousTargetRate):
    """Median of target per class."""

    __name__ = "target_median"

    def _compute(self, xagg: pd.Series | pd.DataFrame) -> pd.Series:
        """Computes the mean target rate."""
        return xagg.apply(np.median)


# class TargetVariance(ContinuousTargetRate):
#     """Variance of target per class."""

#     __name__ = "target_variance"

#     def _compute(self, xagg: pd.DataFrame) -> pd.Series:
#         """Computes the mean target rate."""
#         return xagg.apply(var)


# class TargetStd(ContinuousTargetRate):
#     """Std of target per class."""

#     __name__ = "target_std"

#     def _compute(self, xagg: pd.DataFrame) -> pd.Series:
#         """Computes the mean target rate."""
#         return xagg.apply(std)


# class TargetIqr(ContinuousTargetRate):
#     """IQR of target per class."""

#     __name__ = "target_iqr"

#     def _compute(self, xagg: pd.DataFrame) -> pd.Series:
#         """Computes the mean target rate."""
#         return xagg.apply(iqr)


# class TargetRange(ContinuousTargetRate):
#     """Range of target per class."""

#     __name__ = "target_range"

#     def _compute(self, xagg: pd.DataFrame) -> pd.Series:
#         """Computes the mean target rate."""
#         return xagg.apply(lambda x: x.max() - x.min())
