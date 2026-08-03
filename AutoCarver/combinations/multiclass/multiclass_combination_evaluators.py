"""Module for multiclass combination evaluators."""

from abc import ABC

import numpy as np
import pandas as pd

from AutoCarver.combinations.multiclass.multiclass_target_rates import CAScoreRate, MulticlassTargetRate
from AutoCarver.combinations.utils.combination_evaluator import AggregatedSample, CombinationEvaluator
from AutoCarver.combinations.utils.combinations import combination_formatter, group_crosstab
from AutoCarver.combinations.utils.dp import (
    compact_empty_modalities,
    dp_inputs_from_xagg,
    sort_key,
    splits_to_combination,
    top_k_partitions,
)
from AutoCarver.combinations.utils.target_rate import TargetRate
from AutoCarver.features import GroupedList
from AutoCarver.stats.chi2 import cramerv_tschuprowt as _cramerv_tschuprowt
from AutoCarver.stats.chi2 import pearson_chi2 as _chi2_pearson


class MulticlassCombinationEvaluator(CombinationEvaluator[pd.DataFrame], ABC):
    """Multiclass combination evaluator class.

    The aggregation is a crosstab ``feature-groups (rows) × unordered
    target classes (cols)`` — the same shape
    :class:`~AutoCarver.combinations.ordinal.ordinal_combination_evaluators.OrdinalCombinationEvaluator`
    consumes, generalising the binary evaluator's 2-column Pearson chi² to K
    columns. Concrete subclasses pick which chi²-family statistic ranks
    combinations via :attr:`sort_by`:

    * :ref:`tschuprowt` — Tschuprow's T (**default**), whose ``sqrt((B-1)(K-1))``
      correction stays comparable across combinations with different group
      counts;
    * :ref:`cramerv` — Cramér's V.

    For ``K == 2`` both statistics — and the DP search — are numerically
    identical to :class:`~AutoCarver.combinations.binary.binary_combination_evaluators.BinaryCombinationEvaluator`,
    pinned bit-for-bit by a parity test.

    Search uses the same :ref:`progressive top-K interval DP <DPChi2>` as the
    binary path, generalised from a ``(2,)`` per-segment observed vector to a
    ``(K,)`` one.
    """

    is_y_multiclass = True
    _target_rate_classes: list[type[MulticlassTargetRate]] = [CAScoreRate]
    # narrow inherited attribute: multiclass evaluators always carry a MulticlassTargetRate
    # (enforced by _init_target_rate).
    target_rate: MulticlassTargetRate
    # narrow inherited `sort_by: str | None`: concrete subclasses always set a str.
    sort_by: str

    def _init_target_rate(self, target_rate: TargetRate[pd.DataFrame] | None) -> MulticlassTargetRate:
        """Initializes target rate."""
        if target_rate is None:
            return CAScoreRate()
        if not isinstance(target_rate, MulticlassTargetRate):
            raise ValueError("target_rate must be a MulticlassTargetRate")
        return target_rate

    def _grouper(self, xagg: AggregatedSample, groupby: dict) -> pd.DataFrame:
        """Groups a crosstab by ``groupby`` and sums column values by group.

        Shares :func:`group_crosstab` with the binary/ordinal paths: leaders
        are ordered by first appearance so grouping stays independent of
        label text.
        """
        return group_crosstab(xagg, groupby)

    def _association_measure(
        self,
        xagg: AggregatedSample | pd.Series | pd.DataFrame,
        n_obs: int | None = None,
        tol: float = 1e-10,
    ) -> dict[str, float | None]:
        """Computes Cramér's V and Tschuprow's T for a (groups x classes) crosstab.

        Used for the raw (one-shot) distribution and the NaN-fanout scoring
        path. The hot per-combination loop goes through
        :meth:`_get_best_combination_non_nan`'s DP, which evaluates the same
        closed-form chi² directly from prefix-summed per-modality counts so
        the per-modality crosstab does not have to be rebuilt on every
        combination.
        """
        values = np.asarray(xagg.values, dtype=float)
        n_groups, n_classes = values.shape
        chi2 = _chi2_pearson(values + tol)
        total = float(values.sum()) if n_obs is None else float(n_obs)
        cramerv, tschuprowt = _cramerv_tschuprowt(chi2, total, n_groups, n_classes, tol)
        return {"cramerv": cramerv, "tschuprowt": tschuprowt}

    def _get_best_combination_non_nan(self) -> dict | None:
        """DP-based override with progressive top-K (mirrors the ordinal/binary paths).

        Fits (and fixes) the CA axis from the feature's raw train crosstab
        *before* any candidate is scored (see :meth:`MulticlassTargetRate.fit_axis`)
        — every later viability check (train candidate, dev candidate) then
        projects onto that same axis.

        The NaN path (:meth:`_get_best_combination_with_nan`) is **not**
        overridden, mirroring :class:`OrdinalCombinationEvaluator`: it runs
        after this method has applied the best non-NaN grouping, so it
        enumerates over the already-small grouped label set and the
        inherited enumerate-and-score path is cheap there.
        """
        feature_labels = self.feature.labels
        if feature_labels is None:
            raise RuntimeError(f"[{self.__name__}] feature labels are not populated")
        raw_labels = GroupedList(feature_labels[:])

        if self.feature.has_nan:
            if self.feature.dropna:
                raw_labels.remove(self.feature.nan)
            self.samples.dropna(self.feature.nan)

        # fixing the CA axis from the raw (un-grouped) train crosstab, before any
        # combination (train or dev) is scored against it. Must run before the
        # historize/bail-out below: the inherited NaN path (which may now run
        # after a Gate-1 bail-out) needs the axis fitted.
        self.target_rate.fit_axis(self.samples.train.xagg)  # type: ignore

        self._historize_raw_combination()

        if self.samples.train.shape[0] <= 1:
            return None

        raw_index = list(raw_labels)
        # samples.train.xagg is a crosstab DataFrame for multiclass evaluators
        M, n_per_mod, col_sums = dp_inputs_from_xagg(self.samples.train.xagg, raw_index)  # type: ignore

        # Progressive top-K with ×4 growth (shared driver), mirroring binary/ordinal.
        viable = self._search_escalating(
            lambda top_k: _top_k_partitions_chi2_dp_multiclass(
                M,
                n_per_mod,
                col_sums,
                max_n_mod=self.max_n_mod,
                raw_index=raw_index,
                sort_by=self.sort_by,
                top_k=top_k,
            )
        )
        self._rebuild_winner_xagg(viable)

        self._apply_best_combination(viable)
        return viable


class TschuprowtMulticlassCombinations(MulticlassCombinationEvaluator):
    """Tschuprow's T based combination evaluation toolkit (multiclass default).

    Search uses :ref:`progressive top-K interval DP <DPChi2>` over the
    closed-form Pearson :math:`\\chi^2` decomposition, generalised to a
    ``(B, K)`` table. Statistically equivalent to
    :func:`scipy.stats.chi2_contingency` — bit-exact agreement pinned by
    parity tests; for a 2-class target, numerically identical to
    :class:`~AutoCarver.combinations.binary.binary_combination_evaluators.TschuprowtCombinations`.
    """

    sort_by = "tschuprowt"


class CramervMulticlassCombinations(MulticlassCombinationEvaluator):
    """Cramér's V based combination evaluation toolkit.

    Same DP search as :class:`TschuprowtMulticlassCombinations` (see
    :ref:`DPChi2`); only the ``sort_by`` key differs.
    """

    sort_by = "cramerv"


# ---------------------------------------------------------------------------
# Closed-form chi^2 helpers (K-column contingency tables)
# ---------------------------------------------------------------------------
# _chi2_pearson and _cramerv_tschuprowt are shared with the binary family via
# AutoCarver.stats.chi2 (imported above).


def _top_k_partitions_chi2_dp_multiclass(  # noqa: C901
    M: np.ndarray,
    n_per_mod: np.ndarray,
    col_sums: np.ndarray,
    *,
    max_n_mod: int,
    raw_index: list,
    sort_by: str,
    top_k: int,
    tol: float = 1e-10,
) -> list[dict]:
    """Top-K consecutive partitions ranked by ``sort_by`` (cramerv / tschuprowt).

    Multiclass analogue of
    :func:`AutoCarver.combinations.binary.binary_combination_evaluators._top_k_partitions_chi2_dp`,
    generalised from a ``(2,)`` per-segment observed vector to ``(K,)``: the
    per-segment chi² contribution is additive across groups **given a fixed
    number of groups k** (the column marginals and total depend only on
    ``k``, not on the split positions), so a separate interval-DP per
    ``k in [2, K]`` still applies.

    Empty raw modalities (all-zero rows) carry no observations and must never
    form their own group — same hazard the ordinal DP hit once (an empty
    group changes the effective ``B``, silently shifting which candidates are
    even comparable at a given ``k``). Run the DP over the non-empty
    modalities only, then fold each empty modality back into an adjacent
    group when emitting (leading empties join the first group, trailing
    empties the last).
    """
    if sort_by not in ("cramerv", "tschuprowt"):
        raise ValueError(f"sort_by must be 'cramerv' or 'tschuprowt', got {sort_by!r}")

    n_classes = M.shape[1]
    total_n = float(n_per_mod.sum())

    keep, kept_M, _ = compact_empty_modalities(M, n_per_mod)
    n_kept = len(keep)
    cap = min(max_n_mod, n_kept)
    if cap < 2 or total_n < 2:
        return []

    col_totals = col_sums.astype(np.float64)
    prefix = np.concatenate([np.zeros((1, n_classes)), np.cumsum(kept_M, axis=0)], axis=0)

    all_entries: list[tuple[float, float, float, tuple[int, ...]]] = []

    for k_groups in range(2, cap + 1):
        C = col_totals + k_groups * tol
        N_with_tol = total_n + k_groups * n_classes * tol
        yates = k_groups == 2 and n_classes == 2

        def seg_cost(i: int, j: int, _C: np.ndarray = C, _N: float = N_with_tol, _yates: bool = yates) -> float:
            obs = (prefix[j] - prefix[i]) + tol
            R = float(obs.sum())
            E = R * _C / _N
            if _yates:
                diff = E - obs
                shift = np.minimum(0.5, np.abs(diff)) * np.sign(diff)
                obs = obs + shift
            return float(((obs - E) ** 2 / E).sum())

        # Only the final row (k == k_groups) is used: rebuilding the DP per k_groups
        # is required because C/N/yates depend on k_groups (see docstring).
        entries = top_k_partitions(n_mod=n_kept, cap=k_groups, seg_cost=seg_cost, top_k=top_k, maximize=True)

        for k, chi2, splits in entries:
            if k != k_groups:
                continue
            cramerv, tschuprowt = _cramerv_tschuprowt(chi2, total_n, k_groups, n_classes, tol)
            sort_val = tschuprowt if sort_by == "tschuprowt" else cramerv
            all_entries.append((sort_key(sort_val), cramerv, tschuprowt, splits))

    all_entries.sort(key=lambda e: e[0], reverse=True)
    all_entries = all_entries[:top_k]

    out: list[dict] = []
    for _, cv, tt, splits in all_entries:
        combination = splits_to_combination(splits, raw_index, keep=keep)
        out.append(
            {
                "combination": combination,
                "index_to_groupby": combination_formatter(combination),
                "cramerv": float(cv),
                "tschuprowt": float(tt),
            }
        )
    return out
