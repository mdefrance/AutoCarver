"""Module for multiclass combination evaluators."""

import math
from abc import ABC

import numpy as np
import pandas as pd

from AutoCarver.combinations.multiclass.multiclass_target_rates import CAScoreRate, MulticlassTargetRate
from AutoCarver.combinations.utils.combination_evaluator import AggregatedSample, CombinationEvaluator
from AutoCarver.combinations.utils.combinations import combination_formatter, group_crosstab
from AutoCarver.combinations.utils.target_rate import TargetRate
from AutoCarver.features import GroupedList


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
        M, n_per_mod, col_sums = _dp_inputs_from_xagg(self.samples.train.xagg, raw_index)  # type: ignore

        # Progressive top-K with doubling, mirroring the binary/ordinal DPs.
        top_k = self.dp_top_k_initial
        walked = 0
        viable: dict | None = None
        while True:
            associations = _top_k_partitions_chi2_dp_multiclass(
                M,
                n_per_mod,
                col_sums,
                max_n_mod=self.max_n_mod,
                raw_index=raw_index,
                sort_by=self.sort_by,
                top_k=top_k,
            )
            viable, walked = self._walk_for_viable(associations, start=walked)
            if viable is not None:
                break
            if walked < top_k:
                break  # DP exhausted every consecutive partition; no viable exists
            top_k *= 2

        if viable is not None and viable.get("xagg") is None:
            index_to_groupby = viable.get("index_to_groupby") or combination_formatter(viable["combination"])
            viable["xagg"] = self._grouper(self.samples.train, index_to_groupby)

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


def _chi2_pearson(obs: np.ndarray) -> float:
    """Pearson :math:`\\chi^2` for a ``(B, K)`` observed contingency table.

    Replicates :func:`scipy.stats.chi2_contingency` defaults: expected
    frequencies via the outer product of marginals divided by N, with Yates
    correction iff the table is exactly 2x2 (matches scipy's own threshold).
    Structurally identical to
    :func:`AutoCarver.combinations.binary.binary_combination_evaluators._chi2_pearson_2col`,
    which is already shape-agnostic beyond that single check; kept as its own
    copy here so the multiclass family doesn't reach into binary's private
    helpers.
    """
    R = obs.sum(axis=1)
    C = obs.sum(axis=0)
    N = float(obs.sum())
    expected = np.outer(R, C) / N

    if obs.shape == (2, 2):
        diff = expected - obs
        direction = np.sign(diff)
        magnitude = np.minimum(0.5, np.abs(diff))
        obs = obs + magnitude * direction

    return float(((obs - expected) ** 2 / expected).sum())


def _cramerv_tschuprowt(chi2: float, n_obs: float, n_groups: int, n_classes: int, tol: float) -> tuple[float, float]:
    """Cramér's V and Tschuprow's T from a chi² computed on an ``(n_groups, n_classes)`` table.

    ``V = sqrt(chi2 / (N * (min(B,K)-1)))``; ``T = sqrt(chi2 / (N *
    sqrt((B-1)(K-1))))``. Both are ``NaN`` when their denominator vanishes
    (mirrors the binary/ordinal ``None``-on-degenerate convention).

    For ``n_classes == 2``, ``T`` is instead derived from the (already
    rounded) ``V`` via ``V / (B-1)**0.25`` — the exact expression
    :func:`AutoCarver.combinations.binary.binary_combination_evaluators._chi2_assoc_for_combination`
    uses. Both formulas are mathematically identical at ``K=2``, but only
    computing it this way guarantees the two evaluators agree bit-for-bit
    (independent ``sqrt``/``pow`` call sequences are not guaranteed to round
    identically) — pinned by the K=2 parity test.
    """
    v_denom = min(n_groups, n_classes) - 1
    if v_denom > 0 and n_obs > 0:
        cramerv = math.sqrt(chi2 / (n_obs * v_denom))
        cramerv = round(cramerv / tol) * tol
    else:
        cramerv = float("nan")

    if n_classes == 2:
        if n_groups > 1:
            tschuprowt = cramerv / math.sqrt(math.sqrt(n_groups - 1))
            if pd.notna(tschuprowt):
                tschuprowt = round(tschuprowt / tol) * tol
        else:
            tschuprowt = cramerv
    else:
        t_denom = math.sqrt((n_groups - 1) * (n_classes - 1)) if n_groups > 1 else 0.0
        if t_denom > 0 and n_obs > 0:
            tschuprowt = math.sqrt(chi2 / (n_obs * t_denom))
            tschuprowt = round(tschuprowt / tol) * tol
        else:
            tschuprowt = float("nan")

    return cramerv, tschuprowt


def _dp_inputs_from_xagg(raw_xagg: pd.DataFrame, raw_index: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aligns a raw crosstab to ``raw_index`` for the DP.

    Returns ``(M, n_per_mod, col_sums)`` where ``M`` is the ``(len(raw_index), K)``
    per-modality column-count matrix (rows absent from ``raw_xagg`` are zero),
    ``n_per_mod`` the row totals and ``col_sums`` the target marginal.
    """
    position = {label: i for i, label in enumerate(raw_xagg.index)}
    values = np.asarray(raw_xagg.values, dtype=float)
    M = np.zeros((len(raw_index), values.shape[1]))
    for row, label in enumerate(raw_index):
        source = position.get(label)
        if source is not None:
            M[row] = values[source]
    return M, M.sum(axis=1), M.sum(axis=0)


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

    n_mod = len(raw_index)
    n_classes = M.shape[1]
    total_n = float(n_per_mod.sum())

    keep = np.flatnonzero(n_per_mod > 0)
    n_kept = len(keep)
    cap = min(max_n_mod, n_kept)
    if cap < 2 or total_n < 2:
        return []

    kept_M = M[keep]
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

        # dp[g][j]: up to ``top_k`` (chi2_partial, splits) with the LARGEST
        # chi2_partial, where splits = (0, s_1, ..., s_{g-1}, j).
        dp: list[list[list[tuple[float, tuple[int, ...]]]]] = [
            [[] for _ in range(n_kept + 1)] for _ in range(k_groups + 1)
        ]
        for j in range(1, n_kept + 1):
            dp[1][j] = [(seg_cost(0, j), (0, j))]
        for g in range(2, k_groups + 1):
            for j in range(g, n_kept + 1):
                candidates: list[tuple[float, tuple[int, ...]]] = []
                for i in range(g - 1, j):
                    c = seg_cost(i, j)
                    for prev_s, prev_splits in dp[g - 1][i]:
                        candidates.append((prev_s + c, prev_splits + (j,)))
                if candidates:
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    dp[g][j] = candidates[:top_k]

        for chi2, splits in dp[k_groups][n_kept]:
            cramerv, tschuprowt = _cramerv_tschuprowt(chi2, total_n, k_groups, n_classes, tol)
            sort_val = tschuprowt if sort_by == "tschuprowt" else cramerv
            all_entries.append((_sort_key(sort_val), cramerv, tschuprowt, splits))

    all_entries.sort(key=lambda e: e[0], reverse=True)
    all_entries = all_entries[:top_k]

    out: list[dict] = []
    for _, cv, tt, splits in all_entries:
        # map compacted cut points back to raw_index: each cut sits just before the first
        # kept modality of the next group, so empty modalities attach to the preceding group
        # (leading empties join the first group, trailing empties the last).
        bounds = [0, *(int(keep[s]) for s in splits[1:-1]), n_mod]
        combination = [list(raw_index[bounds[g] : bounds[g + 1]]) for g in range(len(bounds) - 1)]
        out.append(
            {
                "combination": combination,
                "index_to_groupby": combination_formatter(combination),
                "cramerv": float(cv),
                "tschuprowt": float(tt),
            }
        )
    return out


def _sort_key(value: float | None) -> float:
    """Sort key putting ``None`` / ``NaN`` metrics last (descending sort)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("-inf")
    return float(value)
