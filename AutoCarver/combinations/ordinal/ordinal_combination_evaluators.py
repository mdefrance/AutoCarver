"""Module for ordinal combination evaluators."""

import math
from abc import ABC

import numpy as np
import pandas as pd

from AutoCarver.combinations.ordinal.ordinal_target_rates import OrdinalTargetRate, TargetMeanLevel
from AutoCarver.combinations.utils.combination_evaluator import AggregatedSample, CombinationEvaluator
from AutoCarver.combinations.utils.combinations import combination_formatter, group_crosstab
from AutoCarver.combinations.utils.target_rate import TargetRate
from AutoCarver.features import GroupedList


class OrdinalCombinationEvaluator(CombinationEvaluator[pd.DataFrame], ABC):
    """Ordinal combination evaluator class.

    The aggregation is an ordered contingency table
    ``feature-groups (rows, in target-rate order) × ordinal-target-levels
    (cols, ascending)`` — the binary crosstab generalised from 2 columns to as
    many columns as the target has levels. Three rank-association statistics are
    computed per combination; concrete subclasses pick which one ranks
    combinations via :attr:`sort_by`:

    * :ref:`tau_c` — Kendall/Stuart's tau-c (**default**, rectangular-table
      correction; self-balances to a robust, meaningful number of modalities);
    * :ref:`tau_b` — Kendall's tau-b (matches :func:`scipy.stats.kendalltau`);
    * :ref:`somersd` — the original asymmetric Somers' D ``D(Y|X)`` (target given
      feature).

    The symmetric Kendall taus reward a split only when it is genuinely
    discriminative and otherwise favour fewer, more robust modalities — like
    :class:`TschuprowtCombinations` and the Kruskal effect sizes. Somers' D is
    asymmetric and leans strongly toward the coarsest split.

    Search uses the inherited enumerate-and-score path.
    """

    is_y_ordinal = True
    _target_rate_classes: list[type[OrdinalTargetRate]] = [TargetMeanLevel]
    # narrow inherited attribute: ordinal evaluators always carry an OrdinalTargetRate
    # (enforced by _init_target_rate).
    target_rate: OrdinalTargetRate
    # narrow inherited `sort_by: str | None`: concrete subclasses always set a str.
    sort_by: str

    def _init_target_rate(self, target_rate: TargetRate[pd.DataFrame] | None) -> OrdinalTargetRate:
        """Initializes target rate."""
        if target_rate is None:
            return TargetMeanLevel()
        if not isinstance(target_rate, OrdinalTargetRate):
            raise ValueError("target_rate must be an OrdinalTargetRate")
        return target_rate

    def _grouper(self, xagg: AggregatedSample, groupby: dict) -> pd.DataFrame:
        """Groups a crosstab by ``groupby`` and sums column values by group.

        Shares :func:`group_crosstab` with the binary path: leaders are ordered
        by first appearance so grouping stays independent of label text.
        """
        return group_crosstab(xagg, groupby)

    def _association_measure(
        self,
        xagg: AggregatedSample | pd.Series | pd.DataFrame,
        n_obs: int | None = None,
        tol: float = 1e-10,
    ) -> dict[str, float | None]:
        """Computes Kendall's tau-b, tau-c and Somers' D between feature and target.

        Parameters
        ----------
        xagg : pd.DataFrame
            Ordered contingency table (rows = feature groups, cols = ordinal
            target levels). ``n_obs`` / ``tol`` are unused (the rank statistics
            only depend on the table's cell counts).

        Returns
        -------
        dict[str, float | None]
            ``{"tau_b": ..., "tau_c": ..., "somersd": ...}``; any may be
            ``None`` for a degenerate table.
        """
        _, _ = n_obs, tol  # unused
        return _ordinal_associations(np.asarray(xagg.values, dtype=float))

    def _get_best_combination_non_nan(self) -> dict | None:
        """DP-based override with progressive top-K (mirrors the continuous path).

        Replaces ``consecutive_combinations`` + enumerate-and-score with the
        interval-DP in :func:`_top_k_partitions_ordinal_dp` over the additively
        decomposable ``C−D`` numerator. Exact for tau-c (per-k constant
        denominator); a progressively-grown top-K approximation for tau-b /
        Somers' D, whose denominators depend on the group sizes.

        The NaN path (:meth:`_get_best_combination_with_nan`) is **not** overridden:
        it runs after this method has applied the best non-NaN grouping, so it
        enumerates over the already-small grouped label set and the inherited
        enumerate-and-score path is cheap there.
        """
        feature_labels = self.feature.labels
        if feature_labels is None:
            raise RuntimeError(f"[{self.__name__}] feature labels are not populated")
        raw_labels = GroupedList(feature_labels[:])

        if self.feature.has_nan:
            if self.feature.dropna:
                raw_labels.remove(self.feature.nan)
            self.samples.dropna(self.feature.nan)

        if self.samples.train.shape[0] <= 1:
            return None

        self._historize_raw_combination()

        raw_index = list(raw_labels)
        # samples.train.xagg is a crosstab DataFrame for ordinal evaluators
        M, n_per_mod, col_sums = _dp_inputs_from_xagg(self.samples.train.xagg, raw_index)  # type: ignore

        # Progressive top-K with doubling, mirroring the binary/continuous DPs.
        top_k = self.dp_top_k_initial
        walked = 0
        viable: dict | None = None
        while True:
            associations = _top_k_partitions_ordinal_dp(
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


class KendallTauCCombinations(OrdinalCombinationEvaluator):
    """Kendall's tau-c based combination evaluation toolkit (ordinal default).

    Stuart's tau-c applies a ``min(r, c)`` correction tailored to **rectangular**
    tables — exactly our shape (few feature groups × many target levels) — so its
    magnitude stays comparable across combinations with different group counts and
    it leans toward fewer, robust modalities, only adding one when a split is
    genuinely meaningful.
    """

    sort_by = "tau_c"


class KendallTauBCombinations(OrdinalCombinationEvaluator):
    """Kendall's tau-b based combination evaluation toolkit.

    Bit-exact with :func:`scipy.stats.kendalltau` (the ``tau-b`` variant) on the
    grouped contingency table — pinned by parity tests. Normalised by the
    geometric mean of both margins' untied pairs; tends to retain more modalities
    on smoothly monotone signals than :class:`KendallTauCCombinations`.
    """

    sort_by = "tau_b"


class SomersDCombinations(OrdinalCombinationEvaluator):
    """Somers' D based combination evaluation toolkit.

    The original asymmetric Somers' D ``D(Y|X)`` — concordant minus discordant
    pairs over pairs untied on the feature ``X`` — matching
    ``scipy.stats.somersd(table).statistic``. Being asymmetric it leans strongly
    toward the coarsest split (its maximum over groupings is typically two
    modalities); offered for users who specifically want raw Somers' D rather
    than the self-balancing Kendall taus.
    """

    sort_by = "somersd"


def _concordant_minus_discordant(values: np.ndarray) -> float:
    """Concordant minus discordant pairs ``C − D`` of an ordered table.

    ``values`` is ``(r, c)`` with rows / columns already ascending.
    """
    # concordant partners of each cell: counts strictly down-right (k>i, l>j)
    suffix = np.cumsum(np.cumsum(values[::-1, ::-1], axis=0), axis=1)[::-1, ::-1]
    down_right = np.zeros_like(values)
    down_right[:-1, :-1] = suffix[1:, 1:]

    # discordant partners of each cell: counts strictly down-left (k>i, l<j)
    suffix_rows_prefix_cols = np.cumsum(np.cumsum(values[::-1, :], axis=0)[::-1, :], axis=1)
    down_left = np.zeros_like(values)
    down_left[:-1, 1:] = suffix_rows_prefix_cols[1:, :-1]

    return float((values * down_right).sum()) - float((values * down_left).sum())


def _taus_from_counts(
    cd: float, n: float, untied_on_feature: float, untied_on_target: float, m: int
) -> dict[str, float | None]:
    """Assembles tau-b, tau-c and Somers' D from pre-computed pair counts.

    Shared by the closed form (:func:`_ordinal_associations`) and the DP path
    so both produce bit-identical values.

    * ``tau_b = (C − D) / sqrt((P0 − T_X)(P0 − T_Y))`` — matches
      ``scipy.stats.kendalltau``;
    * ``tau_c = 2·m·(C − D) / (n²·(m − 1))`` (Stuart's rectangular-table
      correction);
    * ``somersd = (C − D) / (P0 − T_X)`` — the original Somers' D ``D(Y|X)``.

    Each measure is ``None`` when its denominator vanishes.
    """
    denominator_b = math.sqrt(untied_on_feature * untied_on_target)
    return {
        "tau_b": cd / denominator_b if denominator_b > 0 else None,
        "tau_c": (2.0 * m * cd) / (n * n * (m - 1)) if m > 1 else None,
        "somersd": cd / untied_on_feature if untied_on_feature > 0 else None,
    }


def _ordinal_associations(values: np.ndarray) -> dict[str, float | None]:
    """Kendall's tau-b, tau-c and Somers' D ``D(Y|X)`` for an ordered table.

    ``values`` is the ``(r, c)`` cell-count array with rows = ``X`` (feature
    groups) and columns = ``Y`` (target levels), both already in ascending order.
    Each measure is ``None`` when its denominator vanishes (degenerate table),
    mirroring the continuous evaluator's ``None`` convention.
    """
    n = float(values.sum())
    if n < 2:
        return {"tau_b": None, "tau_c": None, "somersd": None}

    cd = _concordant_minus_discordant(values)
    row = values.sum(axis=1)
    col = values.sum(axis=0)
    all_pairs = n * (n - 1) / 2.0
    untied_on_feature = all_pairs - float((row * (row - 1) / 2.0).sum())
    untied_on_target = all_pairs - float((col * (col - 1) / 2.0).sum())
    m = min(int((row > 0).sum()), int((col > 0).sum()))
    return _taus_from_counts(cd, n, untied_on_feature, untied_on_target, m)


# ---------------------------------------------------------------------------
# Phase-B: progressive top-K interval DP over the additive C−D numerator
# ---------------------------------------------------------------------------
#
# C−D of a consecutive grouping decomposes additively:
#
#     C−D(grouping) = TotalBetween − Σ_g WithinSegment(g)
#
# where TotalBetween (the C−D of the fully-split table) is constant and
# WithinSegment is prefix-summable. So an interval DP that keeps, per number of
# groups k, the partitions with the largest numerator (smallest Σ WithinSegment)
# enumerates the best candidates without materialising every consecutive
# partition. For tau-c the per-k denominator is constant, so numerator-optimal
# == metric-optimal (the DP is exact). For tau-b / Somers' D the denominator
# depends on the group sizes (T_X), so the kept top-K candidates are re-scored
# with their true denominators and ranked — exact when top_k is exhaustive, a
# top-K approximation otherwise.


def _dp_inputs_from_xagg(raw_xagg: pd.DataFrame, raw_index: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aligns a raw crosstab to ``raw_index`` for the DP.

    Returns ``(M, n_per_mod, col_sums)`` where ``M`` is the ``(len(raw_index), c)``
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


def _segment_within_costs(M: np.ndarray) -> np.ndarray:
    """WithinSegment ``C−D`` for every consecutive row segment.

    Returns ``seg`` of shape ``(n_mod, n_mod + 1)`` where ``seg[a, b]`` is the
    concordant−discordant count among observations whose modalities both lie in
    the consecutive block ``[a, b)`` — the within-segment pairs that grouping
    removes from ``C−D``. Computed in ``O(n_mod² · c)`` via the bilinearity of
    the between-modality concordance in the two rows' column vectors.
    """
    n_mod, c = M.shape
    seg = np.zeros((n_mod, n_mod + 1))
    for a in range(n_mod):
        block = M[a].astype(float).copy()
        within = 0.0
        for b in range(a + 1, n_mod):
            # between(block=[a,b) lower rows, row b higher): bilinear in column vectors
            inclusive = np.cumsum(block)
            strictly_lower = inclusive - block  # target mass below each column, within the block
            strictly_higher = block.sum() - inclusive  # target mass above each column, within the block
            within += float((M[b] * (strictly_lower - strictly_higher)).sum())
            block = block + M[b]
            seg[a, b + 1] = within
    return seg


def _build_partition_dp(
    seg: np.ndarray, *, n_mod: int, cap: int, top_k: int
) -> list[list[list[tuple[float, tuple[int, ...]]]]]:
    """dp[g][j]: up to ``top_k`` ``(sum_seg, splits)`` with the SMALLEST sum_seg
    (largest numerator), where ``splits = (0, s_1, ..., s_{g-1}, j)`` and ``g`` is
    the number of groups.
    """
    dp: list[list[list[tuple[float, tuple[int, ...]]]]] = [[[] for _ in range(n_mod + 1)] for _ in range(cap + 1)]
    for j in range(1, n_mod + 1):
        dp[1][j] = [(float(seg[0, j]), (0, j))]
    for g in range(2, cap + 1):
        for j in range(g, n_mod + 1):
            candidates: list[tuple[float, tuple[int, ...]]] = []
            for i in range(g - 1, j):
                seg_ij = float(seg[i, j])
                for prev_sum, prev_splits in dp[g - 1][i]:
                    candidates.append((prev_sum + seg_ij, prev_splits + (j,)))
            if candidates:
                candidates.sort(key=lambda x: x[0])  # smallest Σ within-segment first
                dp[g][j] = candidates[:top_k]
    return dp


def _score_partition(
    sum_seg: float,
    splits: tuple[int, ...],
    *,
    total_between: float,
    n_prefix: np.ndarray,
    total_n: float,
    all_pairs: float,
    untied_on_target: float,
    c_nonempty: int,
) -> dict:
    """Compute tau_b / tau_c / somersd for a single consecutive partition."""
    cd = total_between - sum_seg
    tied_on_feature = 0.0
    non_empty_groups = 0
    for g in range(len(splits) - 1):
        size = n_prefix[splits[g + 1]] - n_prefix[splits[g]]
        tied_on_feature += size * (size - 1) / 2.0
        if size > 0:
            non_empty_groups += 1
    # m matches the closed form: min over *non-empty* grouped rows and target levels
    m = min(non_empty_groups, c_nonempty)
    return _taus_from_counts(cd, total_n, all_pairs - tied_on_feature, untied_on_target, m)


def _top_k_partitions_ordinal_dp(
    M: np.ndarray,
    n_per_mod: np.ndarray,
    col_sums: np.ndarray,
    *,
    max_n_mod: int,
    raw_index: list,
    sort_by: str,
    top_k: int,
) -> list[dict]:
    """Top-K consecutive partitions ranked by ``sort_by`` (tau_b / tau_c / somersd).

    ``M`` is the ``(n_mod, c)`` per-modality column-count matrix aligned to
    ``raw_index``; ``col_sums`` is the target marginal (for ``T_Y``). Returns a
    list of ``{combination, index_to_groupby, tau_b, tau_c, somersd}`` dicts
    sorted by ``sort_by`` desc — same shape the streaming pipeline yields, so it
    drops into the viability walk.
    """
    n_mod = len(raw_index)
    cap = min(max_n_mod, n_mod)
    total_n = float(n_per_mod.sum())
    if cap < 2 or total_n < 2:
        return []

    all_pairs = total_n * (total_n - 1) / 2.0
    untied_on_target = all_pairs - float((col_sums * (col_sums - 1) / 2.0).sum())
    c_nonempty = int((col_sums > 0).sum())
    total_between = _concordant_minus_discordant(M)
    seg = _segment_within_costs(M)
    n_prefix = np.concatenate([[0.0], np.cumsum(n_per_mod.astype(float))])

    dp = _build_partition_dp(seg, n_mod=n_mod, cap=cap, top_k=top_k)

    entries: list[tuple[float, dict, tuple[int, ...]]] = []
    for k in range(2, cap + 1):
        for sum_seg, splits in dp[k][n_mod]:
            metrics = _score_partition(
                sum_seg,
                splits,
                total_between=total_between,
                n_prefix=n_prefix,
                total_n=total_n,
                all_pairs=all_pairs,
                untied_on_target=untied_on_target,
                c_nonempty=c_nonempty,
            )
            entries.append((_sort_key(metrics.get(sort_by)), metrics, splits))

    entries.sort(key=lambda e: e[0], reverse=True)
    entries = entries[:top_k]

    out: list[dict] = []
    for _, metrics, splits in entries:
        combination = [list(raw_index[splits[g] : splits[g + 1]]) for g in range(len(splits) - 1)]
        out.append({"combination": combination, "index_to_groupby": combination_formatter(combination), **metrics})
    return out


def _sort_key(value: float | None) -> float:
    """Sort key putting ``None`` / ``NaN`` metrics last (descending sort)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("-inf")
    return float(value)
