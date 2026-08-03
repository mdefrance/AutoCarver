"""Module for ordinal combination evaluators."""

import math
from abc import ABC

import numpy as np
import pandas as pd

from AutoCarver.combinations.ordinal.ordinal_target_rates import (
    OrdinalTargetRate,
    TargetMeanLevel,
    TargetMeanRidit,
)
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
    _target_rate_classes: list[type[OrdinalTargetRate]] = [TargetMeanRidit, TargetMeanLevel]
    # narrow inherited attribute: ordinal evaluators always carry an OrdinalTargetRate
    # (enforced by _init_target_rate).
    target_rate: OrdinalTargetRate
    # narrow inherited `sort_by: str | None`: concrete subclasses always set a str.
    sort_by: str

    def _init_target_rate(self, target_rate: TargetRate[pd.DataFrame] | None) -> OrdinalTargetRate:
        """Initializes target rate."""
        if target_rate is None:
            return TargetMeanRidit()
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

        # fixing the ridit reference from the raw (un-grouped) train crosstab, before any
        # combination (train or dev) is scored against it (no-op for TargetMeanLevel).
        # Must run before the historize/bail-out below: the inherited NaN path
        # (which may now run after a Gate-1 bail-out) needs the reference fitted.
        self.target_rate.fit_reference(self.samples.train.xagg)  # type: ignore

        self._historize_raw_combination()

        if self.samples.train.shape[0] <= 1:
            return None

        raw_index = list(raw_labels)
        # samples.train.xagg is a crosstab DataFrame for ordinal evaluators
        M, n_per_mod, col_sums = dp_inputs_from_xagg(self.samples.train.xagg, raw_index)  # type: ignore

        # Progressive top-K with ×4 growth (shared driver), mirroring binary/continuous.
        viable = self._search_escalating(
            lambda top_k: _top_k_partitions_ordinal_dp(
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
    total_n = float(n_per_mod.sum())

    # Empty modalities (all-zero rows — e.g. ordinal levels absent from the crosstab,
    # which _dp_inputs_from_xagg zero-fills) carry no observations and must never form
    # their own group: an empty group lowers the non-empty-group count m, which changes
    # tau-c's per-k denominator and breaks the constant-denominator premise that makes the
    # additive-numerator DP exact at top_k=1. So run the DP over the non-empty modalities
    # only, then fold each empty modality back into an adjacent group when emitting.
    keep, kept_M, kept_n_per_mod = compact_empty_modalities(M, n_per_mod)
    n_kept = len(keep)
    cap = min(max_n_mod, n_kept)
    if cap < 2 or total_n < 2:
        return []

    all_pairs = total_n * (total_n - 1) / 2.0
    untied_on_target = all_pairs - float((col_sums * (col_sums - 1) / 2.0).sum())
    c_nonempty = int((col_sums > 0).sum())
    total_between = _concordant_minus_discordant(kept_M)
    seg = _segment_within_costs(kept_M)
    n_prefix = np.concatenate([[0.0], np.cumsum(kept_n_per_mod.astype(float))])

    def seg_cost(i: int, j: int) -> float:
        return float(seg[i, j])

    dp_entries = top_k_partitions(n_mod=n_kept, cap=cap, seg_cost=seg_cost, top_k=top_k, maximize=False)

    entries: list[tuple[float, dict, tuple[int, ...]]] = []
    for _, sum_seg, splits in dp_entries:
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
        entries.append((sort_key(metrics.get(sort_by)), metrics, splits))

    entries.sort(key=lambda e: e[0], reverse=True)
    entries = entries[:top_k]

    out: list[dict] = []
    for _, metrics, splits in entries:
        combination = splits_to_combination(splits, raw_index, keep=keep)
        out.append({"combination": combination, "index_to_groupby": combination_formatter(combination), **metrics})
    return out
