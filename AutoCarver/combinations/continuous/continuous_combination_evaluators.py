"""Module for continuous combination evaluators."""

from abc import ABC
from collections.abc import Iterable, Iterator

import numpy as np
import pandas as pd
from scipy.stats import kruskal, rankdata, tiecorrect
from tqdm import tqdm

from AutoCarver.combinations.continuous.continuous_target_rates import ContinuousTargetRate, TargetMean, TargetMedian
from AutoCarver.combinations.utils.combination_evaluator import AggregatedSample, CombinationEvaluator
from AutoCarver.combinations.utils.combinations import combination_formatter


class ContinuousCombinationEvaluator(CombinationEvaluator, ABC):
    """Continuous combination evaluator class."""

    is_y_continuous = True
    _target_rate_classes: list[type[ContinuousTargetRate]] = [TargetMean, TargetMedian]

    def _init_target_rate(self, target_rate: ContinuousTargetRate | None) -> ContinuousTargetRate:
        """Initializes target rate."""
        if target_rate is None:
            return TargetMean()
        elif not isinstance(target_rate, ContinuousTargetRate):
            raise ValueError("target_rate must be a ContinuousTargetRate")
        return target_rate

    def _association_measure(
        self, xagg: AggregatedSample, n_obs: int | None = None, tol: float = 1e-10
    ) -> dict[str, float | None]:
        """Computes measures of association between feature and quantitative target.

        Used for the raw (one-shot) distribution. The hot per-combination loop
        goes through :meth:`_compute_associations` which evaluates the same
        Kruskal–Wallis H statistic in closed form without re-ranking — see
        :func:`_modality_rank_stats` and :func:`_kruskal_h_from_modality_stats`.

        Parameters
        ----------
        xagg : pd.DataFrame
            Values taken by y for each of x's modalities.

        Returns
        -------
        dict[str, float]
            Kruskal-Wallis' H as a dict.
        """
        _, _ = n_obs, tol  # unused attribute

        # Kruskal-Wallis' H
        try:
            return {"kruskal": kruskal(*tuple(xagg.values))[0]}
        except (ValueError, IndexError):
            return {"kruskal": None}

    def _grouper(self, xagg: AggregatedSample, groupby: dict[str, str]) -> pd.Series:
        """Groups values of y

        Parameters
        ----------
        yval : pd.Series
            _description_
        groupby : _type_
            _description_

        Returns
        -------
        Series
            _description_
        """
        # NOTE: kept as list-concatenating groupby.sum() for compatibility with
        # downstream consumers (target rates, viability tests, public API tests
        # that pin the Series-of-lists shape). The Kruskal-Wallis hot loop no
        # longer goes through this path — see _compute_associations below.
        return xagg.groupby(groupby).sum()

    def _group_xagg_by_combinations(self, combinations: Iterable[list]) -> Iterator[dict]:
        """Streams combinations *without* building the lists-of-lists xagg.

        The continuous Kruskal–Wallis path is closed-form and only needs
        ``index_to_groupby`` plus the precomputed per-raw-modality rank stats
        (see :meth:`_compute_associations`). Building the heavy
        ``Series.groupby(...).sum()`` of per-modality y-lists per combination
        was the dominant memory + time cost (Python list concatenation),
        so we skip it here entirely. The xagg is rebuilt lazily, on demand,
        inside :meth:`_test_viability_train` for the handful of top
        combinations actually checked for viability.
        """
        for combination in combinations:
            yield {
                "combination": combination,
                "index_to_groupby": combination_formatter(combination),
            }

    def _compute_associations(self, grouped_xaggs: Iterable[dict]) -> Iterator[dict]:
        """Closed-form, streaming Kruskal–Wallis across combinations.

        Statistically identical to ``scipy.stats.kruskal`` (including the
        tie correction factor); the only difference is that y is ranked **once**
        from :attr:`self.samples.train.xagg` instead of being re-ranked from
        scratch for every combination.

        Combinations are processed in batches of :data:`_KRUSKAL_BATCH_SIZE`.
        For each batch:

        * the per-combination group assignment is encoded as a 0/1 matrix
          ``A_c`` of shape ``(max_g, n_mod)`` and the batch is stacked into a
          single ``(B, max_g, n_mod)`` tensor ``A``;
        * the per-group rank sums and counts are obtained in **one BLAS call**
          as ``A @ R_per_mod`` and ``A @ n_per_mod`` (shape ``(B, max_g)``);
        * the Kruskal–Wallis H statistic is computed in closed form across
          the batch

          .. math::

              H = \\frac{12}{N(N+1)} \\sum_j \\frac{R_j^2}{n_j} - 3(N+1),
              \\quad H_{\\text{corrected}} = H \\,/\\, \\left(1 -
              \\frac{\\sum_i (t_i^3 - t_i)}{N^3 - N}\\right)

          where the tie correction factor depends only on the y values and is
          computed once per feature.

        Yields light association dicts ``{combination, index_to_groupby,
        kruskal}`` in arrival order — sorting happens in
        :meth:`CombinationEvaluator._get_best_association`.

        Edge cases follow ``scipy.stats.kruskal``:

        * any group with ``n_j == 0`` → ``H = NaN``;
        * fewer than 2 groups, or fewer than 2 observations total → ``None``
          (matches the existing ``ValueError`` swallowing in
          :meth:`_association_measure`).
        """
        raw_xagg = self.samples.train.xagg
        # Pre-rank y once for the whole feature
        R_per_mod, n_per_mod, N, tie_corr = _modality_rank_stats(raw_xagg)

        # Map modality label -> position in R_per_mod / n_per_mod
        mod_to_pos: dict = {m: i for i, m in enumerate(raw_xagg.index)}
        n_mod = len(mod_to_pos)

        batch: list[dict] = []
        for grouped_xagg in tqdm(grouped_xaggs, desc="Computing associations", disable=not self.verbose):
            batch.append(grouped_xagg)
            if len(batch) >= _KRUSKAL_BATCH_SIZE:
                yield from _kruskal_h_batch(
                    batch=batch,
                    R_per_mod=R_per_mod,
                    n_per_mod=n_per_mod,
                    N=N,
                    tie_corr=tie_corr,
                    mod_to_pos=mod_to_pos,
                    n_mod=n_mod,
                )
                batch = []
        if batch:
            yield from _kruskal_h_batch(
                batch=batch,
                R_per_mod=R_per_mod,
                n_per_mod=n_per_mod,
                N=N,
                tie_corr=tie_corr,
                mod_to_pos=mod_to_pos,
                n_mod=n_mod,
            )


class KruskalCombinations(ContinuousCombinationEvaluator):
    """Kruskal-Wallis' H based combination evaluation toolkit"""

    sort_by = "kruskal"


# Number of combinations processed per batched matmul call. Trades peak RAM
# (``B * max_g * n_mod`` doubles for the assignment tensor) for amortized
# Python overhead. 1024 keeps the tensor under ~10 MB for the typical
# (n_mod ≤ 40, max_g ≤ 7) sizes while making per-combination Python cost
# (dict iteration to fill the assign row) the dominant remaining cost.
_KRUSKAL_BATCH_SIZE = 1024


# ---------------------------------------------------------------------------
# Closed-form Kruskal–Wallis helpers
# ---------------------------------------------------------------------------


def _modality_rank_stats(
    raw_xagg: pd.Series,
) -> tuple[np.ndarray | None, np.ndarray, int, float | None]:
    """Rank ``raw_xagg``'s pooled y once and return per-modality stats.

    Returns ``(R_per_mod, n_per_mod, N, tie_corr)`` where:

    * ``R_per_mod[i]`` is the sum of average ranks of the y values in the i-th
      raw modality of ``raw_xagg`` (``rank_sum`` in Kruskal–Wallis notation);
    * ``n_per_mod[i]`` is the count of observations in the i-th raw modality;
    * ``N`` is the total number of observations;
    * ``tie_corr`` is the Kruskal–Wallis tie correction factor
      ``1 - Σ(t_i³ - t_i) / (N³ - N)`` — depends only on the y multiset.

    When ``N < 2``, ``R_per_mod`` and ``tie_corr`` are returned as ``None``
    so the per-combination caller can short-circuit.
    """
    raw_lists = [np.asarray(v, dtype=float) for v in raw_xagg.values]
    n_per_mod = np.fromiter((len(v) for v in raw_lists), dtype=np.int64, count=len(raw_lists))
    N = int(n_per_mod.sum())

    if N < 2 or len(raw_lists) == 0:
        return None, n_per_mod, N, None

    all_y = np.concatenate(raw_lists)
    ranks = rankdata(all_y, method="average")
    tie_corr = tiecorrect(ranks)

    offsets = np.concatenate([[0], np.cumsum(n_per_mod)])
    R_per_mod = np.empty(len(raw_lists), dtype=float)
    for i in range(len(raw_lists)):
        R_per_mod[i] = ranks[offsets[i] : offsets[i + 1]].sum()
    return R_per_mod, n_per_mod, N, tie_corr


def _kruskal_h_for_combination(
    *,
    R_per_mod: np.ndarray | None,
    n_per_mod: np.ndarray,
    N: int,
    tie_corr: float | None,
    mod_to_pos: dict,
    n_mod: int,
    index_to_groupby: dict,
) -> float | None:
    """Closed-form Kruskal–Wallis H for one combination.

    ``mod_to_pos`` is a precomputed ``{modality_label: position_in_R_per_mod}``
    map; ``index_to_groupby`` is the per-combination ``{modality: group_leader}``
    dict produced by :func:`combination_formatter`.
    """
    if R_per_mod is None or N < 2:
        return None

    # Build integer group assignment for this combination
    leader_to_grp: dict = {}
    assign = np.empty(n_mod, dtype=np.intp)
    assigned = np.zeros(n_mod, dtype=bool)
    for mod, leader in index_to_groupby.items():
        gid = leader_to_grp.get(leader)
        if gid is None:
            gid = len(leader_to_grp)
            leader_to_grp[leader] = gid
        pos = mod_to_pos[mod]
        assign[pos] = gid
        assigned[pos] = True

    # Modalities present in raw_xagg but not in index_to_groupby become their
    # own singleton groups so bincount has a well-defined assignment. Matches
    # the legacy binary `_grouper`'s `groupby.get(iv, iv)` semantics; the
    # continuous test suite reaches this path only in an invalid-state fixture
    # (has_nan=False but xagg carries a nan row) and the resulting Kruskal value
    # is discarded downstream when `xagg_apply_combination` raises on the
    # length mismatch — i.e. the user-visible behaviour is unchanged.
    for pos in range(n_mod):
        if not assigned[pos]:
            leader_to_grp[("__unmapped__", pos)] = len(leader_to_grp)
            assign[pos] = leader_to_grp[("__unmapped__", pos)]

    n_groups = len(leader_to_grp)
    # scipy.stats.kruskal requires at least 2 groups; mirror that here.
    if n_groups < 2:
        return None

    # Per-group rank sums and counts (vectorized).
    R_g = np.bincount(assign, weights=R_per_mod, minlength=n_groups)
    n_g = np.bincount(assign, weights=n_per_mod.astype(float), minlength=n_groups)

    # Empty groups -> NaN, matching scipy (0**2 / 0 → nan).
    if (n_g == 0).any():
        with np.errstate(invalid="ignore", divide="ignore"):
            ssbn = float((R_g**2 / n_g).sum())
        if not np.isfinite(ssbn):
            return float("nan")
    else:
        ssbn = float((R_g**2 / n_g).sum())

    h = (12.0 / (N * (N + 1))) * ssbn - 3.0 * (N + 1)

    # All values identical → tie_corr == 0; scipy returns nan from H/0.
    if tie_corr is None or tie_corr == 0:
        return float("nan")

    return h / tie_corr


def _kruskal_h_batch(  # noqa: C901
    *,
    batch: list[dict],
    R_per_mod: np.ndarray | None,
    n_per_mod: np.ndarray,
    N: int,
    tie_corr: float | None,
    mod_to_pos: dict,
    n_mod: int,
) -> Iterator[dict]:
    """Closed-form Kruskal–Wallis H for a batch of combinations via matmul.

    Encodes each combination's group assignment as a ``(max_g, n_mod)`` 0/1
    matrix stacked into ``A`` of shape ``(B, max_g, n_mod)``. Per-group rank
    sums and counts are obtained in one BLAS call as ``A @ R_per_mod`` and
    ``A @ n_per_mod``. Padding columns (``g >= n_groups[b]``) contribute zero
    by construction; in-range empty groups (``n_g == 0``) propagate NaN
    through ``ssbn`` exactly as :func:`_kruskal_h_for_combination` does.

    For ``B == 1`` ``max_g`` equals ``n_groups[0]``, so the math reduces to
    the scalar path and produces bit-identical floats — preserving the
    pinned values in the parity tests.
    """
    B = len(batch)

    # Short-circuit when ranking failed upstream (N<2 etc.); mirrors
    # :func:`_kruskal_h_for_combination`'s ``None`` return.
    if R_per_mod is None or N < 2:
        for item in batch:
            yield {
                "combination": item["combination"],
                "index_to_groupby": item["index_to_groupby"],
                "kruskal": None,
            }
        return

    # Build integer assignment matrix `assign[b, pos] = group_id`, plus
    # n_groups[b]. Unmapped positions become their own singleton groups —
    # matches :func:`_kruskal_h_for_combination` exactly.
    assign = np.empty((B, n_mod), dtype=np.intp)
    n_groups = np.empty(B, dtype=np.intp)
    for b, item in enumerate(batch):
        leader_to_grp: dict = {}
        assigned = np.zeros(n_mod, dtype=bool)
        for mod, leader in item["index_to_groupby"].items():
            gid = leader_to_grp.get(leader)
            if gid is None:
                gid = len(leader_to_grp)
                leader_to_grp[leader] = gid
            pos = mod_to_pos[mod]
            assign[b, pos] = gid
            assigned[pos] = True
        for pos in range(n_mod):
            if not assigned[pos]:
                assign[b, pos] = len(leader_to_grp)
                leader_to_grp[("__unmapped__", pos)] = len(leader_to_grp)
        n_groups[b] = len(leader_to_grp)

    max_g = int(n_groups.max())

    # 0/1 assignment tensor: A[b, g, m] = 1 iff modality m is in group g of combination b
    A = np.zeros((B, max_g, n_mod), dtype=np.float64)
    b_idx = np.repeat(np.arange(B), n_mod)
    mod_idx = np.tile(np.arange(n_mod), B)
    A[b_idx, assign.ravel(), mod_idx] = 1.0

    # Batched grouped stats via BLAS matmul
    R_g = A @ R_per_mod
    n_g = A @ n_per_mod.astype(np.float64)

    # Per-cell contribution to ssbn. Padding cells (g >= n_groups[b]) have
    # R_g == n_g == 0 → contribute NaN through 0/0; we zero them out so they
    # don't poison the per-combination sum. In-range cells with n_g == 0
    # (empty raw modality grouped alone) keep their NaN, which propagates
    # through `ssbn.sum()` exactly like the scalar path.
    in_range_mask = np.arange(max_g)[None, :] < n_groups[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        contrib = (R_g**2) / n_g
    contrib = np.where(in_range_mask, contrib, 0.0)
    ssbn = contrib.sum(axis=1)

    h = (12.0 / (N * (N + 1))) * ssbn - 3.0 * (N + 1)

    # All values identical → tie_corr == 0; scipy returns nan from H/0.
    if tie_corr is None or tie_corr == 0:
        h = np.full(B, float("nan"))
    else:
        h = h / tie_corr

    for b, item in enumerate(batch):
        if n_groups[b] < 2:
            h_val: float | None = None
        else:
            h_val = float(h[b])
        yield {
            "combination": item["combination"],
            "index_to_groupby": item["index_to_groupby"],
            "kruskal": h_val,
        }
