"""Module for continuous combination evaluators."""

import math
from abc import ABC
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kruskal, rankdata, tiecorrect
from tqdm import tqdm

from AutoCarver.combinations.continuous.continuous_target_rates import ContinuousTargetRate, TargetMean, TargetMedian
from AutoCarver.combinations.utils.combination_evaluator import (
    AggregatedSample,
    CombinationEvaluator,
    _nan_fanout_variants,
)
from AutoCarver.combinations.utils.combinations import combination_formatter
from AutoCarver.combinations.utils.target_rate import TargetRate
from AutoCarver.combinations.utils.testing import Keys, is_viable, test_viability
from AutoCarver.features import GroupedList


class ContinuousCombinationEvaluator(CombinationEvaluator[pd.Series], ABC):
    """Continuous combination evaluator class."""

    is_y_continuous = True
    _target_rate_classes: list[type[ContinuousTargetRate]] = [TargetMean, TargetMedian]
    # narrow the inherited `target_rate: TargetRate` annotation — continuous
    # carvers always carry a ContinuousTargetRate (enforced by _init_target_rate).
    target_rate: ContinuousTargetRate

    def _init_target_rate(self, target_rate: TargetRate[pd.Series] | None) -> ContinuousTargetRate:
        """Initializes target rate."""
        if target_rate is None:
            return TargetMean()
        elif not isinstance(target_rate, ContinuousTargetRate):
            raise ValueError("target_rate must be a ContinuousTargetRate")
        return target_rate

    def _association_measure(
        self,
        xagg: AggregatedSample | pd.Series | pd.DataFrame,
        n_obs: int | None = None,
        tol: float = 1e-10,
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
        # sort=False keeps groups in ordinal order (first appearance), not label
        # text order, so order-sensitive viability tests are label-independent.
        return xagg.groupby(groupby, sort=False).sum()

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
        # Pre-rank y once for the whole feature.
        R_per_mod, n_per_mod, N, tie_corr = _modality_rank_stats(raw_xagg)  # type: ignore

        # Map modality label -> position in R_per_mod / n_per_mod
        mod_to_pos: dict = {m: i for i, m in enumerate(raw_xagg.index)}
        n_mod = len(mod_to_pos)

        # Cache per-modality (n, sum_y) for the viability fast path.
        # Resets each time _compute_associations runs so the nan-pass refreshes
        # the cache after _apply_best_combination changes samples.train.xagg.
        sum_y_per_mod = _modality_sum_y(raw_xagg)  # type: ignore
        # Why: heterogeneous-value dict; annotate `Any` so downstream readers (line 203-204
        # and _get_dev_modality_stats) can narrow to the per-key concrete type without ty
        # unioning across all value types.
        self._train_modality_stats: dict[str, Any] = {
            "n_per_mod": n_per_mod.astype(float),
            "sum_y_per_mod": sum_y_per_mod,
            "mod_to_pos": mod_to_pos,
            "n_mod": n_mod,
        }
        self._dev_modality_stats: dict[str, Any] | None = None  # lazy; aligned to train's mod_to_pos
        self._dev_modality_stats_id: int | None = None

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

    def _get_dev_modality_stats(self) -> dict | None:
        """Lazily build per-modality ``(n, sum_y)`` for the dev sample,
        aligned to ``self._train_modality_stats['mod_to_pos']`` (zeros for
        modalities absent from dev). Returns ``None`` when no dev sample is set.

        Cache is keyed by ``id(dev_xagg)`` so external reassignment of
        ``samples.dev`` between viability iterations triggers a fresh
        computation (the unit tests rely on this; production flows reassign
        dev only via ``samples.set`` at the start of ``get_best_combination``).
        """
        if not self.samples.dev.has_xagg:
            return None
        dev_xagg = self.samples.dev.xagg
        if self._dev_modality_stats is not None and self._dev_modality_stats_id == id(dev_xagg):
            return self._dev_modality_stats
        train_stats = self._train_modality_stats
        mod_to_pos: dict = train_stats["mod_to_pos"]
        n_mod: int = train_stats["n_mod"]

        n = np.zeros(n_mod, dtype=float)
        sum_y = np.zeros(n_mod, dtype=float)
        for mod, vals in dev_xagg.items():
            pos = mod_to_pos.get(mod)
            if pos is None:
                continue  # dev has a modality train doesn't — skip
            arr = np.asarray(vals, dtype=float)
            n[pos] = arr.size
            sum_y[pos] = float(arr.sum())

        self._dev_modality_stats = {
            "n_per_mod": n,
            "sum_y_per_mod": sum_y,
            "mod_to_pos": mod_to_pos,
            "n_mod": n_mod,
        }
        self._dev_modality_stats_id = id(dev_xagg)
        return self._dev_modality_stats

    def _test_viability_train(self, combination: dict) -> dict:
        """Fast-path viability on train; falls back to legacy when the active
        target rate's ``compute_from_stats`` returns ``None`` (e.g.
        ``TargetMedian`` whose default closed-form path is a no-op).
        """
        stats = getattr(self, "_train_modality_stats", None)
        if stats is not None:
            train_rates = self.target_rate.compute_from_stats(
                stats=stats, index_to_groupby=combination["index_to_groupby"]
            )
            if train_rates is not None:
                return test_viability(train_rates, self.min_freq, self.target_rate.__name__, self.min_freq_alpha)
        # Fallback: legacy grouper + apply(np.mean/median) over Python lists
        return super()._test_viability_train(combination)

    def _get_best_combination_non_nan(self) -> dict | None:
        """DP-based override with progressive top-K.

        Replaces ``consecutive_combinations + _compute_associations`` with the
        interval-DP in :func:`_top_k_partitions_kruskal_dp`, which returns the
        top-K consecutive partitions ranked by Kruskal-Wallis H descending.

        **Progressive search.** Starts with ``top_k = self.dp_top_k_initial``.
        If the viability walk doesn't find a viable candidate within that top-K,
        doubles top_k and re-runs DP — walking only the new entries from where
        we left off. Repeats until either a viable is found or DP exhausts
        every consecutive partition (signalled by ``len(result) < top_k``).
        Total work bounded by ~2× a single DP run at the final top_k.

        This makes the search **exhaustive in the worst case**, matching the
        legacy enumerate-and-score path's correctness while keeping the common
        case (viable found in top ~100) essentially free.

        The NaN-fan-out path (:meth:`_get_best_combination_with_nan`) still
        goes through the legacy enumerate-and-score loop — handled in §8.3.
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

        # Pre-rank y once for the whole feature (same as _compute_associations).
        raw_xagg = self.samples.train.xagg
        R_per_mod, n_per_mod, N, tie_corr = _modality_rank_stats(raw_xagg)  # type: ignore
        sum_y_per_mod = _modality_sum_y(raw_xagg)  # type: ignore
        mod_to_pos: dict = {m: i for i, m in enumerate(raw_xagg.index)}
        n_mod = len(mod_to_pos)
        raw_index = list(raw_xagg.index)

        # Cache for viability fast path (mirrors _compute_associations).
        self._train_modality_stats: dict[str, Any] = {
            "n_per_mod": n_per_mod.astype(float),
            "sum_y_per_mod": sum_y_per_mod,
            "mod_to_pos": mod_to_pos,
            "n_mod": n_mod,
        }
        self._dev_modality_stats: dict[str, Any] | None = None
        self._dev_modality_stats_id: int | None = None

        # Progressive top-K with doubling. See docstring.
        top_k = self.dp_top_k_initial
        walked = 0
        viable: dict | None = None
        associations: list[dict] = []
        while True:
            associations = _top_k_partitions_kruskal_dp(
                R_per_mod,
                n_per_mod,
                N,
                tie_corr,
                max_n_mod=self.max_n_mod,
                raw_index=raw_index,
                top_k=top_k,
            )
            viable, walked = self._walk_for_viable(associations, start=walked)
            if viable is not None:
                break
            if walked < top_k:
                break  # DP exhausted every consecutive partition; no viable exists
            top_k *= 2

        # Rebuild grouped xagg for the winner (fast path skipped this).
        if viable is not None and viable.get("xagg") is None:
            index_to_groupby = viable.get("index_to_groupby") or combination_formatter(viable["combination"])
            viable["xagg"] = self._grouper(self.samples.train, index_to_groupby)

        self._apply_best_combination(viable)
        return viable

    def _get_best_combination_with_nan(self, best_combination: dict | None) -> dict | None:
        """DP-based override with NaN fan-out.

        Replaces ``nan_combinations + _get_best_association`` with:

        1. DP top-K base consecutive partitions over the non-nan labels
           (:func:`_top_k_partitions_kruskal_dp` on a restricted view of
           the per-modality stats);
        2. fan each base out across NaN placements exactly like
           :func:`nan_combinations` (nan folded into each group, then nan
           as its own group when ``len(base) < max_n_mod``, plus the final
           ``[all_non_nan, [nan]]`` partition);
        3. re-score every variant in closed form with
           :func:`_kruskal_h_for_combination` against the **full** per-modality
           stats (the nan row is included because :meth:`_get_best_combination_non_nan`'s
           ``_apply_best_combination`` repopulates ``samples.train.xagg`` with
           the nan modality intact);
        4. walk the sorted variants for the first viable, with progressive
           top-K doubling on the base DP — dedup'd via a per-partition seen
           set so combinations carried over from a smaller ``top_k`` are not
           re-tested / re-historized.

        Falls back to the parent implementation when the guard condition
        (``self.dropna and feature.has_nan and best_combination is not None``)
        is not met — matches the legacy short-circuit behaviour.
        """
        if not (self.dropna and self.feature.has_nan and best_combination is not None):
            return super()._get_best_combination_with_nan(best_combination)

        if self.verbose:
            print(f"[{self.__name__}] Grouping NaNs")

        feature_labels = self.feature.labels
        if feature_labels is None:
            raise RuntimeError(f"[{self.__name__}] feature labels are not populated")
        raw_labels = GroupedList(feature_labels[:])
        raw_labels.remove(self.feature.nan)
        nan_label = self.feature.nan

        # Full per-modality stats — the nan row is still in xagg because
        # _apply_best_combination on the non-nan winner rebuilt it from raw.
        raw_xagg = self.samples.train.xagg
        R_per_mod, n_per_mod, N, tie_corr = _modality_rank_stats(raw_xagg)  # type: ignore
        if R_per_mod is None or tie_corr is None or tie_corr == 0:
            # Degenerate cases (N<2 or all-identical y): legacy path returns
            # NaN/None scores and walks them anyway. Defer to it for parity.
            return super()._get_best_combination_with_nan(best_combination)

        sum_y_per_mod = _modality_sum_y(raw_xagg)  # type: ignore
        mod_to_pos: dict = {m: i for i, m in enumerate(raw_xagg.index)}
        n_mod = len(mod_to_pos)

        # Refresh viability fast-path cache to the with-nan stats.
        self._train_modality_stats: dict[str, Any] = {
            "n_per_mod": n_per_mod.astype(float),
            "sum_y_per_mod": sum_y_per_mod,
            "mod_to_pos": mod_to_pos,
            "n_mod": n_mod,
        }
        self._dev_modality_stats: dict[str, Any] | None = None
        self._dev_modality_stats_id: int | None = None

        # Non-nan subset, aligned to raw_labels order, for the base DP.
        non_nan_index = list(raw_labels)
        R_non_nan = np.fromiter(
            (R_per_mod[mod_to_pos[m]] for m in non_nan_index),
            dtype=float,
            count=len(non_nan_index),
        )
        n_non_nan = np.fromiter(
            (n_per_mod[mod_to_pos[m]] for m in non_nan_index),
            dtype=float,
            count=len(non_nan_index),
        )
        N_non_nan = int(n_non_nan.sum())

        historized: set[tuple] = set()
        base_top_k = self.dp_top_k_initial
        viable: dict | None = None

        while True:
            base_partitions = _top_k_partitions_kruskal_dp(
                R_non_nan,
                n_non_nan,
                N_non_nan,
                tie_corr,
                max_n_mod=self.max_n_mod,
                raw_index=non_nan_index,
                top_k=base_top_k,
            )
            scored = _score_nan_variants_kruskal(
                base_partitions=base_partitions,
                nan_label=nan_label,
                raw_labels=non_nan_index,
                max_n_mod=self.max_n_mod,
                R_per_mod=R_per_mod,
                n_per_mod=n_per_mod,
                N=N,
                tie_corr=tie_corr,
                mod_to_pos=mod_to_pos,
                n_mod=n_mod,
            )
            viable = self._walk_nan_variants(scored, historized)
            if viable is not None:
                break
            if len(base_partitions) < base_top_k:
                break  # DP exhausted every consecutive partition
            base_top_k *= 2

        if viable is not None and viable.get("xagg") is None:
            index_to_groupby = viable.get("index_to_groupby") or combination_formatter(viable["combination"])
            viable["xagg"] = self._grouper(self.samples.train, index_to_groupby)

        self._apply_best_combination(viable)
        return viable

    def _get_viable_combination(self, associations: list[dict]) -> dict | None:
        """Walks associations under the fast viability path and materialises
        the winning combination's grouped xagg once at the end.

        The fast path skips ``combination['xagg']`` because the closed-form
        viability check doesn't need it; downstream consumers (debug, tests,
        and any future code that introspects the winner) still expect to see
        it, so we rebuild it for the winner only — that's one ``_grouper``
        call per feature instead of ~13k per feature.
        """
        viable = super()._get_viable_combination(associations)
        if viable is not None and viable.get("xagg") is None:
            # `clean_combination` pops `index_to_groupby` during historization
            # earlier in the loop, so rebuild it from the still-present
            # `combination` list-of-groups.
            index_to_groupby = viable.get("index_to_groupby")
            if index_to_groupby is None:
                index_to_groupby = combination_formatter(viable["combination"])
            viable["xagg"] = self._grouper(self.samples.train, index_to_groupby)
        return viable

    def _test_viability_dev(self, test_results: dict, combination: dict) -> dict:
        """Fast-path viability on dev; falls back to legacy when the active
        target rate's ``compute_from_stats`` returns ``None``.
        """
        if not test_results[Keys.VIABLE.value] or not self.samples.dev.has_xagg:
            return {**test_results, "dev": {Keys.VIABLE.value: None}}

        dev_stats = self._get_dev_modality_stats()
        if dev_stats is not None:
            dev_rates = self.target_rate.compute_from_stats(
                stats=dev_stats, index_to_groupby=combination["index_to_groupby"]
            )
            if dev_rates is not None:
                train_target_rate = test_results["train_rates"][self.target_rate.__name__]
                dev_results = test_viability(
                    dev_rates, self.min_freq, self.target_rate.__name__, self.min_freq_alpha, train_target_rate
                )
                merged = {**test_results, **dev_results}
                merged[Keys.VIABLE.value] = is_viable(merged)
                return merged
        return super()._test_viability_dev(test_results, combination)


class KruskalCombinations(ContinuousCombinationEvaluator):
    """Kruskal-Wallis' H based combination evaluation toolkit.

    Search uses :ref:`progressive top-K interval DP <DPKruskal>` over the
    closed-form Kruskal-Wallis H decomposition (rank once over pooled ``y``,
    prefix-sum per-modality rank stats). Statistically equivalent to
    :func:`scipy.stats.kruskal` — bit-exact agreement pinned by parity tests.
    """

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


def _modality_sum_y(raw_xagg: pd.Series) -> np.ndarray:
    """Per-modality ``sum_y`` aligned with ``raw_xagg.index``.

    Used by the viability fast path (Step 3.5) to compute group target means
    in closed form (``sum_y_g / n_g``) instead of applying ``np.mean`` to
    Python lists of y values per candidate.
    """
    return np.fromiter(
        (float(np.asarray(v, dtype=float).sum()) for v in raw_xagg.values), dtype=float, count=len(raw_xagg)
    )


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


def _top_k_partitions_kruskal_dp(  # noqa: C901
    R_per_mod: np.ndarray | None,
    n_per_mod: np.ndarray,
    N: int,
    tie_corr: float | None,
    *,
    max_n_mod: int,
    raw_index: list,
    top_k: int = 1000,
) -> list[dict]:
    """Top-K consecutive-segmentation partitions ranked by Kruskal-Wallis H.

    Replaces enumerate-and-score with an interval-DP that exploits two facts:

    1. ``consecutive_combinations`` only emits segmentations of ``raw_index``
       (no out-of-order groupings) — a combination is fully determined by
       integer split positions ``0 = s_0 < ... < s_k = n_mod``.
    2. Kruskal-Wallis H is additively decomposable across groups:
       ``ssbn = Σ_g R_g² / n_g`` where ``R_g`` and ``n_g`` are obtained in
       O(1) from prefix sums.

    Complexity: O(K · n_mod² · top_k · log top_k). At ``n_mod = 40,
    max_n_mod = 7, top_k = 1000`` that's ~5.6 M ops — independent of the
    combination count (which can reach ~8 M at the same n_mod / max_n_mod).

    Returns a list of ``{combination, index_to_groupby, kruskal}`` dicts
    sorted by ``kruskal`` desc. Mirrors the shape ``_compute_associations``
    currently yields, so it can be dropped into the streaming pipeline.

    Edge cases (mirror :func:`_kruskal_h_for_combination`):
    * ``R_per_mod is None`` or ``N < 2`` or ``tie_corr is None or == 0``:
      returns ``[]`` (caller treats as "no scorable combinations").
    * Empty-modality segments (``Σ n_per_mod[i:j] == 0``) are excluded
      (they would otherwise produce ``nan`` H and lose to any non-empty
      partition in the sort anyway).
    """
    if R_per_mod is None or N < 2 or tie_corr is None or tie_corr == 0:
        return []

    n_mod = len(raw_index)
    K = min(max_n_mod, n_mod)
    if K < 2:
        return []

    R_prefix = np.concatenate([[0.0], np.cumsum(R_per_mod.astype(np.float64))])
    n_prefix = np.concatenate([[0.0], np.cumsum(n_per_mod.astype(np.float64))])

    def seg_cost(i: int, j: int) -> float:
        nn = n_prefix[j] - n_prefix[i]
        if nn <= 0:
            return float("-inf")  # empty segment — exclude
        r = R_prefix[j] - R_prefix[i]
        return (r * r) / nn

    # dp[k][j] holds up to ``top_k`` (S_total, splits_tuple) pairs sorted by
    # S_total desc, where splits_tuple = (0, s_1, ..., s_{k-1}, j).
    # k = number of groups, j = right boundary of the partition prefix.
    dp: list[list[list[tuple[float, tuple[int, ...]]]]] = [[[] for _ in range(n_mod + 1)] for _ in range(K + 1)]

    # Base case: a single group covering [0, j].
    for j in range(1, n_mod + 1):
        c = seg_cost(0, j)
        if c != float("-inf"):
            dp[1][j] = [(c, (0, j))]

    # Recurrence: dp[k][j] = best top_k of  dp[k-1][i] + seg_cost(i, j)  over i.
    for k in range(2, K + 1):
        for j in range(k, n_mod + 1):
            candidates: list[tuple[float, tuple[int, ...]]] = []
            for i in range(k - 1, j):
                c = seg_cost(i, j)
                if c == float("-inf"):
                    continue
                for prev_s, prev_splits in dp[k - 1][i]:
                    candidates.append((prev_s + c, prev_splits + (j,)))
            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                dp[k][j] = candidates[:top_k]

    # Collect full-coverage partitions for each k ∈ [2, K], translate ssbn → H.
    coef = 12.0 / (N * (N + 1))
    offset = 3.0 * (N + 1)
    final: list[tuple[float, tuple[int, ...]]] = []
    for k in range(2, K + 1):
        for s, splits in dp[k][n_mod]:
            h = (coef * s - offset) / tie_corr
            final.append((h, splits))
    final.sort(key=lambda x: x[0], reverse=True)
    final = final[:top_k]

    out: list[dict] = []
    for h, splits in final:
        combination = [list(raw_index[splits[g] : splits[g + 1]]) for g in range(len(splits) - 1)]
        out.append(
            {
                "combination": combination,
                "index_to_groupby": combination_formatter(combination),
                "kruskal": float(h),
            }
        )
    return out


def _score_nan_variants_kruskal(
    *,
    base_partitions: list[dict],
    nan_label: str,
    raw_labels: list,
    max_n_mod: int,
    R_per_mod: np.ndarray,
    n_per_mod: np.ndarray,
    N: int,
    tie_corr: float,
    mod_to_pos: dict,
    n_mod: int,
) -> list[dict]:
    """Score every NaN-fanout variant via closed-form Kruskal H, sorted desc.

    Uses :func:`_kruskal_h_for_combination` per variant (O(k) per call),
    which is plenty fast for the ~top_k * (max_n_mod + 1) + 1 variants
    produced by :func:`_nan_fanout_variants`.
    """
    scored: list[dict] = []
    for variant in _nan_fanout_variants(base_partitions, nan_label, raw_labels, max_n_mod):
        index_to_groupby = combination_formatter(variant)
        h = _kruskal_h_for_combination(
            R_per_mod=R_per_mod,
            n_per_mod=n_per_mod,
            N=N,
            tie_corr=tie_corr,
            mod_to_pos=mod_to_pos,
            n_mod=n_mod,
            index_to_groupby=index_to_groupby,
        )
        scored.append(
            {
                "combination": variant,
                "index_to_groupby": index_to_groupby,
                "kruskal": h,
            }
        )

    def _key(a: dict) -> float:
        v = a["kruskal"]
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return float("-inf")
        return float(v)

    scored.sort(key=_key, reverse=True)
    return scored
