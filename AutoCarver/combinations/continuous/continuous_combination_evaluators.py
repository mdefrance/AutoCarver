"""Module for continuous combination evaluators."""

import warnings
from abc import ABC
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kruskal, rankdata, tiecorrect

from AutoCarver.combinations.continuous.continuous_target_rates import ContinuousTargetRate, TargetMean, TargetMedian
from AutoCarver.combinations.utils.combination_evaluator import AggregatedSample, CombinationEvaluator
from AutoCarver.combinations.utils.combinations import combination_formatter
from AutoCarver.combinations.utils.dp import (
    build_group_assignment,
    score_nan_variants,
    splits_to_combination,
    top_k_partitions,
)
from AutoCarver.combinations.utils.target_rate import TargetRate
from AutoCarver.combinations.utils.testing import Keys, is_viable, test_viability
from AutoCarver.features import GroupedList
from AutoCarver.stats.kruskal import h_from_rank_sums


class ContinuousCombinationEvaluator(CombinationEvaluator[pd.Series], ABC):
    """Continuous combination evaluator class."""

    is_y_continuous = True
    _target_rate_classes: list[type[ContinuousTargetRate]] = [TargetMean, TargetMedian]
    # narrow the inherited `target_rate: TargetRate` annotation — continuous
    # carvers always carry a ContinuousTargetRate (enforced by _init_target_rate).
    target_rate: ContinuousTargetRate

    # viability fast-path cache, (re)built by the DP paths in
    # `_get_best_combination_non_nan` / `_get_best_combination_with_nan`. Declared
    # here because those paths can bail out before building it (single-modality
    # train sample) while the legacy parent path still runs viability tests after
    # them — `None` means "no closed form available, use the legacy grouper".
    _train_modality_stats: dict[str, Any] | None = None
    _dev_modality_stats: dict[str, Any] | None = None
    _dev_modality_stats_id: int | None = None

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

        Used for the raw (one-shot) distribution and the NaN-fanout scoring path.
        The hot per-combination loop goes through :meth:`_get_best_combination_non_nan`'s
        DP, which evaluates the same Kruskal–Wallis H statistic in closed form
        without re-ranking — see :func:`_modality_rank_stats` and
        :func:`_kruskal_h_for_combination`.

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

        # Kruskal-Wallis' H (degenerate groups legitimately yield NaN: scipy's
        # SmallSampleWarning / tie-correction RuntimeWarning are silenced here so
        # end users don't see them)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
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
        # longer goes through this path — see _get_best_combination_non_nan's DP.
        # sort=False keeps groups in ordinal order (first appearance), not label
        # text order, so order-sensitive viability tests are label-independent.
        return xagg.groupby(groupby, sort=False).sum()

    def _get_dev_modality_stats(self) -> dict | None:
        """Lazily build per-modality ``(n, sum_y)`` for the dev sample,
        aligned to ``self._train_modality_stats['mod_to_pos']`` (zeros for
        modalities absent from dev). Returns ``None`` when no dev sample is set,
        or when the train-side cache was never built (the DP paths bail out
        before building it on a single-modality train sample) — the caller then
        falls back to the legacy grouper.

        Cache is keyed by ``id(dev_xagg)`` so external reassignment of
        ``samples.dev`` between viability iterations triggers a fresh
        computation (the unit tests rely on this; production flows reassign
        dev only via ``samples.set`` at the start of ``get_best_combination``).
        """
        if not self.samples.dev.has_xagg:
            return None
        train_stats = self._train_modality_stats
        if train_stats is None:
            return None
        dev_xagg = self.samples.dev.xagg
        if self._dev_modality_stats is not None and self._dev_modality_stats_id == id(dev_xagg):
            return self._dev_modality_stats
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

        Replaces ``consecutive_combinations`` + enumerate-and-score with the
        interval-DP in :func:`_top_k_partitions_kruskal_dp`, which returns the
        top-K consecutive partitions ranked by Kruskal-Wallis H descending.

        **Progressive search.** Starts with ``top_k = self.dp_top_k_initial``.
        If the viability walk doesn't find a viable candidate within that top-K
        and escalation is enabled (``dp_escalate``), grows top_k ×4 and re-runs
        DP — walking only the new entries from where we left off. Repeats until
        either a viable is found or DP exhausts every consecutive partition
        (signalled by ``len(result) < top_k``). Total work bounded by ~1.33× a
        single DP run at the final top_k.

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

        self._historize_raw_combination()

        if self.samples.train.shape[0] <= 1:
            return None

        # Pre-rank y once for the whole feature.
        raw_xagg = self.samples.train.xagg
        R_per_mod, n_per_mod, N, tie_corr = _modality_rank_stats(raw_xagg)  # type: ignore
        sum_y_per_mod = _modality_sum_y(raw_xagg)  # type: ignore
        mod_to_pos: dict = {m: i for i, m in enumerate(raw_xagg.index)}
        n_mod = len(mod_to_pos)
        raw_index = list(raw_xagg.index)

        # Cache for the viability fast path (_test_viability_train/_test_viability_dev).
        self._train_modality_stats = {
            "n_per_mod": n_per_mod.astype(float),
            "sum_y_per_mod": sum_y_per_mod,
            "mod_to_pos": mod_to_pos,
            "n_mod": n_mod,
        }
        self._dev_modality_stats = None
        self._dev_modality_stats_id = None

        # Progressive top-K with ×4 growth (shared driver). See docstring.
        viable = self._search_escalating(
            lambda top_k: _top_k_partitions_kruskal_dp(
                R_per_mod,
                n_per_mod,
                N,
                tie_corr,
                max_n_mod=self.max_n_mod,
                raw_index=raw_index,
                top_k=top_k,
            )
        )
        self._rebuild_winner_xagg(viable)
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
           top-K ×4 growth on the base DP — dedup'd via a per-partition seen
           set so combinations carried over from a smaller ``top_k`` are not
           re-tested / re-historized.

        Falls back to the parent implementation when the guard condition
        (``self.dropna and feature.has_nan``) is not met — matches the legacy
        short-circuit behaviour. Runs even when the non-nan search failed
        (``best_combination is None``): the nan row is restored from raw
        before reading per-modality stats below.
        """
        if not (self.dropna and self.feature.has_nan):
            return super()._get_best_combination_with_nan(best_combination)

        # non-nan search failed -> xaggs are still nan-filtered; restore the nan
        # row before reading per-modality stats below
        if best_combination is None:
            self.samples.restore_raw()

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
        self._train_modality_stats = {
            "n_per_mod": n_per_mod.astype(float),
            "sum_y_per_mod": sum_y_per_mod,
            "mod_to_pos": mod_to_pos,
            "n_mod": n_mod,
        }
        self._dev_modality_stats = None
        self._dev_modality_stats_id = None

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

        def _run_round(top_k: int) -> tuple[list[dict], int]:
            base_partitions = _top_k_partitions_kruskal_dp(
                R_non_nan,
                n_non_nan,
                N_non_nan,
                tie_corr,
                max_n_mod=self.max_n_mod,
                raw_index=non_nan_index,
                top_k=top_k,
            )

            def _scorer(index_to_groupby: dict) -> dict:
                h = _kruskal_h_for_combination(
                    R_per_mod=R_per_mod,
                    n_per_mod=n_per_mod,
                    N=N,
                    tie_corr=tie_corr,
                    mod_to_pos=mod_to_pos,
                    n_mod=n_mod,
                    index_to_groupby=index_to_groupby,
                )
                return {"kruskal": h}

            scored = score_nan_variants(
                base_partitions=base_partitions,
                nan_label=nan_label,
                raw_labels=non_nan_index,
                max_n_mod=self.max_n_mod,
                scorer=_scorer,
                sort_by="kruskal",
            )
            return scored, len(base_partitions)

        viable = self._search_escalating_nan(_run_round)
        self._rebuild_winner_xagg(viable)

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
    # scipy.stats.tiecorrect is bit-exact against scipy elsewhere in this module's
    # parity tests; AutoCarver.stats.kruskal.tie_correction is its scalar twin but
    # is not substituted here to keep that scipy pin intact.
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

    # Build integer group assignment for this combination. Modalities present in
    # raw_xagg but not in index_to_groupby become their own singleton groups so
    # bincount has a well-defined assignment. Matches the legacy binary
    # `_grouper`'s `groupby.get(iv, iv)` semantics; the continuous test suite
    # reaches this path only in an invalid-state fixture (has_nan=False but xagg
    # carries a nan row) and the resulting Kruskal value is discarded downstream
    # when `xagg_apply_combination` raises on the length mismatch — i.e. the
    # user-visible behaviour is unchanged.
    assign, n_groups = build_group_assignment(index_to_groupby, mod_to_pos, n_mod)
    # scipy.stats.kruskal requires at least 2 groups; mirror that here.
    if n_groups < 2:
        return None

    # Per-group rank sums and counts (vectorized).
    R_g = np.bincount(assign, weights=R_per_mod, minlength=n_groups)
    n_g = np.bincount(assign, weights=n_per_mod.astype(float), minlength=n_groups)

    # All values identical → tie_corr == 0; scipy returns nan from H/0.
    if tie_corr is None:
        return float("nan")

    return h_from_rank_sums(R_g, n_g, N, tie_corr)


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
    sorted by ``kruskal`` desc — the shape the viability walk expects.

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

    # Recurrence: dp[k][j] = best top_k of  dp[k-1][i] + seg_cost(i, j)  over i.
    entries = top_k_partitions(
        n_mod=n_mod, cap=K, seg_cost=seg_cost, top_k=top_k, maximize=True, skip_cost=float("-inf")
    )

    # Collect full-coverage partitions for each k ∈ [2, K], translate ssbn → H.
    coef = 12.0 / (N * (N + 1))
    offset = 3.0 * (N + 1)
    final: list[tuple[float, tuple[int, ...]]] = []
    for _, s, splits in entries:
        h = (coef * s - offset) / tie_corr
        final.append((h, splits))
    final.sort(key=lambda x: x[0], reverse=True)
    final = final[:top_k]

    out: list[dict] = []
    for h, splits in final:
        combination = splits_to_combination(splits, raw_index)
        out.append(
            {
                "combination": combination,
                "index_to_groupby": combination_formatter(combination),
                "kruskal": float(h),
            }
        )
    return out
