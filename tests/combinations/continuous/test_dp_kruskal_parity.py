"""Parity tests for the DP-based top-K Kruskal-Wallis segmentation path.

SPEEDUP_PLAN §8.1. ``_top_k_partitions_kruskal_dp`` enumerates the K best
*consecutive segmentations* of raw_index via an interval-DP over prefix sums,
instead of the current enumerate-then-score loop. These tests assert that the
DP's H values and partition shapes match the exhaustive enumeration path
(``consecutive_combinations`` + ``_kruskal_h_for_combination``).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from AutoCarver.combinations.continuous.continuous_combination_evaluators import (
    _kruskal_h_for_combination,
    _modality_rank_stats,
    _top_k_partitions_kruskal_dp,
)
from AutoCarver.combinations.utils.combinations import (
    combination_formatter,
    consecutive_combinations,
)


def _exhaustive_top_k(
    raw_xagg: pd.Series,
    max_n_mod: int,
    top_k: int,
) -> list[tuple[float, tuple[tuple, ...]]]:
    """Brute-force: score every combination, keep the top-K by H desc.

    Returns ``(H, combination_as_tuple_of_tuples)`` so combinations are
    hashable for set comparisons.
    """
    raw_labels = list(raw_xagg.index)
    R, n, N, tie_corr = _modality_rank_stats(raw_xagg)
    mod_to_pos = {m: i for i, m in enumerate(raw_labels)}
    n_mod = len(raw_labels)

    scored: list[tuple[float, tuple[tuple, ...]]] = []
    for combination in consecutive_combinations(raw_labels, max_n_mod):
        h = _kruskal_h_for_combination(
            R_per_mod=R,
            n_per_mod=n,
            N=N,
            tie_corr=tie_corr,
            mod_to_pos=mod_to_pos,
            n_mod=n_mod,
            index_to_groupby=combination_formatter(combination),
        )
        if h is None or math.isnan(h):
            continue
        scored.append((float(h), tuple(tuple(g) for g in combination)))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def _dp_call(raw_xagg: pd.Series, max_n_mod: int, top_k: int) -> list[dict]:
    R, n, N, tie_corr = _modality_rank_stats(raw_xagg)
    return _top_k_partitions_kruskal_dp(
        R,
        n,
        N,
        tie_corr,
        max_n_mod=max_n_mod,
        raw_index=list(raw_xagg.index),
        top_k=top_k,
    )


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("n_mod", [3, 5, 7, 10])
def test_dp_top1_matches_exhaustive_no_ties(seed: int, n_mod: int):
    """DP's single best H matches exhaustive best H on random continuous y."""
    rng = np.random.default_rng(seed * 100 + n_mod)
    sizes = rng.integers(2, 15, size=n_mod)
    raw = {f"m{i}": rng.standard_normal(size=int(sizes[i])).tolist() for i in range(n_mod)}
    xagg = pd.Series(raw)
    max_n_mod = min(7, n_mod)

    exhaustive = _exhaustive_top_k(xagg, max_n_mod=max_n_mod, top_k=10)
    dp = _dp_call(xagg, max_n_mod=max_n_mod, top_k=10)

    assert dp, "DP returned no candidates"
    assert exhaustive, "exhaustive returned no candidates"
    # closed form on prefix sums vs bincount — same math, different float order
    assert math.isclose(dp[0]["kruskal"], exhaustive[0][0], rel_tol=1e-10, abs_tol=1e-12)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("n_mod", [3, 5, 7, 10])
def test_dp_top_k_set_matches_exhaustive(seed: int, n_mod: int):
    """Top-K *partitions* (modulo H ties) match exhaustive enumeration."""
    rng = np.random.default_rng(seed * 100 + n_mod + 999)
    sizes = rng.integers(2, 12, size=n_mod)
    raw = {f"m{i}": rng.standard_normal(size=int(sizes[i])).tolist() for i in range(n_mod)}
    xagg = pd.Series(raw)
    max_n_mod = min(7, n_mod)
    top_k = 25

    exhaustive = _exhaustive_top_k(xagg, max_n_mod=max_n_mod, top_k=top_k)
    dp = _dp_call(xagg, max_n_mod=max_n_mod, top_k=top_k)

    # Compare H sequences pairwise to rtol=1e-10. Float order differs (prefix
    # subtraction vs bincount accumulation) so bit-exactness isn't expected.
    n_compare = min(len(exhaustive), len(dp))
    assert n_compare > 0
    for rank in range(n_compare):
        h_dp = dp[rank]["kruskal"]
        h_ex = exhaustive[rank][0]
        assert math.isclose(h_dp, h_ex, rel_tol=1e-10, abs_tol=1e-12), (
            f"rank {rank}: DP H={h_dp} vs exhaustive H={h_ex}"
        )

    # NOTE: partition-shape equality at rank 0 is *not* asserted. When two
    # adjacent raw modalities have identical mean ranks (R_a/n_a == R_b/n_b),
    # merging them leaves H unchanged — so multiple partitions are tied at
    # the same H. The H equality above is the correctness invariant; tie
    # resolution is implementation-defined and doesn't affect downstream
    # viability semantics.


@pytest.mark.parametrize("seed", range(5))
def test_dp_handles_heavy_ties(seed: int):
    """y drawn from a tiny integer alphabet → tie_corr < 1 but non-zero."""
    rng = np.random.default_rng(seed + 4242)
    n_mod = int(rng.integers(3, 7))
    sizes = rng.integers(3, 10, size=n_mod)
    raw = {f"m{i}": rng.integers(0, 4, size=int(sizes[i])).astype(float).tolist() for i in range(n_mod)}
    xagg = pd.Series(raw)
    max_n_mod = min(5, n_mod)

    exhaustive = _exhaustive_top_k(xagg, max_n_mod=max_n_mod, top_k=5)
    dp = _dp_call(xagg, max_n_mod=max_n_mod, top_k=5)

    assert dp, "DP returned no candidates"
    assert exhaustive, "exhaustive returned no candidates"
    assert math.isclose(dp[0]["kruskal"], exhaustive[0][0], rel_tol=1e-10, abs_tol=1e-12)


def test_dp_returns_empty_when_all_y_identical():
    """tie_corr == 0 → DP returns [] (caller treats as no scorable combos)."""
    xagg = pd.Series({"a": [1.0, 1.0, 1.0], "b": [1.0, 1.0], "c": [1.0, 1.0]})
    result = _dp_call(xagg, max_n_mod=3, top_k=10)
    assert result == []


def test_dp_returns_empty_when_n_below_two():
    xagg = pd.Series({"a": [1.0]})
    result = _dp_call(xagg, max_n_mod=2, top_k=10)
    assert result == []


def test_dp_excludes_empty_modality_segments():
    """A raw modality with zero observations must not appear as a singleton
    group in the DP output (would give n_g == 0 → NaN H, matching the
    exhaustive path which discards NaN scores)."""
    xagg = pd.Series({"a": [1.0, 2.0, 3.0], "b": [], "c": [4.0, 5.0]})
    result = _dp_call(xagg, max_n_mod=3, top_k=10)
    # exhaustive path drops NaN scores → only partitions where "b" is merged
    # with a neighbour are scorable. DP must agree.
    for entry in result:
        for group in entry["combination"]:
            # "b" must not appear alone
            assert group != ["b"]


def test_dp_index_to_groupby_matches_combination_formatter():
    """``index_to_groupby`` in DP output must equal
    ``combination_formatter(combination)`` so downstream viability tests work
    unchanged."""
    rng = np.random.default_rng(0)
    raw = {f"m{i}": rng.standard_normal(size=int(rng.integers(2, 8))).tolist() for i in range(6)}
    xagg = pd.Series(raw)
    result = _dp_call(xagg, max_n_mod=4, top_k=20)
    for entry in result:
        assert entry["index_to_groupby"] == combination_formatter(entry["combination"])


def test_dp_output_is_sorted_desc_by_kruskal():
    rng = np.random.default_rng(7)
    raw = {f"m{i}": rng.standard_normal(size=int(rng.integers(3, 10))).tolist() for i in range(7)}
    xagg = pd.Series(raw)
    result = _dp_call(xagg, max_n_mod=5, top_k=30)
    kruskals = [e["kruskal"] for e in result]
    assert kruskals == sorted(kruskals, reverse=True)
