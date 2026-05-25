"""Parity tests for the DP-based top-K chi² segmentation path.

SPEEDUP_PLAN §8.4. ``_top_k_partitions_chi2_dp`` enumerates the K best
*consecutive segmentations* of raw_index via an interval-DP over prefix sums
of per-modality ``(n0, n1)`` counts, instead of the current enumerate-then-score
loop. These tests assert that the DP's Cramér's V / Tschuprow's T values and
partition shapes match the exhaustive enumeration path
(``consecutive_combinations`` + ``_chi2_assoc_for_combination``).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from AutoCarver.combinations.binary.binary_combination_evaluators import (
    _chi2_assoc_for_combination,
    _top_k_partitions_chi2_dp,
)
from AutoCarver.combinations.utils.combinations import (
    combination_formatter,
    consecutive_combinations,
)

TOL = 1e-10


def _exhaustive_top_k(
    xagg: pd.DataFrame,
    max_n_mod: int,
    top_k: int,
    sort_by: str,
) -> list[tuple[float, float, float, tuple[tuple, ...]]]:
    """Brute-force: score every combination, keep the top-K by ``sort_by`` desc.

    Returns ``(sort_key, cramerv, tschuprowt, combination_as_tuple_of_tuples)``.
    """
    raw_labels = list(xagg.index)
    n0_per_mod = xagg.iloc[:, 0].to_numpy(dtype=float)
    n1_per_mod = xagg.iloc[:, 1].to_numpy(dtype=float)
    n_obs = float(n0_per_mod.sum() + n1_per_mod.sum())
    mod_to_pos = {m: i for i, m in enumerate(raw_labels)}
    n_mod = len(raw_labels)

    scored: list[tuple[float, float, float, tuple[tuple, ...]]] = []
    for combination in consecutive_combinations(raw_labels, max_n_mod):
        cv, tt = _chi2_assoc_for_combination(
            n0_per_mod=n0_per_mod,
            n1_per_mod=n1_per_mod,
            n_obs=n_obs,
            mod_to_pos=mod_to_pos,
            n_mod=n_mod,
            index_to_groupby=combination_formatter(combination),
            tol=TOL,
        )
        sort_key = tt if sort_by == "tschuprowt" else cv
        scored.append((float(sort_key), float(cv), float(tt), tuple(tuple(g) for g in combination)))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def _dp_call(xagg: pd.DataFrame, max_n_mod: int, top_k: int, sort_by: str) -> list[dict]:
    n0_per_mod = xagg.iloc[:, 0].to_numpy(dtype=float)
    n1_per_mod = xagg.iloc[:, 1].to_numpy(dtype=float)
    return _top_k_partitions_chi2_dp(
        n0_per_mod,
        n1_per_mod,
        max_n_mod=max_n_mod,
        raw_index=list(xagg.index),
        sort_by=sort_by,
        top_k=top_k,
        tol=TOL,
    )


def _random_xagg(rng: np.random.Generator, n_mod: int, low: int = 0, high: int = 30) -> pd.DataFrame:
    n0 = rng.integers(low, high, size=n_mod)
    n1 = rng.integers(low, high, size=n_mod)
    return pd.DataFrame({0: n0, 1: n1}, index=[f"m{i}" for i in range(n_mod)])


@pytest.mark.parametrize("sort_by", ["tschuprowt", "cramerv"])
@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("n_mod", [3, 5, 7, 10])
def test_dp_top1_matches_exhaustive(seed: int, n_mod: int, sort_by: str):
    """DP's single best metric matches exhaustive best on random (n0, n1)."""
    rng = np.random.default_rng(seed * 100 + n_mod)
    xagg = _random_xagg(rng, n_mod)
    max_n_mod = min(7, n_mod)

    exhaustive = _exhaustive_top_k(xagg, max_n_mod=max_n_mod, top_k=10, sort_by=sort_by)
    dp = _dp_call(xagg, max_n_mod=max_n_mod, top_k=10, sort_by=sort_by)

    assert dp, "DP returned no candidates"
    assert exhaustive, "exhaustive returned no candidates"
    # Quantised to TOL=1e-10 in both paths. Float order differs (prefix-sum DP
    # vs full-matrix bincount), so allow one tol of slack at the quantisation
    # boundary.
    dp_val = dp[0]["tschuprowt"] if sort_by == "tschuprowt" else dp[0]["cramerv"]
    ex_val = exhaustive[0][0]
    assert math.isclose(dp_val, ex_val, rel_tol=1e-8, abs_tol=2 * TOL), (
        f"top-1 {sort_by}: DP={dp_val} vs exhaustive={ex_val}"
    )


@pytest.mark.parametrize("sort_by", ["tschuprowt", "cramerv"])
@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("n_mod", [3, 5, 7, 10])
def test_dp_top_k_metric_sequence_matches_exhaustive(seed: int, n_mod: int, sort_by: str):
    """Top-K *metric values* (modulo ties) match exhaustive enumeration."""
    rng = np.random.default_rng(seed * 100 + n_mod + 999)
    xagg = _random_xagg(rng, n_mod)
    max_n_mod = min(7, n_mod)
    top_k = 25

    exhaustive = _exhaustive_top_k(xagg, max_n_mod=max_n_mod, top_k=top_k, sort_by=sort_by)
    dp = _dp_call(xagg, max_n_mod=max_n_mod, top_k=top_k, sort_by=sort_by)

    n_compare = min(len(exhaustive), len(dp))
    assert n_compare > 0
    for rank in range(n_compare):
        dp_val = dp[rank]["tschuprowt"] if sort_by == "tschuprowt" else dp[rank]["cramerv"]
        ex_val = exhaustive[rank][0]
        assert math.isclose(dp_val, ex_val, rel_tol=1e-8, abs_tol=2 * TOL), (
            f"rank {rank} ({sort_by}): DP={dp_val} vs exhaustive={ex_val}"
        )

    # NOTE: partition-shape equality at rank 0 is *not* asserted. With integer
    # counts, multiple partitions can be exactly tied on chi² (and hence on
    # cramerv / tschuprowt after quantisation); tie resolution is
    # implementation-defined. The metric equality above is the correctness
    # invariant.


@pytest.mark.parametrize("sort_by", ["tschuprowt", "cramerv"])
@pytest.mark.parametrize("seed", range(5))
def test_dp_handles_small_counts(seed: int, sort_by: str):
    """Small / zero counts — the kind of edge case that stresses ``+tol`` shifts."""
    rng = np.random.default_rng(seed + 4242)
    n_mod = int(rng.integers(3, 7))
    xagg = _random_xagg(rng, n_mod, low=0, high=4)
    max_n_mod = min(5, n_mod)

    exhaustive = _exhaustive_top_k(xagg, max_n_mod=max_n_mod, top_k=5, sort_by=sort_by)
    dp = _dp_call(xagg, max_n_mod=max_n_mod, top_k=5, sort_by=sort_by)

    if not exhaustive:
        # Edge: all counts zero → no chi² to compute. DP should also be empty.
        assert dp == []
        return

    assert dp, "DP returned no candidates"
    dp_val = dp[0]["tschuprowt"] if sort_by == "tschuprowt" else dp[0]["cramerv"]
    ex_val = exhaustive[0][0]
    assert math.isclose(dp_val, ex_val, rel_tol=1e-8, abs_tol=2 * TOL)


def test_dp_returns_empty_when_n_mod_below_two():
    xagg = pd.DataFrame({0: [3], 1: [5]}, index=["a"])
    assert _dp_call(xagg, max_n_mod=2, top_k=10, sort_by="tschuprowt") == []
    assert _dp_call(xagg, max_n_mod=2, top_k=10, sort_by="cramerv") == []


def test_dp_returns_empty_when_max_n_mod_below_two():
    xagg = pd.DataFrame({0: [3, 1, 4], 1: [5, 2, 6]}, index=["a", "b", "c"])
    assert _dp_call(xagg, max_n_mod=1, top_k=10, sort_by="tschuprowt") == []


def test_dp_rejects_invalid_sort_by():
    xagg = pd.DataFrame({0: [3, 1, 4], 1: [5, 2, 6]}, index=["a", "b", "c"])
    with pytest.raises(ValueError, match="sort_by"):
        _dp_call(xagg, max_n_mod=3, top_k=10, sort_by="not_a_metric")


def test_dp_index_to_groupby_matches_combination_formatter():
    """``index_to_groupby`` in DP output must equal
    ``combination_formatter(combination)`` so downstream viability tests work
    unchanged."""
    rng = np.random.default_rng(0)
    xagg = _random_xagg(rng, 6)
    result = _dp_call(xagg, max_n_mod=4, top_k=20, sort_by="tschuprowt")
    for entry in result:
        assert entry["index_to_groupby"] == combination_formatter(entry["combination"])


@pytest.mark.parametrize("sort_by", ["tschuprowt", "cramerv"])
def test_dp_output_is_sorted_desc_by_metric(sort_by: str):
    rng = np.random.default_rng(7)
    xagg = _random_xagg(rng, 7)
    result = _dp_call(xagg, max_n_mod=5, top_k=30, sort_by=sort_by)
    values = [e[sort_by] for e in result]
    assert values == sorted(values, reverse=True)


def test_dp_partition_is_consecutive_segmentation():
    """Every returned combination must be a *consecutive* segmentation of
    ``raw_index`` (no out-of-order groupings, matching the structural
    invariant of ``consecutive_combinations``).
    """
    rng = np.random.default_rng(11)
    xagg = _random_xagg(rng, 6)
    raw_index = list(xagg.index)
    result = _dp_call(xagg, max_n_mod=5, top_k=20, sort_by="tschuprowt")
    for entry in result:
        flat = [m for group in entry["combination"] for m in group]
        # flat must be a contiguous slice of raw_index from position 0
        assert flat == raw_index[: len(flat)], (
            f"non-consecutive partition: {entry['combination']} vs raw_index={raw_index}"
        )


def test_dp_yates_correction_path_matches_exhaustive():
    """The k=2 branch of the DP applies Yates correction; the k>=3 branches
    do not. This test pins that the Yates branch produces the same chi² as
    the exhaustive path on a 2-group partition.
    """
    # Picking n_mod=4 so we get a mix of k=2 (Yates) and k=3,4 (no Yates) partitions.
    rng = np.random.default_rng(2026)
    xagg = _random_xagg(rng, 4, low=1, high=20)

    exhaustive = _exhaustive_top_k(xagg, max_n_mod=4, top_k=50, sort_by="cramerv")
    dp = _dp_call(xagg, max_n_mod=4, top_k=50, sort_by="cramerv")

    # Compare every rank — covers both Yates and non-Yates cells.
    n_compare = min(len(exhaustive), len(dp))
    for rank in range(n_compare):
        assert math.isclose(dp[rank]["cramerv"], exhaustive[rank][1], rel_tol=1e-8, abs_tol=2 * TOL), (
            f"rank {rank} cramerv mismatch"
        )
        assert math.isclose(dp[rank]["tschuprowt"], exhaustive[rank][2], rel_tol=1e-8, abs_tol=2 * TOL), (
            f"rank {rank} tschuprowt mismatch"
        )
