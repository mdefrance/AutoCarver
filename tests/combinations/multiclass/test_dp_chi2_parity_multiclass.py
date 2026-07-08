"""Parity tests for the multiclass DP-based top-K chi² segmentation path.

``_top_k_partitions_chi2_dp_multiclass`` enumerates the K best *consecutive
segmentations* of raw_index via an interval-DP over prefix sums of
per-modality ``(K,)`` class counts, generalising
:func:`AutoCarver.combinations.binary.binary_combination_evaluators._top_k_partitions_chi2_dp`.
These tests assert the DP's Cramér's V / Tschuprow's T values match exhaustive
enumeration (``consecutive_combinations`` + the closed-form chi²), for K in
{2, 3, 5}, and that K=2 also matches the binary evaluator's own DP.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from AutoCarver.combinations.binary.binary_combination_evaluators import _top_k_partitions_chi2_dp
from AutoCarver.combinations.multiclass.multiclass_combination_evaluators import (
    _chi2_pearson,
    _cramerv_tschuprowt,
    _dp_inputs_from_xagg,
    _top_k_partitions_chi2_dp_multiclass,
)
from AutoCarver.combinations.utils.combinations import combination_formatter, consecutive_combinations

TOL = 1e-10


def _exhaustive_top_k(M: np.ndarray, raw_index: list, max_n_mod: int, top_k: int, sort_by: str) -> list[tuple]:
    """Brute-force: score every consecutive combination, keep the top-K desc."""
    n_classes = M.shape[1]
    total_n = float(M.sum())
    scored: list[tuple[float, float, float, tuple[tuple, ...]]] = []
    for combination in consecutive_combinations(raw_index, max_n_mod):
        itog = combination_formatter(combination)
        leaders = list(dict.fromkeys(itog[m] for m in raw_index))
        leader_pos = {leader: i for i, leader in enumerate(leaders)}
        grouped = np.zeros((len(leaders), n_classes))
        for i, mod in enumerate(raw_index):
            grouped[leader_pos[itog[mod]]] += M[i]
        chi2 = _chi2_pearson(grouped + TOL)
        cv, tt = _cramerv_tschuprowt(chi2, total_n, len(leaders), n_classes, TOL)
        sort_key = tt if sort_by == "tschuprowt" else cv
        key = sort_key if pd.notna(sort_key) else float("-inf")
        scored.append((float(key), float(cv), float(tt), tuple(tuple(g) for g in combination)))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def _dp_call(M: np.ndarray, raw_index: list, max_n_mod: int, top_k: int, sort_by: str) -> list[dict]:
    n_per_mod = M.sum(axis=1)
    col_sums = M.sum(axis=0)
    return _top_k_partitions_chi2_dp_multiclass(
        M, n_per_mod, col_sums, max_n_mod=max_n_mod, raw_index=raw_index, sort_by=sort_by, top_k=top_k
    )


def _random_M(rng: np.random.Generator, n_mod: int, n_classes: int, low: int = 0, high: int = 30) -> np.ndarray:
    return rng.integers(low, high, size=(n_mod, n_classes)).astype(float)


@pytest.mark.parametrize("sort_by", ["tschuprowt", "cramerv"])
@pytest.mark.parametrize("n_classes", [2, 3, 5])
@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("n_mod", [3, 5, 7])
def test_dp_top1_matches_exhaustive(seed: int, n_mod: int, n_classes: int, sort_by: str):
    """DP's single best metric matches exhaustive best on random (n_mod, K) counts."""
    rng = np.random.default_rng(seed * 1000 + n_mod * 10 + n_classes)
    M = _random_M(rng, n_mod, n_classes)
    raw_index = [f"m{i}" for i in range(n_mod)]
    max_n_mod = min(6, n_mod)

    exhaustive = _exhaustive_top_k(M, raw_index, max_n_mod=max_n_mod, top_k=10, sort_by=sort_by)
    dp = _dp_call(M, raw_index, max_n_mod=max_n_mod, top_k=10, sort_by=sort_by)

    assert dp, "DP returned no candidates"
    assert exhaustive, "exhaustive returned no candidates"
    dp_val = dp[0][sort_by]
    ex_val = exhaustive[0][0]
    assert math.isclose(dp_val, ex_val, rel_tol=1e-8, abs_tol=2 * TOL), (
        f"top-1 {sort_by} K={n_classes}: DP={dp_val} vs exhaustive={ex_val}"
    )


@pytest.mark.parametrize("sort_by", ["tschuprowt", "cramerv"])
@pytest.mark.parametrize("n_classes", [3, 5])
@pytest.mark.parametrize("seed", range(5))
def test_dp_top_k_metric_sequence_matches_exhaustive(seed: int, n_classes: int, sort_by: str):
    """Top-K metric values (modulo ties) match exhaustive enumeration."""
    rng = np.random.default_rng(seed * 100 + n_classes + 999)
    n_mod = 7
    M = _random_M(rng, n_mod, n_classes)
    raw_index = [f"m{i}" for i in range(n_mod)]
    max_n_mod = 6
    top_k = 25

    exhaustive = _exhaustive_top_k(M, raw_index, max_n_mod=max_n_mod, top_k=top_k, sort_by=sort_by)
    dp = _dp_call(M, raw_index, max_n_mod=max_n_mod, top_k=top_k, sort_by=sort_by)

    n_compare = min(len(exhaustive), len(dp))
    assert n_compare > 0
    for rank in range(n_compare):
        dp_val = dp[rank][sort_by]
        ex_val = exhaustive[rank][0]
        assert math.isclose(dp_val, ex_val, rel_tol=1e-8, abs_tol=2 * TOL), f"rank {rank} ({sort_by}): mismatch"


@pytest.mark.parametrize("sort_by", ["tschuprowt", "cramerv"])
@pytest.mark.parametrize("seed", range(6))
def test_dp_k2_matches_binary_dp(seed: int, sort_by: str):
    """At K=2, the multiclass DP matches the binary DP bit-for-bit (same chi²
    formulas; the K=2-parity anchor extended to the DP path)."""
    rng = np.random.default_rng(seed + 55)
    n_mod = int(rng.integers(3, 9))
    M = _random_M(rng, n_mod, 2, low=1, high=20)
    raw_index = [f"m{i}" for i in range(n_mod)]
    max_n_mod = min(5, n_mod)

    dp_multiclass = _dp_call(M, raw_index, max_n_mod=max_n_mod, top_k=20, sort_by=sort_by)
    dp_binary = _top_k_partitions_chi2_dp(
        M[:, 0], M[:, 1], max_n_mod=max_n_mod, raw_index=raw_index, sort_by=sort_by, top_k=20
    )

    n_compare = min(len(dp_multiclass), len(dp_binary))
    assert n_compare > 0
    for rank in range(n_compare):
        assert dp_multiclass[rank]["cramerv"] == dp_binary[rank]["cramerv"], f"rank {rank} cramerv"
        assert dp_multiclass[rank]["tschuprowt"] == dp_binary[rank]["tschuprowt"], f"rank {rank} tschuprowt"
        assert dp_multiclass[rank]["combination"] == dp_binary[rank]["combination"], f"rank {rank} combination"


def test_dp_empty_modality_is_compacted():
    """An all-zero raw modality (row) must never form its own group; it folds
    into an adjacent group instead — the ordinal DP's zero-row bug, re-tested
    here for the multiclass DP."""
    raw_index = ["a", "b", "c", "d"]
    M = np.array(
        [
            [10.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],  # empty modality
            [1.0, 10.0, 1.0],
            [1.0, 1.0, 10.0],
        ]
    )
    result = _dp_call(M, raw_index, max_n_mod=4, top_k=50, sort_by="tschuprowt")
    assert result, "DP returned no candidates"
    for entry in result:
        for group in entry["combination"]:
            # every group must have at least one non-empty modality contributing
            # (a group made of only "b" would be an all-zero, degenerate group)
            assert any(m != "b" for m in group) or len(entry["combination"]) == 1


def test_dp_returns_empty_when_all_modalities_are_empty():
    raw_index = ["a", "b", "c"]
    M = np.zeros((3, 3))
    assert _dp_call(M, raw_index, max_n_mod=3, top_k=10, sort_by="tschuprowt") == []


def test_dp_rejects_invalid_sort_by():
    raw_index = ["a", "b", "c"]
    M = np.array([[3.0, 5.0, 1.0], [1.0, 2.0, 4.0], [4.0, 6.0, 0.0]])
    with pytest.raises(ValueError, match="sort_by"):
        _dp_call(M, raw_index, max_n_mod=3, top_k=10, sort_by="not_a_metric")


def test_dp_index_to_groupby_matches_combination_formatter():
    rng = np.random.default_rng(0)
    M = _random_M(rng, 6, 3)
    raw_index = [f"m{i}" for i in range(6)]
    result = _dp_call(M, raw_index, max_n_mod=4, top_k=20, sort_by="tschuprowt")
    for entry in result:
        assert entry["index_to_groupby"] == combination_formatter(entry["combination"])


@pytest.mark.parametrize("sort_by", ["tschuprowt", "cramerv"])
def test_dp_output_is_sorted_desc_by_metric(sort_by: str):
    rng = np.random.default_rng(7)
    M = _random_M(rng, 7, 4)
    raw_index = [f"m{i}" for i in range(7)]
    result = _dp_call(M, raw_index, max_n_mod=5, top_k=30, sort_by=sort_by)
    values = [e[sort_by] for e in result]
    assert values == sorted(values, reverse=True)


def test_dp_partition_is_consecutive_segmentation():
    rng = np.random.default_rng(11)
    M = _random_M(rng, 6, 3)
    raw_index = [f"m{i}" for i in range(6)]
    result = _dp_call(M, raw_index, max_n_mod=5, top_k=20, sort_by="tschuprowt")
    for entry in result:
        flat = [m for group in entry["combination"] for m in group]
        assert flat == raw_index[: len(flat)]


def test_dp_inputs_from_xagg_zero_fills_missing_rows():
    xagg = pd.DataFrame({1: [3, 5], 2: [1, 2], 3: [0, 4]}, index=["a", "c"])
    raw_index = ["a", "b", "c"]
    M, n_per_mod, col_sums = _dp_inputs_from_xagg(xagg, raw_index)
    assert M.shape == (3, 3)
    assert list(n_per_mod) == [4.0, 0.0, 11.0]
    assert list(col_sums) == [8.0, 3.0, 4.0]
