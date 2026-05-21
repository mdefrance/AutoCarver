"""Parity tests for the closed-form Kruskal–Wallis path.

`ContinuousCombinationEvaluator._compute_associations` evaluates the
Kruskal–Wallis H statistic in closed form (single global ranking + per-group
reductions). These tests assert that the result matches `scipy.stats.kruskal`
across a wide variety of inputs, including:

  * many random partitions of random continuous y (no ties),
  * heavily-tied y values (so the tie correction factor actually does work),
  * empty groups in the partition,
  * 1-group partitions (must return None like scipy's swallowed ValueError),
  * very small samples (N < 2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kruskal

from AutoCarver.combinations.continuous.continuous_combination_evaluators import (
    KruskalCombinations,
    _kruskal_h_for_combination,
    _modality_rank_stats,
)
from AutoCarver.combinations.utils.combination_evaluator import AggregatedSample


def _scipy_kruskal_or_none(groups: list[list[float]]) -> float | None:
    """scipy.stats.kruskal wrapped to mirror the evaluator's error swallowing."""
    try:
        return float(kruskal(*tuple(groups))[0])
    except (ValueError, IndexError):
        return None


def _eval_via_evaluator(xagg: pd.Series, index_to_groupby: dict) -> float | None:
    """Drive `_compute_associations` on a single combination and return H."""
    evaluator = KruskalCombinations()
    evaluator.samples.train = AggregatedSample(xagg)
    grouped = {
        "xagg": xagg.groupby(index_to_groupby).sum(),
        "combination": [],
        "index_to_groupby": index_to_groupby,
    }
    result = evaluator._compute_associations([grouped])
    return result[0]["kruskal"]


def _scipy_groups_from(xagg: pd.Series, index_to_groupby: dict) -> list[list[float]]:
    """Reconstruct the per-group y lists in the same order scipy would see."""
    groups: dict[str, list[float]] = {}
    for mod in xagg.index:
        leader = index_to_groupby[mod]
        groups.setdefault(leader, []).extend(list(xagg[mod]))
    return list(groups.values())


# ---------------------------------------------------------------------------
# Pinned values from the existing test_continuous_combinations test suite
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "xagg, index_to_groupby, expected",
    [
        (pd.Series({"a": [0, 2, 0], "b": [2, 1], "c": [2, 0]}), {"a": "a", "b": "b", "c": "b"}, 0.5833333333333333),
        (pd.Series({"a": [0, 2, 0], "b": [2, 1], "c": [2, 0]}), {"a": "a", "b": "a", "c": "c"}, 0.0),
        (pd.Series({"A": [0, 2, 0], "B": [2, 1], "C": [2, 0]}), {"A": "A", "B": "B", "C": "C"}, 0.8333333333333345),
        (
            pd.Series(
                {
                    "A": [0, 2, 0],
                    "B": [2, 1],
                    "C": [2, 0, 5, 6],
                    "D": [1, 3, 4],
                    "E": [0, 1, 2, 3],
                    "F": [4, 5],
                    "G": [6, 7, 8],
                    "H": [9, 10],
                    "I": [11, 12, 13],
                    "J": [7, 8],
                }
            ),
            {"A": "A", "B": "B", "C": "B", "D": "B", "E": "B", "F": "F", "G": "F", "H": "H", "I": "H", "J": "H"},
            20.728840695728103,
        ),
    ],
)
def test_matches_pinned_test_values(xagg, index_to_groupby, expected):
    """Bit-for-bit match against the float values baked into existing tests."""
    got = _eval_via_evaluator(xagg, index_to_groupby)
    assert got == expected


# ---------------------------------------------------------------------------
# Empty groups → NaN, matching scipy behavior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "xagg, index_to_groupby",
    [
        (pd.Series({"A": [0, 2, 0], "B": [2, 1], "C": []}), {"A": "A", "B": "B", "C": "C"}),
        (pd.Series({"A": [0, 2, 0], "B": [2, 1], "C": []}), {"A": "A", "B": "A", "C": "C"}),
    ],
)
def test_empty_group_yields_nan(xagg, index_to_groupby):
    got = _eval_via_evaluator(xagg, index_to_groupby)
    assert got is not None
    assert np.isnan(got)


# ---------------------------------------------------------------------------
# Random parity: many random partitions of random y, mine vs scipy
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def _random_partition(rng: np.random.Generator, n_mod: int, n_groups: int) -> dict[int, int]:
    """Randomly assign modalities to groups, preserving the constraint that
    every group receives at least one modality (matches the carver's
    combination semantics).
    """
    n_groups = min(n_groups, n_mod)
    # ensure surjection: first n_groups modalities seed each group
    assign = list(range(n_groups)) + list(rng.integers(0, n_groups, size=n_mod - n_groups))
    rng.shuffle(assign)
    # convert raw group ids → group "leader" labels (mimic combination_formatter)
    leader_per_grp: dict[int, int] = {}
    out: dict[int, int] = {}
    for mod_idx, grp_id in enumerate(assign):
        if grp_id not in leader_per_grp:
            leader_per_grp[grp_id] = mod_idx
        out[mod_idx] = leader_per_grp[grp_id]
    return out


@pytest.mark.parametrize("seed", range(8))
def test_parity_no_ties(seed: int):
    """Continuous random y (vanishing tie probability) — pure closed form path."""
    rng = np.random.default_rng(seed)
    n_mod = int(rng.integers(2, 10))
    n_groups = int(rng.integers(2, n_mod + 1))
    sizes = rng.integers(1, 20, size=n_mod)
    raw: dict[int, list[float]] = {i: rng.standard_normal(size=sizes[i]).tolist() for i in range(n_mod)}
    xagg = pd.Series(raw)
    itog = _random_partition(rng, n_mod, n_groups)
    got = _eval_via_evaluator(xagg, itog)
    expected = _scipy_kruskal_or_none(_scipy_groups_from(xagg, itog))
    assert got == expected


@pytest.mark.parametrize("seed", range(8))
def test_parity_heavy_ties(seed: int):
    """y drawn from a tiny integer alphabet → tie correction factor exercises."""
    rng = np.random.default_rng(seed)
    n_mod = int(rng.integers(2, 8))
    n_groups = int(rng.integers(2, n_mod + 1))
    sizes = rng.integers(2, 15, size=n_mod)
    raw: dict[int, list[float]] = {i: rng.integers(0, 4, size=sizes[i]).astype(float).tolist() for i in range(n_mod)}
    xagg = pd.Series(raw)
    itog = _random_partition(rng, n_mod, n_groups)
    got = _eval_via_evaluator(xagg, itog)
    expected = _scipy_kruskal_or_none(_scipy_groups_from(xagg, itog))
    # scipy returns NaN for all-identical y; closed form does the same
    if expected is None:
        assert got is None
    elif np.isnan(expected):
        assert got is not None and np.isnan(got)
    else:
        assert got == expected


# ---------------------------------------------------------------------------
# Edge cases: tiny samples, 1-group combos, all-equal y
# ---------------------------------------------------------------------------


def test_single_group_returns_none():
    """A 1-group combination triggers ValueError in scipy (needs ≥2 groups);
    the closed form mirrors that by returning None."""
    xagg = pd.Series({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0]})
    itog = {"a": "a", "b": "a"}
    got = _eval_via_evaluator(xagg, itog)
    assert got is None


def test_total_below_two_returns_none():
    """N < 2 observations → None (mirrors scipy's ValueError swallowing)."""
    xagg = pd.Series({"a": [1.0]})
    itog = {"a": "a"}
    got = _eval_via_evaluator(xagg, itog)
    assert got is None


def test_all_identical_y_returns_nan():
    """Tie correction factor is 0 when every y is identical → NaN."""
    xagg = pd.Series({"a": [1.0, 1.0, 1.0], "b": [1.0, 1.0]})
    itog = {"a": "a", "b": "b"}
    got = _eval_via_evaluator(xagg, itog)
    expected = _scipy_kruskal_or_none(_scipy_groups_from(xagg, itog))
    if expected is None or (isinstance(expected, float) and np.isnan(expected)):
        assert got is not None and np.isnan(got)
    else:
        assert got == expected


# ---------------------------------------------------------------------------
# Helpers can be exercised independently for diagnostic use
# ---------------------------------------------------------------------------


def test_modality_rank_stats_returns_none_below_two_obs():
    raw_xagg = pd.Series({"a": [1.0]})
    R, n, N, tie = _modality_rank_stats(raw_xagg)
    assert R is None
    assert N == 1
    assert tie is None


def test_kruskal_h_for_combination_handles_none_inputs():
    """Defensive: if rank stats are None (N<2) the closed form returns None."""
    h = _kruskal_h_for_combination(
        R_per_mod=None,
        n_per_mod=np.array([1], dtype=np.int64),
        N=1,
        tie_corr=None,
        mod_to_pos={"a": 0},
        n_mod=1,
        index_to_groupby={"a": "a"},
    )
    assert h is None
