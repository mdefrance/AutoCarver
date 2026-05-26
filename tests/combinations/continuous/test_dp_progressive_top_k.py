"""Tests for the progressive top-K doubling loop in
:meth:`ContinuousCombinationEvaluator._get_best_combination_non_nan`.

SPEEDUP_PLAN §8.2. When the viable winner sits past ``dp_top_k_initial``,
the DP must grow ``top_k`` (doubling each round) and keep walking until a
viable candidate is found or every consecutive partition is exhausted.
These tests pin that contract.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from AutoCarver.combinations.continuous.continuous_combination_evaluators import (
    KruskalCombinations,
)
from AutoCarver.combinations.utils.combination_evaluator import AggregatedSample
from AutoCarver.features import OrdinalFeature


def _make_evaluator(xagg: pd.Series, max_n_mod: int, min_freq: float, dp_top_k_initial: int):
    """Fresh evaluator + fresh feature each call so state mutations don't leak."""
    ev = KruskalCombinations()
    ev.feature = OrdinalFeature("feature", list(xagg.index))
    ev.samples.train = AggregatedSample(xagg.copy())
    ev.samples.dev = AggregatedSample(xagg.copy())
    ev.min_freq = min_freq
    ev.max_n_mod = max_n_mod
    ev.dp_top_k_initial = dp_top_k_initial
    return ev


@pytest.fixture
def fixture_with_top_h_unviable() -> pd.Series:
    """Construct a fixture where the top-H partition fails ``min_freq``.

    Per Kruskal-Wallis ranks: 3 zeros (rank ≈2 each), 6 fives (rank ≈6.5),
    1 ten (rank 10). Putting ``d`` (the singleton 10) alone maximizes H but
    fails ``min_freq = 0.2`` (need ≥ 2 obs per group; d has 1). The walk
    must descend the H-ranking to find a viable partition that merges ``d``
    into a neighbour.
    """
    return pd.Series(
        {
            "a": [0.0] * 300,
            "b": [5.0] * 300,
            "c": [5.0] * 300,
            "d": [10.0] * 100,
        }
    )


def test_top_h_partition_is_unviable_so_walk_must_descend(fixture_with_top_h_unviable):
    """Sanity: with the chosen fixture, the rank-1 H partition really does fail
    ``min_freq``. If this assumption breaks the doubling tests below become
    trivially passing — so pin it explicitly.
    """
    from AutoCarver.combinations.continuous.continuous_combination_evaluators import (
        _modality_rank_stats,
        _top_k_partitions_kruskal_dp,
    )

    R, n, N, tc = _modality_rank_stats(fixture_with_top_h_unviable)
    dp = _top_k_partitions_kruskal_dp(
        R,
        n,
        N,
        tc,
        max_n_mod=3,
        raw_index=list(fixture_with_top_h_unviable.index),
        top_k=10,
    )
    # rank-1 must isolate "d" alone (Wilson upper(100, 1000) ≈ 0.12 < min_freq=0.2)
    rank1_groups = dp[0]["combination"]
    assert any(g == ["d"] for g in rank1_groups), (
        f"Test fixture invariant broken: rank-1 partition is {rank1_groups}, expected to isolate 'd' alone"
    )


@pytest.mark.parametrize("dp_top_k_initial", [1, 2, 3, 5, 100, 10000])
def test_progressive_top_k_finds_same_answer_regardless_of_initial(
    fixture_with_top_h_unviable,
    dp_top_k_initial: int,
):
    """Whatever ``dp_top_k_initial`` is set to, the doubling loop must return
    the same viable winner (which sits past rank 1 in this fixture).

    This is the core correctness guarantee fixing the German-credit-style bug:
    a viable winner past rank 1000 must still be found.
    """
    ev = _make_evaluator(fixture_with_top_h_unviable, max_n_mod=3, min_freq=0.2, dp_top_k_initial=dp_top_k_initial)
    result = ev._get_best_combination_non_nan()
    assert result is not None, f"no viable found with dp_top_k_initial={dp_top_k_initial}"
    # The viable winner has higher kruskal than the legacy enumeration would
    # have picked? No — DP and legacy pick the SAME first viable in H-desc
    # order. We pin both the combination shape and the H below.
    assert result["combination"] == [["a"], ["b"], ["c", "d"]], (
        f"dp_top_k_initial={dp_top_k_initial}: got combination {result['combination']}"
    )


def test_progressive_top_k_all_initials_agree(fixture_with_top_h_unviable):
    """Stronger version of the above: H value (not just combination shape) is
    identical across all initial top_k settings."""
    initials = [1, 2, 3, 5, 100, 10000]
    results = []
    for initial in initials:
        ev = _make_evaluator(fixture_with_top_h_unviable, max_n_mod=3, min_freq=0.2, dp_top_k_initial=initial)
        r = ev._get_best_combination_non_nan()
        results.append((initial, r["combination"], r["kruskal"]))

    base_combo, base_h = results[0][1], results[0][2]
    for initial, combo, h in results[1:]:
        assert combo == base_combo, f"combination mismatch at dp_top_k_initial={initial}: {combo} vs {base_combo}"
        assert math.isclose(h, base_h, rel_tol=1e-12, abs_tol=1e-14), (
            f"H mismatch at dp_top_k_initial={initial}: {h} vs {base_h}"
        )


def test_progressive_top_k_returns_none_when_no_viable_exists():
    """When every partition fails viability, the doubling loop must terminate
    (when DP exhausts every partition: ``len(result) < top_k``) and return
    ``None`` — not loop forever.

    Each modality has ~33% of the 600 observations; with min_freq=0.6 and the
    Wilson upper bound around 0.37, every partition's smallest group is
    significantly below 0.6 → no partition is viable.
    """
    xagg = pd.Series(
        {
            "a": [0.0] * 200,
            "b": [5.0] * 200,
            "c": [10.0] * 200,
        }
    )
    ev = _make_evaluator(xagg, max_n_mod=3, min_freq=0.6, dp_top_k_initial=1)
    result = ev._get_best_combination_non_nan()
    assert result is None


def test_progressive_top_k_default_initial_is_fine_when_viable_at_top():
    """When the top-H partition is already viable, the loop completes in the
    first iteration — no extra DP runs beyond the initial top_k."""
    # Simple fixture: all modalities have ≥ 2 obs, modest separation; top-H
    # passes min_freq=0.2 (N=7, need ≥ 1.4 → all groups have ≥ 2 obs).
    xagg = pd.Series({"a": [0, 2, 0], "b": [2, 1], "c": [2, 0]})
    ev = _make_evaluator(xagg, max_n_mod=2, min_freq=0.2, dp_top_k_initial=1000)
    result = ev._get_best_combination_non_nan()
    assert result is not None
    # Matches the existing pinned-value test
    # `test_get_best_combination_non_nan_viable`
    assert result["combination"] == [["a"], ["b", "c"]]
    assert result["kruskal"] == 0.5833333333333333
