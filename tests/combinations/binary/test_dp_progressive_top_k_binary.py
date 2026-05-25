"""Tests for the progressive top-K doubling loop in
:meth:`BinaryCombinationEvaluator._get_best_combination_non_nan`.

SPEEDUP_PLAN §8.4. When the viable winner sits past ``dp_top_k_initial``,
the DP must grow ``top_k`` (doubling each round) and keep walking until a
viable candidate is found or every consecutive partition is exhausted.
These tests pin that contract — mirrors
``tests/combinations/continuous/test_dp_progressive_top_k.py``.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from AutoCarver.combinations.binary.binary_combination_evaluators import (
    CramervCombinations,
    TschuprowtCombinations,
)
from AutoCarver.combinations.utils.combination_evaluator import AggregatedSample
from AutoCarver.features import OrdinalFeature


@pytest.fixture(params=[TschuprowtCombinations, CramervCombinations])
def evaluator_cls(request):
    return request.param


def _make_evaluator(cls, xagg: pd.DataFrame, max_n_mod: int, min_freq: float, dp_top_k_initial: int):
    """Fresh evaluator + fresh feature each call so state mutations don't leak."""
    ev = cls()
    ev.feature = OrdinalFeature("feature", list(xagg.index))
    ev.samples.train = AggregatedSample(xagg.copy())
    ev.samples.dev = AggregatedSample(xagg.copy())
    ev.min_freq = min_freq
    ev.max_n_mod = max_n_mod
    ev.dp_top_k_initial = dp_top_k_initial
    return ev


@pytest.fixture
def fixture_with_top_partition_unviable() -> pd.DataFrame:
    """Construct a fixture where the top-metric partition fails ``min_freq``.

    Counts are arranged so the partition that isolates the rare modality ``d``
    (1 obs total) maximises chi² but fails ``min_freq=0.2`` (need ≥ 2 obs in
    every group; ``d`` alone has 1 of 10 = 0.1). The walk must descend the
    ranking to find a viable partition that merges ``d`` into a neighbour.
    """
    return pd.DataFrame(
        {0: [3, 0, 0, 0], 1: [0, 3, 3, 1]},
        index=["a", "b", "c", "d"],
    )


@pytest.mark.parametrize("dp_top_k_initial", [1, 2, 3, 5, 100, 10000])
def test_progressive_top_k_finds_same_answer_regardless_of_initial(
    fixture_with_top_partition_unviable,
    evaluator_cls,
    dp_top_k_initial: int,
):
    """Whatever ``dp_top_k_initial`` is set to, the doubling loop must return
    the same viable winner.

    This is the core correctness guarantee: a viable winner past rank N must
    still be found regardless of the doubling start point.
    """
    ev = _make_evaluator(
        evaluator_cls,
        fixture_with_top_partition_unviable,
        max_n_mod=3,
        min_freq=0.2,
        dp_top_k_initial=dp_top_k_initial,
    )
    result = ev._get_best_combination_non_nan()
    assert result is not None, f"no viable found with dp_top_k_initial={dp_top_k_initial}"


def test_progressive_top_k_all_initials_agree(fixture_with_top_partition_unviable, evaluator_cls):
    """Stronger version: combination shape and metric values are identical
    across all initial top_k settings."""
    initials = [1, 2, 3, 5, 100, 10000]
    results = []
    for initial in initials:
        ev = _make_evaluator(
            evaluator_cls,
            fixture_with_top_partition_unviable,
            max_n_mod=3,
            min_freq=0.2,
            dp_top_k_initial=initial,
        )
        r = ev._get_best_combination_non_nan()
        assert r is not None
        results.append((initial, r["combination"], r["cramerv"], r["tschuprowt"]))

    base = results[0]
    for initial, combo, cv, tt in results[1:]:
        assert combo == base[1], f"combination mismatch at dp_top_k_initial={initial}: {combo} vs {base[1]}"
        assert math.isclose(cv, base[2], rel_tol=1e-12, abs_tol=1e-14), (
            f"cramerv mismatch at dp_top_k_initial={initial}: {cv} vs {base[2]}"
        )
        assert math.isclose(tt, base[3], rel_tol=1e-12, abs_tol=1e-14), (
            f"tschuprowt mismatch at dp_top_k_initial={initial}: {tt} vs {base[3]}"
        )


def test_progressive_top_k_returns_none_when_no_viable_exists(evaluator_cls):
    """When every partition fails viability, the doubling loop must terminate
    (when DP exhausts every partition: ``len(result) < top_k``) and return
    ``None`` — not loop forever.
    """
    # min_freq high enough that no partition can pass: any group needs ≥ 0.7
    # of the total observations; with N=6 that's 4.2, but each modality has 2
    # so no 2-or-3-group split can put 4+ obs in *every* group.
    xagg = pd.DataFrame({0: [1, 1, 1], 1: [1, 1, 1]}, index=["a", "b", "c"])
    ev = _make_evaluator(evaluator_cls, xagg, max_n_mod=3, min_freq=0.7, dp_top_k_initial=1)
    result = ev._get_best_combination_non_nan()
    assert result is None


def test_progressive_top_k_default_initial_is_fine_when_viable_at_top(evaluator_cls):
    """When the top-metric partition is already viable, the loop completes in
    the first iteration — no extra DP runs beyond the initial top_k."""
    # Matches the existing pinned test test_get_best_combination_non_nan_viable
    xagg = pd.DataFrame({0: [0, 2, 0], 1: [2, 0, 1]}, index=["a", "b", "c"])
    ev = _make_evaluator(evaluator_cls, xagg, max_n_mod=2, min_freq=0.2, dp_top_k_initial=1000)
    result = ev._get_best_combination_non_nan()
    assert result is not None
    assert result["combination"] == [["a"], ["b", "c"]]
    assert result["cramerv"] == 0.25
    assert result["tschuprowt"] == 0.25
