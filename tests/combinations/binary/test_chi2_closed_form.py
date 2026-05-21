"""Parity tests for the closed-form chi² path.

`BinaryCombinationEvaluator._compute_associations` evaluates Cramér's V and
Tschuprow's T in closed form (single global aggregation + per-group reductions
via `np.bincount`). These tests assert the rounded values match the historical
``scipy.stats.chi2_contingency`` path on a wide variety of inputs, including:

  * many random partitions of random (n0, n1) modality counts (k > 2),
  * the 2×2 case where scipy applies Yates correction,
  * pinned values from the existing binary test suite,
  * the edge case where a modality is present in the crosstab but absent from
    ``index_to_groupby`` (now treated as its own singleton group, matching the
    legacy `_grouper`'s `groupby.get(iv, iv)` fallback).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2_contingency

from AutoCarver.combinations.binary.binary_combination_evaluators import (
    TschuprowtCombinations,
    _chi2_assoc_for_combination,
    _chi2_pearson_2col,
)
from AutoCarver.combinations.utils.combination_evaluator import AggregatedSample

TOL = 1e-10


def _scipy_assoc(grouped_table: np.ndarray, n_obs: float) -> dict[str, float]:
    """Reproduces the historical scipy-based BinaryCombinationEvaluator path."""
    n_mod_x = grouped_table.shape[0]
    chi2 = float(chi2_contingency(grouped_table + TOL)[0])
    cramerv = float(np.sqrt(chi2 / n_obs))
    if pd.notna(cramerv):
        cramerv = round(cramerv / TOL) * TOL
    if n_mod_x > 1:
        tschuprowt = cramerv / float(np.sqrt(np.sqrt(n_mod_x - 1)))
        if pd.notna(tschuprowt):
            tschuprowt = round(tschuprowt / TOL) * TOL
    else:
        tschuprowt = cramerv
    return {"cramerv": cramerv, "tschuprowt": tschuprowt}


def _eval_via_evaluator(xagg: pd.DataFrame, index_to_groupby: dict) -> dict[str, float]:
    """Drive `_compute_associations` on a single combination and return its dict.

    `_compute_associations` is a streaming generator; consume the single entry.
    """
    evaluator = TschuprowtCombinations()
    evaluator.samples.train = AggregatedSample(xagg)
    grouped = {
        "combination": [],
        "index_to_groupby": index_to_groupby,
    }
    assoc = next(iter(evaluator._compute_associations([grouped])))
    return {"cramerv": assoc["cramerv"], "tschuprowt": assoc["tschuprowt"]}


# ---------------------------------------------------------------------------
# Pinned values from the existing binary test suite
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "xagg, index_to_groupby, expected_cv, expected_tt",
    [
        # test_compute_associations: 2x2 (Yates applied)
        (
            pd.DataFrame({0: [0, 2, 0], 1: [2, 0, 1]}, index=["a", "b", "c"]),
            {"a": "a", "b": "b", "c": "b"},
            0.25,
            0.25,
        ),
        (
            pd.DataFrame({0: [0, 2, 0], 1: [2, 0, 1]}, index=["a", "b", "c"]),
            {"a": "a", "b": "a", "c": "c"},
            0.0,
            0.0,
        ),
        # test_compute_associations_with_three_rows: 3x2 (no Yates)
        (
            pd.DataFrame({0: [0, 2, 0], 1: [2, 0, 1]}, index=["A", "B", "C"]),
            {"A": "A", "B": "B", "C": "C"},
            0.9999999999,
            0.8408964152,
        ),
        # test_compute_associations_with_ten_labels: 3-group best from 10x2
        (
            pd.DataFrame(
                {0: [0, 2, 0, 1, 3, 0, 2, 1, 6, 2], 1: [5, 6, 1, 1, 2, 1, 0, 2, 1, 4]},
                index=[chr(i) for i in range(65, 75)],
            ),
            {"A": "A", "B": "A", "C": "A", "D": "D", "E": "D", "F": "D", "G": "D", "H": "D", "I": "D", "J": "J"},
            0.4719639494,
            0.3968727932,
        ),
    ],
)
def test_matches_pinned_test_values(xagg, index_to_groupby, expected_cv, expected_tt):
    """Bit-for-bit match against the rounded float values baked into existing tests."""
    got = _eval_via_evaluator(xagg, index_to_groupby)
    assert got["cramerv"] == expected_cv
    assert got["tschuprowt"] == expected_tt


# ---------------------------------------------------------------------------
# Random parity: many random partitions of random count tables, mine vs scipy
# ---------------------------------------------------------------------------


def _random_partition(rng: np.random.Generator, n_mod: int, n_groups: int) -> dict[int, int]:
    """Surjective random assignment of modalities to groups, then convert raw
    group ids → group "leader" labels (mimicking combination_formatter)."""
    n_groups = min(n_groups, n_mod)
    assign = list(range(n_groups)) + list(rng.integers(0, n_groups, size=n_mod - n_groups))
    rng.shuffle(assign)
    leader_per_grp: dict[int, int] = {}
    out: dict[int, int] = {}
    for mod_idx, grp_id in enumerate(assign):
        if grp_id not in leader_per_grp:
            leader_per_grp[grp_id] = mod_idx
        out[mod_idx] = leader_per_grp[grp_id]
    return out


def _grouped_table(xagg: pd.DataFrame, index_to_groupby: dict) -> np.ndarray:
    """Reconstruct the grouped (k, 2) table scipy would see for the legacy path."""
    rows: dict = {}
    for mod in xagg.index:
        leader = index_to_groupby[mod]
        if leader not in rows:
            rows[leader] = [0.0, 0.0]
        rows[leader][0] += xagg.iloc[xagg.index.get_loc(mod), 0]
        rows[leader][1] += xagg.iloc[xagg.index.get_loc(mod), 1]
    return np.array(list(rows.values()), dtype=float)


@pytest.mark.parametrize("seed", range(8))
def test_parity_random_partitions_kgt2(seed: int):
    """k > 2 groups, no Yates → vanilla Pearson chi²."""
    rng = np.random.default_rng(seed)
    n_mod = int(rng.integers(3, 12))
    n_groups = int(rng.integers(3, n_mod + 1))
    n0 = rng.integers(0, 20, size=n_mod).astype(float)
    n1 = rng.integers(0, 20, size=n_mod).astype(float)
    xagg = pd.DataFrame({0: n0, 1: n1}, index=[f"m{i}" for i in range(n_mod)])
    itog = {f"m{i}": f"m{leader}" for i, leader in _random_partition(rng, n_mod, n_groups).items()}
    n_obs = float(n0.sum() + n1.sum())
    got = _eval_via_evaluator(xagg, itog)
    expected = _scipy_assoc(_grouped_table(xagg, itog), n_obs)
    assert got == expected


@pytest.mark.parametrize("seed", range(8))
def test_parity_random_partitions_k2_yates(seed: int):
    """2 groups → 2×2 table → Yates correction applied by both paths."""
    rng = np.random.default_rng(100 + seed)
    n_mod = int(rng.integers(2, 10))
    n0 = rng.integers(1, 20, size=n_mod).astype(float)
    n1 = rng.integers(1, 20, size=n_mod).astype(float)
    xagg = pd.DataFrame({0: n0, 1: n1}, index=[f"m{i}" for i in range(n_mod)])
    itog = {f"m{i}": f"m{leader}" for i, leader in _random_partition(rng, n_mod, 2).items()}
    n_obs = float(n0.sum() + n1.sum())
    got = _eval_via_evaluator(xagg, itog)
    expected = _scipy_assoc(_grouped_table(xagg, itog), n_obs)
    assert got == expected


# ---------------------------------------------------------------------------
# Edge case: unmapped modality (treated as own singleton group)
# ---------------------------------------------------------------------------


def test_unmapped_modality_becomes_own_group():
    """Reproduces the legacy `_grouper`'s `groupby.get(iv, iv)` fallback.

    When the crosstab carries a row whose label is not in ``index_to_groupby``,
    that row must be treated as its own singleton group (otherwise bincount
    crashes on an uninitialized assignment slot). Verify the result matches
    what scipy would see on the equivalent (k+1, 2) table.
    """
    xagg = pd.DataFrame({0: [0, 2, 0, 0], 1: [2, 0, 1, 3]}, index=["a", "b", "c", "__NAN__"])
    itog = {"a": "a", "b": "b", "c": "b"}  # __NAN__ not present → becomes own group
    got = _eval_via_evaluator(xagg, itog)

    # scipy reference: same fallback semantics (each unmapped row is its own group)
    legacy_itog = {**itog, "__NAN__": "__NAN__"}
    expected = _scipy_assoc(_grouped_table(xagg, legacy_itog), n_obs=float(xagg.values.sum()))
    assert got == expected


# ---------------------------------------------------------------------------
# The closed-form chi² helper can be exercised independently
# ---------------------------------------------------------------------------


def test_chi2_pearson_2col_2x2_applies_yates():
    """For 2x2, Yates correction must be applied (matches scipy default)."""
    obs = np.array([[1e-10, 2], [2, 1]], dtype=float)
    got = _chi2_pearson_2col(obs)
    expected = float(chi2_contingency(obs)[0])
    assert got == expected


def test_chi2_pearson_2col_3x2_no_yates():
    """For 3x2 (or larger), no correction is applied."""
    obs = np.array([[1e-10, 2], [2, 1e-10], [1e-10, 1]], dtype=float)
    got = _chi2_pearson_2col(obs)
    expected = float(chi2_contingency(obs)[0])
    assert got == expected


def test_chi2_assoc_for_combination_handles_unassigned_positions():
    """Direct unit test for the helper; an unmapped position becomes its own
    group rather than crashing bincount with an uninitialized index."""
    n0 = np.array([0.0, 2.0, 0.0, 0.0])
    n1 = np.array([2.0, 0.0, 1.0, 3.0])
    mod_to_pos = {"a": 0, "b": 1, "c": 2, "__NAN__": 3}
    itog = {"a": "a", "b": "b", "c": "b"}  # 3 missing
    cv, tt = _chi2_assoc_for_combination(
        n0_per_mod=n0,
        n1_per_mod=n1,
        n_obs=float(n0.sum() + n1.sum()),
        mod_to_pos=mod_to_pos,
        n_mod=4,
        index_to_groupby=itog,
        tol=TOL,
    )
    assert np.isfinite(cv)
    assert np.isfinite(tt)
