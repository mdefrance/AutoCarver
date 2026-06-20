"""Parity tests for the ordinal Phase-B interval DP.

`OrdinalCombinationEvaluator._get_best_combination_non_nan` replaces the
enumerate-and-score search with an interval DP over the additive ``C−D``
numerator (:func:`_top_k_partitions_ordinal_dp`). These tests pin:

  * the DP's per-partition metrics and its argmax against an exhaustive
    brute-force enumeration, for all three measures (exact when top_k is
    exhaustive, which it is for these small tables);
  * that the DP-driven search selects the *identical* combination the inherited
    enumerate-and-score path would, viability included.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from AutoCarver.combinations import (
    KendallTauBCombinations,
    KendallTauCCombinations,
    SomersDCombinations,
)
from AutoCarver.combinations.ordinal.ordinal_combination_evaluators import (
    _ordinal_associations,
    _top_k_partitions_ordinal_dp,
)
from AutoCarver.combinations.utils.combination_evaluator import CombinationEvaluator
from AutoCarver.combinations.utils.combinations import (
    combination_formatter,
    consecutive_combinations,
    group_crosstab,
)
from AutoCarver.features import OrdinalFeature

EVALUATORS = [KendallTauCCombinations, KendallTauBCombinations, SomersDCombinations]
SORT_KEYS = ["tau_c", "tau_b", "somersd"]


def _brute_best(xtab: pd.DataFrame, raw_index: list, max_n_mod: int, sort_by: str) -> float:
    best = -np.inf
    for combo in consecutive_combinations(raw_index, max_n_mod):
        grouped = group_crosstab(xtab, combination_formatter(combo))
        value = _ordinal_associations(grouped.values)[sort_by]
        if value is not None and value > best:
            best = value
    return best


@pytest.mark.parametrize("sort_by", SORT_KEYS)
@pytest.mark.parametrize("seed", range(15))
def test_dp_matches_bruteforce(sort_by: str, seed: int) -> None:
    """Exhaustive DP reproduces the brute-force metric values and argmax."""
    rng = np.random.default_rng(seed)
    n_mod = int(rng.integers(3, 9))
    n_cols = int(rng.integers(2, 7))
    M = rng.integers(0, 8, size=(n_mod, n_cols)).astype(float)
    raw_index = [f"m{i}" for i in range(n_mod)]
    xtab = pd.DataFrame(M, index=raw_index, columns=[f"L{j}" for j in range(n_cols)])
    max_n_mod = int(rng.integers(2, n_mod + 1))

    dp = _top_k_partitions_ordinal_dp(
        M, M.sum(axis=1), M.sum(axis=0), max_n_mod=max_n_mod, raw_index=raw_index, sort_by=sort_by, top_k=10**9
    )

    # every DP candidate's metrics equal the closed form on the grouped table
    for entry in dp:
        reference = _ordinal_associations(group_crosstab(xtab, entry["index_to_groupby"]).values)
        for key in SORT_KEYS:
            got, ref = entry[key], reference[key]
            assert (got is None) == (ref is None)
            if got is not None:
                assert got == pytest.approx(ref, abs=1e-9)

    # the DP's best matches the exhaustive brute-force best
    assert dp[0][sort_by] == pytest.approx(_brute_best(xtab, raw_index, max_n_mod, sort_by), abs=1e-9)


def _select(evaluator: CombinationEvaluator, xtab: pd.DataFrame, feature: OrdinalFeature, *, use_dp: bool):
    ev = evaluator
    ev.feature = copy.deepcopy(feature)
    ev.feature.dropna = False
    ev.max_n_mod, ev.min_freq, ev.dropna, ev.min_freq_alpha = 6, 0.03, False, 0.05
    ev.samples.set(train=xtab.copy())
    best = ev._get_best_combination_non_nan() if use_dp else CombinationEvaluator._get_best_combination_non_nan(ev)
    return None if best is None else [list(group) for group in best["combination"]]


@pytest.mark.parametrize("evaluator_cls", EVALUATORS)
@pytest.mark.parametrize("seed", range(12))
def test_dp_selection_matches_enumerate(evaluator_cls, seed: int) -> None:
    """The DP override selects the same combination as the base enumerate path."""
    rng = np.random.default_rng(seed)
    levels = [str(i) for i in range(int(rng.integers(3, 8)))]
    n_cols = int(rng.integers(2, 6))
    counts = rng.integers(0, 40, size=(len(levels), n_cols)).astype(int)
    xtab = pd.DataFrame(counts, index=levels, columns=list(range(1, n_cols + 1)))
    feature = OrdinalFeature("q", levels)

    dp_choice = _select(evaluator_cls(), xtab, feature, use_dp=True)
    enum_choice = _select(evaluator_cls(), xtab, feature, use_dp=False)
    assert dp_choice == enum_choice
