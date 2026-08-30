"""Ordinal candidates are ranked by association *magnitude*, not signed value.

The ordinal rank statistics (tau-b, tau-c, Somers' D) are signed. A declared ordinal
feature's order is fixed by the caller, so a feature that runs opposite to the target
scores negative on *every* consecutive grouping — ranking those by signed value picks the
least negative, i.e. the weakest grouping available, and discards a real inverse signal.

An inverse ordinal relationship is ordinary (more rainfall, fewer fire claims), so the
strongest association must win regardless of its direction.
"""

import numpy as np
import pandas as pd
import pytest

from AutoCarver import Features, MulticlassCarver, OrdinalCarver
from AutoCarver.combinations import (
    KendallTauBCombinations,
    KendallTauCCombinations,
    SomersDCombinations,
)
from AutoCarver.combinations.ordinal.ordinal_combination_evaluators import _top_k_partitions_ordinal_dp
from AutoCarver.combinations.utils.combinations import (
    combination_formatter,
    consecutive_combinations,
    group_crosstab,
)
from AutoCarver.discretizers import ProcessingConfig
from AutoCarver.stats import rank_associations

SORT_KEYS = ["tau_c", "tau_b", "somersd"]

# declared ascending, target rate strictly DECREASING: every grouping scores negative
LEVELS = ["0_none", "1_low", "2_mid", "3_high", "4_extreme"]
RATES = [0.060, 0.045, 0.030, 0.015, 0.004]
PER_LEVEL = 20_000


def _inverse_crosstab() -> pd.DataFrame:
    """Ordered crosstab (levels x {0, 1, 2}) with a monotonically decreasing target rate."""
    rows = []
    for rate in RATES:
        n_claim = int(PER_LEVEL * rate)
        n_two = n_claim // 4
        rows.append([PER_LEVEL - n_claim, n_claim - n_two, n_two])
    return pd.DataFrame(rows, index=LEVELS, columns=[0, 1, 2], dtype=float)


def _brute_best_magnitude(xtab: pd.DataFrame, max_n_mod: int, sort_by: str) -> float:
    """Largest |metric| over every consecutive grouping — what the search should find."""
    best = 0.0
    for combo in consecutive_combinations(list(xtab.index), max_n_mod):
        value = rank_associations(group_crosstab(xtab, combination_formatter(combo)).values)[sort_by]
        if value is not None and abs(value) > best:
            best = abs(value)
    return best


@pytest.mark.parametrize("sort_by", SORT_KEYS)
def test_dp_finds_strongest_inverse_association(sort_by: str) -> None:
    """The DP's top candidate carries the largest |metric|, not the least negative one."""
    xtab = _inverse_crosstab()
    values = xtab.values

    dp = _top_k_partitions_ordinal_dp(
        values,
        values.sum(axis=1),
        values.sum(axis=0),
        max_n_mod=5,
        raw_index=list(xtab.index),
        sort_by=sort_by,
        top_k=10**9,
    )

    assert dp, "the DP returned no candidate"
    expected = _brute_best_magnitude(xtab, 5, sort_by)
    assert abs(dp[0][sort_by]) == pytest.approx(expected, abs=1e-9)
    # the association really is inverse — otherwise this table would not test anything
    assert dp[0][sort_by] < 0


@pytest.mark.parametrize("evaluator_cls", [KendallTauCCombinations, KendallTauBCombinations, SomersDCombinations])
def test_dp_candidates_ordered_by_magnitude(evaluator_cls) -> None:
    """Candidates come back ranked by |metric| descending, ready for the viability walk."""
    xtab = _inverse_crosstab()
    values = xtab.values
    sort_by = evaluator_cls.sort_by

    dp = _top_k_partitions_ordinal_dp(
        values,
        values.sum(axis=1),
        values.sum(axis=0),
        max_n_mod=5,
        raw_index=list(xtab.index),
        sort_by=sort_by,
        top_k=10**9,
    )

    magnitudes = [abs(entry[sort_by]) for entry in dp if entry[sort_by] is not None]
    assert magnitudes == sorted(magnitudes, reverse=True)


def _carve(carver_cls, **kwargs) -> float:
    """Fits a carver on the inverse feature and returns its grouping's |tau_c|."""
    frame = pd.DataFrame({"rainfall": np.repeat(LEVELS, PER_LEVEL)})
    target = []
    for rate in RATES:
        n_claim = int(PER_LEVEL * rate)
        n_two = n_claim // 4
        target += [2] * n_two + [1] * (n_claim - n_two) + [0] * (PER_LEVEL - n_claim)
    y = pd.Series(target)

    carver = carver_cls(
        features=Features(ordinals={"rainfall": LEVELS}),
        min_freq=0.02,
        max_n_mod=5,
        config=ProcessingConfig(dropna=False, copy=False, verbose=False, n_jobs=1),
        **kwargs,
    )
    carved = carver.fit_transform(frame.copy(), y, X_dev=frame.copy(), y_dev=y)
    return abs(rank_associations(pd.crosstab(carved["rainfall"], y).values)["tau_c"])


def test_ordinal_carver_keeps_the_inverse_signal() -> None:
    """End to end: OrdinalCarver must not lose to a carver that ignores the ordering."""
    ordinal = _carve(OrdinalCarver, target_scale="level")
    multiclass = _carve(MulticlassCarver)

    # the ordinal carver optimises tau-c directly; it must at least match the carver
    # whose objective is sign-free and order-invariant
    assert ordinal >= multiclass - 1e-9
    assert ordinal == pytest.approx(_brute_best_magnitude(_inverse_crosstab(), 5, "tau_c"), abs=1e-9)
