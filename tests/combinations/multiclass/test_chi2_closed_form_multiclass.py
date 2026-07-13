"""Parity tests for the multiclass closed-form chi² path.

`MulticlassCombinationEvaluator` generalises the binary evaluator's 2-column
Pearson chi² to a K-column (unordered) target. These tests assert:

  * the closed form matches ``scipy.stats.chi2_contingency`` on random
    ``(B, K)`` tables for K in {2, 3, 5};
  * for K == 2, the multiclass evaluator's Cramér's V / Tschuprow's T are
    bit-for-bit identical to the binary evaluator's — the strongest
    correctness anchor available, since the two formulas provably coincide
    at K=2.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2_contingency

from AutoCarver.combinations.binary.binary_combination_evaluators import (
    _chi2_assoc_for_combination,
)
from AutoCarver.combinations.multiclass.multiclass_combination_evaluators import (
    MulticlassCombinationEvaluator,
    TschuprowtMulticlassCombinations,
    _chi2_pearson,
    _cramerv_tschuprowt,
)

TOL = 1e-10


def _scipy_chi2(obs: np.ndarray) -> float:
    return float(chi2_contingency(obs)[0])


@pytest.mark.parametrize("n_classes", [2, 3, 5])
@pytest.mark.parametrize("seed", range(8))
def test_chi2_pearson_matches_scipy(seed: int, n_classes: int):
    """Closed-form chi² matches scipy on random (B, K) tables, K in {2,3,5}."""
    rng = np.random.default_rng(seed * 10 + n_classes)
    n_groups = int(rng.integers(2, 8))
    obs = rng.integers(0, 25, size=(n_groups, n_classes)).astype(float) + TOL
    got = _chi2_pearson(obs)
    expected = _scipy_chi2(obs)
    assert got == pytest.approx(expected, abs=1e-8)


def test_chi2_pearson_2x2_applies_yates():
    """For a 2x2 table, Yates correction must be applied (matches scipy default)."""
    obs = np.array([[1e-10, 2], [2, 1]], dtype=float)
    got = _chi2_pearson(obs)
    expected = _scipy_chi2(obs)
    assert got == expected


def test_chi2_pearson_3x2_no_yates():
    """For 3x2 (B > 2), no Yates correction is applied."""
    obs = np.array([[1e-10, 2], [2, 1e-10], [1e-10, 1]], dtype=float)
    got = _chi2_pearson(obs)
    expected = _scipy_chi2(obs)
    assert got == expected


# ---------------------------------------------------------------------------
# K=2 bit-for-bit parity with the binary evaluator
# ---------------------------------------------------------------------------


def _grouped_table(xagg: pd.DataFrame, index_to_groupby: dict) -> np.ndarray:
    rows: dict = {}
    for mod in xagg.index:
        leader = index_to_groupby[mod]
        if leader not in rows:
            rows[leader] = np.zeros(xagg.shape[1])
        rows[leader] = rows[leader] + xagg.loc[mod].to_numpy(dtype=float)
    return np.array(list(rows.values()), dtype=float)


def _random_partition(rng: np.random.Generator, n_mod: int, n_groups: int) -> dict[int, int]:
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


@pytest.mark.parametrize("seed", range(8))
def test_k2_association_measure_matches_binary_bit_for_bit(seed: int):
    """On a 2-column crosstab, `_association_measure`'s cramerv/tschuprowt must
    equal the binary evaluator's `_chi2_assoc_for_combination` bit-for-bit."""
    rng = np.random.default_rng(seed)
    n_mod = int(rng.integers(2, 10))
    n_groups = int(rng.integers(2, n_mod + 1))
    n0 = rng.integers(0, 20, size=n_mod).astype(float)
    n1 = rng.integers(0, 20, size=n_mod).astype(float)
    xagg = pd.DataFrame({0: n0, 1: n1}, index=[f"m{i}" for i in range(n_mod)])
    itog = {f"m{i}": f"m{leader}" for i, leader in _random_partition(rng, n_mod, n_groups).items()}

    grouped = _grouped_table(xagg, itog)
    n_obs = float(n0.sum() + n1.sum())

    evaluator = TschuprowtMulticlassCombinations()
    got = evaluator._association_measure(pd.DataFrame(grouped), n_obs=n_obs)

    mod_to_pos = {m: i for i, m in enumerate(xagg.index)}
    expected_cv, expected_tt = _chi2_assoc_for_combination(
        n0_per_mod=n0,
        n1_per_mod=n1,
        n_obs=n_obs,
        mod_to_pos=mod_to_pos,
        n_mod=n_mod,
        index_to_groupby=itog,
        tol=TOL,
    )
    assert got["cramerv"] == expected_cv
    assert got["tschuprowt"] == expected_tt


def test_cramerv_tschuprowt_k2_matches_binary_formula():
    """`_cramerv_tschuprowt` at n_classes=2 reduces to binary's own expression."""
    rng = np.random.default_rng(42)
    for _ in range(20):
        n_groups = int(rng.integers(2, 10))
        chi2 = float(rng.uniform(0, 50))
        n_obs = float(rng.uniform(10, 200))
        cv, tt = _cramerv_tschuprowt(chi2, n_obs, n_groups, 2, TOL)

        cramerv_ref = float(np.sqrt(chi2 / n_obs))
        if pd.notna(cramerv_ref):
            cramerv_ref = round(cramerv_ref / TOL) * TOL
        tt_ref = cramerv_ref / float(np.sqrt(np.sqrt(n_groups - 1)))
        if pd.notna(tt_ref):
            tt_ref = round(tt_ref / TOL) * TOL

        assert cv == cramerv_ref
        assert tt == tt_ref


def test_evaluator_rejects_non_multiclass_target_rate():
    """A non-MulticlassTargetRate target_rate is rejected at construction."""
    from AutoCarver.combinations.ordinal.ordinal_target_rates import TargetMeanLevel

    with pytest.raises(ValueError, match="MulticlassTargetRate"):
        TschuprowtMulticlassCombinations(target_rate=TargetMeanLevel())


def test_is_y_multiclass_flag():
    assert TschuprowtMulticlassCombinations().is_y_multiclass is True
    assert MulticlassCombinationEvaluator.is_y_multiclass is True
