"""Parity + property tests for the ordinal rank-association evaluators.

`OrdinalCombinationEvaluator._association_measure` scores an ordered contingency
table (feature groups x ordinal target levels) with three rank statistics:
Kendall's tau-b, Stuart's tau-c and the original Somers' D ``D(Y|X)``. These
tests assert:

  * tau-b parity with ``scipy.stats.kendalltau`` and Somers' D parity with
    ``scipy.stats.somersd(table).statistic``;
  * all three match an independent brute-force concordant/discordant count;
  * degenerate tables (N < 2, single level) return ``None``;
  * the cardinality behaviour that motivated the metric choice: the symmetric
    Kendall taus self-balance to an interior number of modalities on clustered
    data (argmax k > 2), while the asymmetric Somers' D collapses to the
    coarsest split (argmax k = 2);
  * ``TargetMeanLevel`` returns the per-group mean ordinal level + frequency.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kendalltau, somersd

from AutoCarver.combinations.ordinal.ordinal_combination_evaluators import (
    KendallTauCCombinations,
    _ordinal_associations,
)
from AutoCarver.combinations.ordinal.ordinal_target_rates import TargetMeanLevel
from AutoCarver.combinations.utils.combination_evaluator import AggregatedSample


def _expand(table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Expands a contingency table to paired (x, y) observations."""
    xs, ys = [], []
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            xs += [i] * int(table[i, j])
            ys += [j] * int(table[i, j])
    return np.array(xs), np.array(ys)


def _brute_reference(table: np.ndarray) -> dict[str, float | None]:
    """Independent O(N^2) reference for tau-b, tau-c and Somers' D(Y|X)."""
    xs, ys = _expand(table)
    n = len(xs)
    if n < 2:
        return {"tau_b": None, "tau_c": None, "somersd": None}
    concordant = discordant = 0
    for a in range(n):
        for b in range(a + 1, n):
            dx = xs[a] - xs[b]
            dy = ys[a] - ys[b]
            if dx == 0 or dy == 0:
                continue
            if (dx > 0) == (dy > 0):
                concordant += 1
            else:
                discordant += 1
    cd = concordant - discordant
    n0 = n * (n - 1) / 2
    ties_x = sum(r * (r - 1) / 2 for r in table.sum(axis=1))
    ties_y = sum(c * (c - 1) / 2 for c in table.sum(axis=0))
    untied_x = n0 - ties_x
    untied_y = n0 - ties_y
    m = min(int((table.sum(axis=1) > 0).sum()), int((table.sum(axis=0) > 0).sum()))
    return {
        "tau_b": cd / math.sqrt(untied_x * untied_y) if untied_x > 0 and untied_y > 0 else None,
        "tau_c": (2 * m * cd) / (n * n * (m - 1)) if m > 1 else None,
        "somersd": cd / untied_x if untied_x > 0 else None,
    }


@pytest.mark.parametrize("seed", range(20))
def test_matches_brute_force_and_scipy(seed: int) -> None:
    """Closed form matches the brute-force reference, scipy.kendalltau and scipy.somersd."""
    rng = np.random.default_rng(seed)
    table = rng.integers(0, 5, size=(int(rng.integers(2, 7)), int(rng.integers(2, 8)))).astype(float)
    got = _ordinal_associations(table)
    ref = _brute_reference(table)
    for key in ("tau_b", "tau_c", "somersd"):
        if ref[key] is None:
            assert got[key] is None
        else:
            assert got[key] == pytest.approx(ref[key], abs=1e-9)

    # cross-check against scipy on non-degenerate tables
    xs, ys = _expand(table)
    if len(set(xs)) > 1 and len(set(ys)) > 1:
        assert got["tau_b"] == pytest.approx(kendalltau(xs, ys)[0], abs=1e-9)
        assert got["somersd"] == pytest.approx(somersd(table).statistic, abs=1e-9)


def test_evaluator_pipeline_returns_all_three() -> None:
    """`_association_measure` (via the evaluator) exposes tau_b, tau_c and somersd."""
    table = np.array([[20, 3, 0], [4, 18, 5], [0, 6, 25]], dtype=float)
    xagg = pd.DataFrame(table, index=["a", "b", "c"], columns=[1, 2, 3])
    evaluator = KendallTauCCombinations()
    evaluator.samples.train = AggregatedSample(xagg)
    measure = evaluator._association_measure(xagg)
    assert set(measure) == {"tau_b", "tau_c", "somersd"}
    assert measure["tau_b"] == pytest.approx(_brute_reference(table)["tau_b"], abs=1e-9)


def test_degenerate_tables_return_none() -> None:
    """N < 2 / empty tables are unscorable; a single target level yields no tau but 0 Somers' D."""
    # fewer than two observations -> nothing is scorable
    assert _ordinal_associations(np.array([[1.0, 0.0]])) == {"tau_b": None, "tau_c": None, "somersd": None}
    assert _ordinal_associations(np.zeros((3, 3))) == {"tau_b": None, "tau_c": None, "somersd": None}
    # a single target level: tau-b/tau-c denominators vanish (None); Somers' D(Y|X) is a
    # well-defined 0.0 (pairs differ on the feature, none on the target -> no concordance)
    single_level = _ordinal_associations(np.array([[5.0], [3.0]]))
    assert single_level["tau_b"] is None
    assert single_level["tau_c"] is None
    assert single_level["somersd"] == 0.0


def _best_per_k(raw_table: np.ndarray, key: str, kmax: int = 8) -> dict[int, float]:
    """Best score per number of consecutive row groups, for a given measure."""
    m = raw_table.shape[0]
    out: dict[int, float] = {}
    for k in range(2, min(kmax, m) + 1):
        best = -np.inf
        for cuts in combinations(range(1, m), k - 1):
            bounds = [0, *cuts, m]
            grouped = np.array([raw_table[bounds[i] : bounds[i + 1]].sum(axis=0) for i in range(k)])
            value = _ordinal_associations(grouped)[key]
            if value is not None and value > best:
                best = value
        out[k] = best
    return out


def test_kendall_self_balances_somersd_collapses() -> None:
    """The defining property: Kendall taus peak at an interior k; Somers' D peaks at k=2."""
    clustered = np.array(
        [
            [40, 5, 0, 0, 0],
            [38, 6, 1, 0, 0],
            [42, 4, 0, 0, 0],
            [0, 5, 40, 5, 0],
            [0, 4, 42, 4, 0],
            [0, 0, 1, 5, 40],
            [0, 0, 0, 6, 38],
            [0, 0, 0, 4, 42],
        ],
        dtype=float,
    )
    tau_c = _best_per_k(clustered, "tau_c")
    tau_b = _best_per_k(clustered, "tau_b")
    somersd = _best_per_k(clustered, "somersd")
    assert max(tau_c, key=tau_c.get) > 2  # type: ignore[arg-type]
    assert max(tau_b, key=tau_b.get) > 2  # type: ignore[arg-type]
    assert max(somersd, key=somersd.get) == 2  # type: ignore[arg-type]


def test_target_mean_level() -> None:
    """TargetMeanLevel returns per-group mean ordinal level, frequency and count."""
    xagg = pd.DataFrame({1: [5, 0, 0], 2: [1, 4, 0], 3: [0, 1, 6]}, index=["a", "b", "c"])
    rates = TargetMeanLevel().compute(xagg)
    assert list(rates.columns) == ["target_mean_level", "frequency", "count"]
    assert rates.loc["a", "target_mean_level"] == pytest.approx((5 * 1 + 1 * 2) / 6)
    assert rates.loc["b", "target_mean_level"] == pytest.approx((4 * 2 + 1 * 3) / 5)
    assert rates.loc["c", "target_mean_level"] == pytest.approx(3.0)
    assert rates["count"].tolist() == [6, 5, 6]
    assert rates["frequency"].sum() == pytest.approx(1.0)
