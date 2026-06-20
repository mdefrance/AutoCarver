"""Property-based tests for the ordinal rank-association evaluators and their DP.

Source: ``combinations/ordinal/ordinal_combination_evaluators.py``. The
seed-parametrised parity suite lives in
``tests/combinations/ordinal/test_ordinal_associations.py``; this module adds
*property* coverage over hypothesis-generated contingency tables:

  * ``_ordinal_associations`` matches an independent brute-force pair count and
    ``scipy`` (tau-b / Somers' D) on every non-degenerate table;
  * the three statistics stay within ``[-1, 1]`` and negate when the target
    order is reversed (a pure orientation flip);
  * ``_concordant_minus_discordant`` matches the brute-force ``C - D``;
  * the interval DP recovers the brute-force-best consecutive partition: exactly
    for tau-c with any ``top_k`` (per-k constant denominator), and for all three
    metrics once ``top_k`` is exhaustive.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from scipy.stats import kendalltau, somersd

from AutoCarver.combinations.ordinal.ordinal_combination_evaluators import (
    _concordant_minus_discordant,
    _ordinal_associations,
    _top_k_partitions_ordinal_dp,
)

SETTINGS = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])

METRICS = ("tau_b", "tau_c", "somersd")


@st.composite
def ordinal_count_table(draw, *, max_rows: int = 6, max_cols: int = 5, max_count: int = 6) -> np.ndarray:
    """A ``(rows, cols)`` non-negative integer contingency table (rows = feature
    groups, cols = ordinal target levels, both ascending)."""
    rows = draw(st.integers(min_value=2, max_value=max_rows))
    cols = draw(st.integers(min_value=2, max_value=max_cols))
    cells = draw(st.lists(st.integers(min_value=0, max_value=max_count), min_size=rows * cols, max_size=rows * cols))
    return np.array(cells, dtype=float).reshape(rows, cols)


# --------------------------------------------------------------------------
# helpers: independent brute-force references
# --------------------------------------------------------------------------
def _expand(table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Expands a contingency table to paired ``(x, y)`` observations."""
    xs, ys = [], []
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            xs += [i] * int(table[i, j])
            ys += [j] * int(table[i, j])
    return np.array(xs), np.array(ys)


def _brute_cd(table: np.ndarray) -> float:
    """Independent ``O(N^2)`` concordant-minus-discordant count."""
    xs, ys = _expand(table)
    n = len(xs)
    cd = 0
    for a in range(n):
        for b in range(a + 1, n):
            dx = xs[a] - xs[b]
            dy = ys[a] - ys[b]
            if dx == 0 or dy == 0:
                continue
            cd += 1 if (dx > 0) == (dy > 0) else -1
    return float(cd)


def _brute_associations(table: np.ndarray) -> dict[str, float | None]:
    """Independent reference for tau-b, tau-c and Somers' D(Y|X)."""
    xs, _ = _expand(table)
    n = len(xs)
    if n < 2:
        return {"tau_b": None, "tau_c": None, "somersd": None}
    cd = _brute_cd(table)
    n0 = n * (n - 1) / 2
    untied_x = n0 - sum(r * (r - 1) / 2 for r in table.sum(axis=1))
    untied_y = n0 - sum(c * (c - 1) / 2 for c in table.sum(axis=0))
    m = min(int((table.sum(axis=1) > 0).sum()), int((table.sum(axis=0) > 0).sum()))
    return {
        "tau_b": cd / math.sqrt(untied_x * untied_y) if untied_x > 0 and untied_y > 0 else None,
        "tau_c": (2 * m * cd) / (n * n * (m - 1)) if m > 1 else None,
        "somersd": cd / untied_x if untied_x > 0 else None,
    }


def _brute_best_partition(table: np.ndarray, sort_by: str, max_n_mod: int) -> float | None:
    """Best ``sort_by`` over every consecutive row grouping with ``k <= max_n_mod``."""
    n_mod = table.shape[0]
    best: float | None = None
    for k in range(2, min(max_n_mod, n_mod) + 1):
        for cuts in combinations(range(1, n_mod), k - 1):
            bounds = [0, *cuts, n_mod]
            grouped = np.array([table[bounds[i] : bounds[i + 1]].sum(axis=0) for i in range(k)])
            value = _ordinal_associations(grouped)[sort_by]
            if value is not None and (best is None or value > best):
                best = value
    return best


# --------------------------------------------------------------------------
# closed-form association == brute force / scipy
# --------------------------------------------------------------------------
@given(ordinal_count_table())
@SETTINGS
def test_associations_match_brute_force(table):
    """The closed form equals the independent brute-force pair count."""
    got = _ordinal_associations(table)
    ref = _brute_associations(table)
    for key in METRICS:
        if ref[key] is None:
            assert got[key] is None
        else:
            assert got[key] is not None
            assert abs(got[key] - ref[key]) < 1e-9


@given(ordinal_count_table())
@SETTINGS
def test_concordant_minus_discordant_matches_brute_force(table):
    """``_concordant_minus_discordant`` equals the O(N^2) reference."""
    assert abs(_concordant_minus_discordant(table) - _brute_cd(table)) < 1e-9


@given(ordinal_count_table())
@SETTINGS
def test_tau_b_and_somersd_match_scipy(table):
    """tau-b matches ``scipy.stats.kendalltau``; Somers' D matches ``scipy.stats.somersd``."""
    xs, ys = _expand(table)
    assume(len(set(xs)) > 1 and len(set(ys)) > 1)  # both margins non-degenerate
    got = _ordinal_associations(table)
    assert got["tau_b"] == _approx(kendalltau(xs, ys)[0])
    assert got["somersd"] == _approx(somersd(table).statistic)


# --------------------------------------------------------------------------
# structural invariants
# --------------------------------------------------------------------------
@given(ordinal_count_table())
@SETTINGS
def test_metrics_within_unit_interval(table):
    """Every defined statistic lies in ``[-1, 1]``."""
    got = _ordinal_associations(table)
    for key in METRICS:
        if got[key] is not None:
            assert -1 - 1e-9 <= got[key] <= 1 + 1e-9, (key, got[key])


@given(ordinal_count_table())
@SETTINGS
def test_reversing_target_order_negates_metrics(table):
    """Reversing the ordinal target columns is a pure orientation flip: every
    statistic negates (denominators, which depend only on the margins, are
    unchanged)."""
    got = _ordinal_associations(table)
    reversed_got = _ordinal_associations(table[:, ::-1])
    for key in METRICS:
        if got[key] is None:
            assert reversed_got[key] is None
        else:
            assert reversed_got[key] == _approx(-got[key])


# --------------------------------------------------------------------------
# the interval DP recovers the brute-force optimum
# --------------------------------------------------------------------------
@given(ordinal_count_table(), st.integers(min_value=2, max_value=6))
@SETTINGS
def test_dp_recovers_brute_force_best_when_exhaustive(table, max_n_mod):
    """With an exhaustive ``top_k`` the DP's best partition matches the brute-force
    best for every metric."""
    for sort_by in METRICS:
        brute = _brute_best_partition(table, sort_by, max_n_mod)
        assume(brute is not None)
        dp = _dp_best(table, sort_by, max_n_mod, top_k=1000)
        assert dp is not None
        assert dp == _approx(brute)


@given(ordinal_count_table(), st.integers(min_value=2, max_value=6))
@SETTINGS
def test_dp_tau_c_exact_with_minimal_top_k(table, max_n_mod):
    """tau-c has a per-k constant denominator, so the additive-numerator DP is
    exact even when only the single best prefix is kept per cell (``top_k=1``)."""
    brute = _brute_best_partition(table, "tau_c", max_n_mod)
    assume(brute is not None)
    dp = _dp_best(table, "tau_c", max_n_mod, top_k=1)
    assert dp is not None
    assert dp == _approx(brute)


# --------------------------------------------------------------------------
# utilities
# --------------------------------------------------------------------------
def _approx(value: float):
    """Symmetric float comparator with a small absolute tolerance."""

    class _A:
        def __eq__(self, other):
            return abs(other - value) < 1e-9

        def __repr__(self):
            return f"~{value}"

    return _A()


def _dp_best(table: np.ndarray, sort_by: str, max_n_mod: int, *, top_k: int) -> float | None:
    """Top ``sort_by`` value the DP returns for ``table`` (None if it returns nothing)."""
    raw_index = list(range(table.shape[0]))
    entries = _top_k_partitions_ordinal_dp(
        table,
        table.sum(axis=1),
        table.sum(axis=0),
        max_n_mod=max_n_mod,
        raw_index=raw_index,
        sort_by=sort_by,
        top_k=top_k,
    )
    valid = [e[sort_by] for e in entries if e[sort_by] is not None]
    return max(valid) if valid else None
