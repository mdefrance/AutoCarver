"""Property-based tests for the multiclass chi²-family evaluators and their DP.

Source: ``combinations/multiclass/multiclass_combination_evaluators.py``. The
seed-parametrised parity suite lives in
``tests/combinations/multiclass/test_chi2_closed_form_multiclass.py`` and
``test_dp_chi2_parity_multiclass.py``; this module adds *property* coverage
over hypothesis-generated ``(B, K)`` count tables:

  * Cramér's V and Tschuprow's T stay within their mathematically-guaranteed
    ``[0, 1]`` range on every table;
  * the interval DP recovers the brute-force-best consecutive partition once
    ``top_k`` is exhaustive, for both metrics and K in {2, 3, 5}.
"""

from itertools import combinations

import numpy as np
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from AutoCarver.combinations.multiclass.multiclass_combination_evaluators import (
    _chi2_pearson,
    _cramerv_tschuprowt,
    _top_k_partitions_chi2_dp_multiclass,
)

SETTINGS = settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])

METRICS = ("cramerv", "tschuprowt")


@st.composite
def multiclass_count_table(draw, *, max_rows: int = 6, max_cols: int = 5, max_count: int = 8) -> np.ndarray:
    """A ``(rows, cols)`` non-negative integer contingency table (rows = feature
    modalities, cols = unordered target classes)."""
    rows = draw(st.integers(min_value=2, max_value=max_rows))
    cols = draw(st.integers(min_value=2, max_value=max_cols))
    cells = draw(st.lists(st.integers(min_value=0, max_value=max_count), min_size=rows * cols, max_size=rows * cols))
    return np.array(cells, dtype=float).reshape(rows, cols)


def _brute_best_partition(table: np.ndarray, sort_by: str, max_n_mod: int) -> float | None:
    """Best ``sort_by`` over every consecutive row grouping with ``k <= max_n_mod``."""
    n_mod = table.shape[0]
    best: float | None = None
    for k in range(2, min(max_n_mod, n_mod) + 1):
        for cuts in combinations(range(1, n_mod), k - 1):
            bounds = [0, *cuts, n_mod]
            grouped = np.array([table[bounds[i] : bounds[i + 1]].sum(axis=0) for i in range(k)])
            chi2 = _chi2_pearson(grouped + 1e-10)
            cv, tt = _cramerv_tschuprowt(chi2, float(table.sum()), k, table.shape[1], 1e-10)
            value = tt if sort_by == "tschuprowt" else cv
            if value is not None and not np.isnan(value) and (best is None or value > best):
                best = value
    return best


def _dp_best(table: np.ndarray, sort_by: str, max_n_mod: int, *, top_k: int) -> float | None:
    raw_index = list(range(table.shape[0]))
    n_per_mod = table.sum(axis=1)
    col_sums = table.sum(axis=0)
    entries = _top_k_partitions_chi2_dp_multiclass(
        table, n_per_mod, col_sums, max_n_mod=max_n_mod, raw_index=raw_index, sort_by=sort_by, top_k=top_k
    )
    valid = [e[sort_by] for e in entries if e[sort_by] is not None and not np.isnan(e[sort_by])]
    return max(valid) if valid else None


# --------------------------------------------------------------------------
# structural invariants
# --------------------------------------------------------------------------
@given(multiclass_count_table())
@SETTINGS
def test_metrics_within_zero_one_interval(table):
    """Cramér's V and Tschuprow's T are mathematically bounded in [0, 1]."""
    n_groups, n_classes = table.shape
    chi2 = _chi2_pearson(table + 1e-10)
    cv, tt = _cramerv_tschuprowt(chi2, float(table.sum()), n_groups, n_classes, 1e-10)
    for value in (cv, tt):
        if value is not None and not np.isnan(value):
            assert -1e-6 <= value <= 1 + 1e-6, value


# --------------------------------------------------------------------------
# the interval DP recovers the brute-force optimum
# --------------------------------------------------------------------------
@given(multiclass_count_table(), st.integers(min_value=2, max_value=6))
@SETTINGS
def test_dp_recovers_brute_force_best_when_exhaustive(table, max_n_mod):
    """With an exhaustive ``top_k`` the DP's best partition matches the
    brute-force best for both metrics.

    Tables with an all-zero row (a modality with zero observations) are
    excluded: the DP deliberately never lets an empty modality stand as its
    own group (see :func:`test_dp_empty_modality_is_compacted`), so a naive
    brute force that *does* allow it is not a valid reference there.
    """
    assume(bool(np.all(table.sum(axis=1) > 0)))
    for sort_by in METRICS:
        brute = _brute_best_partition(table, sort_by, max_n_mod)
        assume(brute is not None)
        dp = _dp_best(table, sort_by, max_n_mod, top_k=2000)
        assert dp is not None
        assert abs(dp - brute) < 1e-8
