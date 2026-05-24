"""Parity tests for the optional Rust kernels (SPEEDUP_PLAN.md §7).

The Rust kernels in ``AutoCarver._kernels`` (``kruskal_h_batch``,
``chi2_assoc_batch``) are wrappers over the same closed-form math as the NumPy
``_kruskal_h_batch`` / ``_chi2_assoc_batch`` paths. Output must be numerically
identical (up to FP rounding on a different summation order) to the NumPy path
across:

- random combinations on random per-modality stats,
- combinations whose partition contains singleton groups (matches the
  ``index_to_groupby`` shape produced by ``combination_formatter``),
- the ``n_groups < 2`` edge case → ``kruskal == None``,
- the ``tie_corr == 0`` edge case → ``kruskal == NaN``,
- the chi² Yates branch (combinations with exactly 2 groups).

Tests are skipped when the Rust kernel isn't installed (dev installs that
haven't run ``maturin develop``).
"""

from __future__ import annotations

import numpy as np
import pytest

from AutoCarver.combinations.binary.binary_combination_evaluators import (
    _HAVE_RUST_CHI2,
    _chi2_assoc_batch,
    _chi2_assoc_batch_rust,
)
from AutoCarver.combinations.continuous.continuous_combination_evaluators import (
    _HAVE_RUST_KRUSKAL,
    _kruskal_h_batch,
    _kruskal_h_batch_rust,
)
from AutoCarver.combinations.utils.combinations import combination_formatter, consecutive_combinations

pytestmark = pytest.mark.skipif(
    not (_HAVE_RUST_KRUSKAL and _HAVE_RUST_CHI2),
    reason="Rust _kernels extension not built (run `maturin develop` to enable)",
)


def _build_batch(n_mod: int, max_n_mod: int, size: int) -> list[dict]:
    out: list[dict] = []
    labels = list(range(n_mod))
    for c in consecutive_combinations(labels, max_n_mod):
        out.append({"combination": c, "index_to_groupby": combination_formatter(c)})
        if len(out) >= size:
            break
    return out


def _assert_aligned_none_nan(np_h: list, rust_h: list) -> None:
    """Either both None, both NaN, or both finite-equal-to-rtol."""
    assert len(np_h) == len(rust_h)
    for a, b in zip(np_h, rust_h):
        if a is None or b is None:
            assert a is None and b is None
            continue
        if np.isnan(a) or np.isnan(b):
            assert np.isnan(a) and np.isnan(b)
            continue
        assert np.isclose(a, b, rtol=1e-10, atol=1e-12), f"{a} vs {b}"


@pytest.mark.parametrize("seed", [0, 1, 42])
@pytest.mark.parametrize("max_n_mod", [3, 5, 6])
def test_kruskal_batch_parity(seed: int, max_n_mod: int) -> None:
    rng = np.random.default_rng(seed)
    n_mod = 20
    n_per_mod = rng.integers(low=100, high=500, size=n_mod).astype(np.int64)
    N = int(n_per_mod.sum())
    R_per_mod = rng.uniform(low=0, high=N * (N + 1), size=n_mod).astype(np.float64)
    tie_corr = 0.97
    mod_to_pos = {m: i for i, m in enumerate(range(n_mod))}

    batch = _build_batch(n_mod, max_n_mod, 2048)

    np_out = [
        d["kruskal"]
        for d in _kruskal_h_batch(
            batch=batch,
            R_per_mod=R_per_mod,
            n_per_mod=n_per_mod,
            N=N,
            tie_corr=tie_corr,
            mod_to_pos=mod_to_pos,
            n_mod=n_mod,
        )
    ]
    rust_out = [
        d["kruskal"]
        for d in _kruskal_h_batch_rust(
            batch=batch,
            R_per_mod=R_per_mod,
            n_per_mod=n_per_mod,
            N=N,
            tie_corr=tie_corr,
            mod_to_pos=mod_to_pos,
            n_mod=n_mod,
        )
    ]
    _assert_aligned_none_nan(np_out, rust_out)


def test_kruskal_batch_returns_none_for_singleton_partition() -> None:
    """A combination that collapses all modalities into one group must yield None
    (matches the NumPy path which yields None when n_groups < 2)."""
    n_mod = 5
    mod_to_pos = {m: i for i, m in enumerate(range(n_mod))}
    R_per_mod = np.arange(n_mod, dtype=np.float64) * 100.0
    n_per_mod = np.full(n_mod, 50, dtype=np.int64)

    # Single-group: all modalities mapped to leader 0
    combination = [list(range(n_mod))]
    batch = [{"combination": combination, "index_to_groupby": combination_formatter(combination)}]

    np_out = list(
        _kruskal_h_batch(
            batch=batch,
            R_per_mod=R_per_mod,
            n_per_mod=n_per_mod,
            N=int(n_per_mod.sum()),
            tie_corr=0.99,
            mod_to_pos=mod_to_pos,
            n_mod=n_mod,
        )
    )
    rust_out = list(
        _kruskal_h_batch_rust(
            batch=batch,
            R_per_mod=R_per_mod,
            n_per_mod=n_per_mod,
            N=int(n_per_mod.sum()),
            tie_corr=0.99,
            mod_to_pos=mod_to_pos,
            n_mod=n_mod,
        )
    )
    assert np_out[0]["kruskal"] is None
    assert rust_out[0]["kruskal"] is None


def test_kruskal_batch_tie_corr_zero_returns_nan() -> None:
    """tie_corr == 0 must yield NaN (matches scipy: H / 0 → NaN)."""
    n_mod = 4
    mod_to_pos = {m: i for i, m in enumerate(range(n_mod))}
    R_per_mod = np.array([1.0, 2.0, 3.0, 4.0])
    n_per_mod = np.array([10, 10, 10, 10], dtype=np.int64)

    combination = [[0, 1], [2, 3]]
    batch = [{"combination": combination, "index_to_groupby": combination_formatter(combination)}]

    rust_out = list(
        _kruskal_h_batch_rust(
            batch=batch,
            R_per_mod=R_per_mod,
            n_per_mod=n_per_mod,
            N=40,
            tie_corr=0.0,
            mod_to_pos=mod_to_pos,
            n_mod=n_mod,
        )
    )
    assert np.isnan(rust_out[0]["kruskal"])


def test_kruskal_batch_short_circuits_when_ranking_failed() -> None:
    """``R_per_mod is None`` or ``N < 2`` must propagate to ``None`` per combination."""
    n_mod = 4
    mod_to_pos = {m: i for i, m in enumerate(range(n_mod))}
    combination = [[0, 1], [2, 3]]
    batch = [{"combination": combination, "index_to_groupby": combination_formatter(combination)}]

    rust_out = list(
        _kruskal_h_batch_rust(
            batch=batch,
            R_per_mod=None,
            n_per_mod=np.array([0, 0, 0, 0], dtype=np.int64),
            N=0,
            tie_corr=None,
            mod_to_pos=mod_to_pos,
            n_mod=n_mod,
        )
    )
    assert rust_out[0]["kruskal"] is None


@pytest.mark.parametrize("seed", [0, 7])
@pytest.mark.parametrize("max_n_mod", [3, 5])
def test_chi2_batch_parity(seed: int, max_n_mod: int) -> None:
    rng = np.random.default_rng(seed)
    n_mod = 20
    n0 = rng.integers(low=100, high=500, size=n_mod).astype(np.float64)
    n1 = rng.integers(low=100, high=500, size=n_mod).astype(np.float64)
    n_obs = float(n0.sum() + n1.sum())
    mod_to_pos = {m: i for i, m in enumerate(range(n_mod))}
    tol = 1e-10

    batch = _build_batch(n_mod, max_n_mod, 1024)

    np_out = list(
        _chi2_assoc_batch(
            batch=batch,
            n0_per_mod=n0,
            n1_per_mod=n1,
            n_obs=n_obs,
            mod_to_pos=mod_to_pos,
            n_mod=n_mod,
            tol=tol,
        )
    )
    rust_out = list(
        _chi2_assoc_batch_rust(
            batch=batch,
            n0_per_mod=n0,
            n1_per_mod=n1,
            n_obs=n_obs,
            mod_to_pos=mod_to_pos,
            n_mod=n_mod,
            tol=tol,
        )
    )
    np_cv = np.array([d["cramerv"] for d in np_out])
    np_tt = np.array([d["tschuprowt"] for d in np_out])
    rust_cv = np.array([d["cramerv"] for d in rust_out])
    rust_tt = np.array([d["tschuprowt"] for d in rust_out])

    # Both paths quantise to tol; allow 2*tol slack.
    assert np.max(np.abs(np_cv - rust_cv)) < 2 * tol
    assert np.max(np.abs(np_tt - rust_tt)) < 2 * tol


def test_chi2_batch_yates_branch_2_groups() -> None:
    """Combinations whose partition has exactly 2 groups must receive the Yates
    correction in both Rust and NumPy paths (and produce equal values)."""
    n_mod = 4
    mod_to_pos = {m: i for i, m in enumerate(range(n_mod))}
    n0 = np.array([100.0, 80.0, 120.0, 90.0])
    n1 = np.array([60.0, 110.0, 70.0, 95.0])
    n_obs = float(n0.sum() + n1.sum())
    tol = 1e-10

    combination = [[0, 1], [2, 3]]
    batch = [{"combination": combination, "index_to_groupby": combination_formatter(combination)}]

    np_out = list(
        _chi2_assoc_batch(
            batch=batch,
            n0_per_mod=n0,
            n1_per_mod=n1,
            n_obs=n_obs,
            mod_to_pos=mod_to_pos,
            n_mod=n_mod,
            tol=tol,
        )
    )
    rust_out = list(
        _chi2_assoc_batch_rust(
            batch=batch,
            n0_per_mod=n0,
            n1_per_mod=n1,
            n_obs=n_obs,
            mod_to_pos=mod_to_pos,
            n_mod=n_mod,
            tol=tol,
        )
    )
    assert abs(np_out[0]["cramerv"] - rust_out[0]["cramerv"]) < 2 * tol
    assert abs(np_out[0]["tschuprowt"] - rust_out[0]["tschuprowt"]) < 2 * tol
