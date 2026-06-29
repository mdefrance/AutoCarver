"""Parity tests: vectorized ``compute_all`` must match scalar ``compute_association``.

Mirrors the spirit of ``tests/combinations/.../test_dp_kruskal_parity.py`` — the
scalar per-feature implementations are the reference, and the batched kernels in
``AutoCarver.selectors.measures._vectorized`` must reproduce them exactly across NaN,
tie, single-modality and multiclass edge cases.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
from pytest import approx, fixture

from AutoCarver.selectors.measures import (
    Chi2Measure,
    CramervMeasure,
    KruskalEpsilonSquaredMeasure,
    KruskalEtaSquaredMeasure,
    KruskalMeasure,
    ModeMeasure,
    NanMeasure,
    PearsonMeasure,
    SpearmanMeasure,
    TschuprowtMeasure,
)


def _features(columns):
    """Lightweight feature stand-ins: ``compute_all`` only needs ``.version``."""
    return [SimpleNamespace(version=col) for col in columns]


def _assert_parity(measure_cls, X, y, *, reversed_xy=False):
    features = _features(X.columns)

    batch = measure_cls()
    if reversed_xy:
        batch.reverse_xy()
    batch_results = batch.compute_all(X, y, features)

    for col in X.columns:
        scalar = measure_cls()
        if reversed_xy:
            scalar.reverse_xy()
        scalar_value = scalar.compute_association(X[col], y)
        batch_value = batch_results[col]["value"]

        if scalar_value is None or (isinstance(scalar_value, float) and np.isnan(scalar_value)):
            assert batch_value is None or np.isnan(batch_value), col
        else:
            assert batch_value == approx(scalar_value, rel=1e-9, abs=1e-9), col


@fixture
def rng():
    return np.random.default_rng(123)


@fixture
def quant_block(rng):
    """Quantitative features: heavy ties, NaNs in some columns, a constant column."""
    X = pd.DataFrame({f"q{j}": rng.integers(0, 8, 400).astype(float) for j in range(5)})
    X.loc[rng.random(400) < 0.15, "q1"] = np.nan
    X.loc[rng.random(400) < 0.4, "q3"] = np.nan
    X["q_const"] = 3.0  # single-value column -> nan statistic
    return X


@fixture
def quali_block(rng):
    """Qualitative features: object dtype, NaNs, a single-modality column."""
    X = pd.DataFrame({f"c{j}": rng.integers(0, 5, 400) for j in range(4)}).astype(object)
    X.loc[rng.random(400) < 0.15, "c2"] = np.nan
    X["c_const"] = "a"
    return X


@fixture
def y_multiclass(rng):
    return pd.Series(rng.integers(0, 3, 400))


@fixture
def y_binary(rng):
    return pd.Series(rng.integers(0, 2, 400))


@fixture
def y_continuous(rng):
    return pd.Series(rng.normal(size=400))


# --- quantitative-feature measures vs qualitative target -------------------


def test_kruskal_parity(quant_block, y_multiclass):
    _assert_parity(KruskalMeasure, quant_block, y_multiclass)


def test_kruskal_effect_size_parity(quant_block, y_multiclass):
    _assert_parity(KruskalEpsilonSquaredMeasure, quant_block, y_multiclass)


def test_kruskal_eta_squared_parity(quant_block, y_multiclass):
    _assert_parity(KruskalEtaSquaredMeasure, quant_block, y_multiclass)


def test_kruskal_eta_squared_binary_parity(quant_block, y_binary):
    _assert_parity(KruskalEtaSquaredMeasure, quant_block, y_binary)


# --- qualitative-feature measures vs continuous target (reversed) ----------


def test_kruskal_eta_squared_reversed_parity(quali_block, y_continuous):
    _assert_parity(KruskalEtaSquaredMeasure, quali_block, y_continuous, reversed_xy=True)


# --- quantitative-feature measures vs continuous target --------------------


def test_spearman_parity(quant_block, y_continuous):
    _assert_parity(SpearmanMeasure, quant_block, y_continuous)


def test_pearson_parity(quant_block, y_continuous):
    _assert_parity(PearsonMeasure, quant_block, y_continuous)


# --- qualitative-feature measures vs qualitative target --------------------


def test_chi2_parity(quali_block, y_multiclass):
    _assert_parity(Chi2Measure, quali_block, y_multiclass)


def test_cramerv_parity(quali_block, y_multiclass):
    _assert_parity(CramervMeasure, quali_block, y_multiclass)


def test_tschuprowt_parity(quali_block, y_multiclass):
    _assert_parity(TschuprowtMeasure, quali_block, y_multiclass)


# --- default outlier gates (Nan / Mode), quantitative + qualitative blocks ----


def test_nan_parity_quant(quant_block, y_binary):
    _assert_parity(NanMeasure, quant_block, y_binary)


def test_nan_parity_quali(quali_block, y_binary):
    _assert_parity(NanMeasure, quali_block, y_binary)


def test_mode_parity_quant(quant_block, y_binary):
    _assert_parity(ModeMeasure, quant_block, y_binary)


def test_mode_parity_quali(quali_block, y_binary):
    _assert_parity(ModeMeasure, quali_block, y_binary)


def test_mode_nan_all_nan_column():
    """Mode is nan / Nan is 1.0 for an all-NaN column, matching the scalar path."""
    X = pd.DataFrame({"ties": [1.0, 1.0, 2.0, 2.0, 3.0], "allnan": [np.nan] * 5})
    features = _features(X.columns)
    for measure_cls in (ModeMeasure, NanMeasure):
        batch = measure_cls().compute_all(X, None, features)
        for col in X.columns:
            scalar = measure_cls().compute_association(X[col], None)
            value = batch[col]["value"]
            if scalar is None or (isinstance(scalar, float) and np.isnan(scalar)):
                assert value is None or np.isnan(value), (measure_cls.__name__, col)
            else:
                assert value == approx(scalar, rel=1e-9, abs=1e-9), (measure_cls.__name__, col)
