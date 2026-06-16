"""Property-based tests for the association measures.

Source: ``AutoCarver.selectors.measures``. Generalizes the fixed-data parity
tests (``tests/selectors/test_vectorized_parity.py``) with hypothesis and adds
value-range, invariance and ``validate`` contract properties. The scalar
``compute_association`` stays the reference; the batched ``compute_all`` must
reproduce it column-by-column.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from strategies import categorical_column, continuous_floats, numerical_column

from AutoCarver.selectors.measures import (
    Chi2Measure,
    CramervMeasure,
    KruskalEpsilonSquaredMeasure,
    KruskalEtaSquaredMeasure,
    KruskalMeasure,
    ModeMeasure,
    NanMeasure,
    PearsonMeasure,
    RMeasure,
    SpearmanMeasure,
    TschuprowtMeasure,
)

nrows = st.integers(min_value=5, max_value=60)


def _features(columns):
    return [SimpleNamespace(version=col) for col in columns]


@st.composite
def quant_block(draw, *, min_cols=1, max_cols=3, nan=True):
    n = draw(nrows)
    p = draw(st.integers(min_value=min_cols, max_value=max_cols))
    nan_rate = 0.2 if nan else 0.0
    cols = {f"q{j}": draw(numerical_column(n, ties=draw(st.booleans()), nan_rate=nan_rate)) for j in range(p)}
    return pd.DataFrame(cols)


@st.composite
def quali_block(draw, *, min_cols=1, max_cols=3, nan=True):
    n = draw(nrows)
    p = draw(st.integers(min_value=min_cols, max_value=max_cols))
    nan_rate = 0.2 if nan else 0.0
    cols = {
        f"c{j}": draw(categorical_column(n, cardinality=draw(st.integers(1, 4)), nan_rate=nan_rate)) for j in range(p)
    }
    return pd.DataFrame(cols)


def _is_nan(value) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


def _same(a, b) -> bool:
    """Equality that treats two NaNs as equal (an all-NaN x yields a NaN mode)."""
    return (_is_nan(a) and _is_nan(b)) or a == b


# --------------------------------------------------------------------------
# Vectorized <-> scalar parity
# --------------------------------------------------------------------------
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

        if _is_nan(scalar_value):
            assert _is_nan(batch_value), col
        else:
            assert batch_value == pytest.approx(scalar_value, rel=1e-9, abs=1e-9), col


@given(st.data())
@settings(max_examples=40)
def test_kruskal_parity(data):
    """KruskalMeasure (and effect sizes) batch == scalar, quanti x vs quali y."""
    block = data.draw(quant_block())
    y = data.draw(categorical_column(block.shape[0], cardinality=data.draw(st.integers(2, 4)), nan_rate=0.0))
    for cls in (KruskalMeasure, KruskalEpsilonSquaredMeasure, KruskalEtaSquaredMeasure):
        _assert_parity(cls, block, y)


@given(st.data())
@settings(max_examples=40)
def test_kruskal_reversed_parity(data):
    """KruskalEtaSquared batch == scalar in reversed mode: quali x vs continuous y."""
    block = data.draw(quali_block())
    y = pd.Series(data.draw(st.lists(continuous_floats, min_size=block.shape[0], max_size=block.shape[0])))
    _assert_parity(KruskalEtaSquaredMeasure, block, y, reversed_xy=True)


@given(st.data())
@settings(max_examples=40)
def test_correlation_parity(data):
    """Pearson/Spearman batch == scalar, quanti x vs continuous y."""
    block = data.draw(quant_block())
    y = pd.Series(data.draw(st.lists(continuous_floats, min_size=block.shape[0], max_size=block.shape[0])))
    for cls in (PearsonMeasure, SpearmanMeasure):
        _assert_parity(cls, block, y)


@st.composite
def multi_quali_y(draw, n):
    """A qualitative target with at least three distinct classes (no NaN).

    Forcing >=3 distinct ``y`` values keeps every contingency table wider than
    2x2, so ``scipy.stats.chi2_contingency`` does *not* apply Yates' continuity
    correction — the regime where the vectorized kernel matches the scalar (see
    :func:`test_chi2_2x2_parity`).
    """
    data = draw(st.lists(st.sampled_from(["p", "q", "r", "s"]), min_size=n, max_size=n))
    data[0], data[1], data[2] = "p", "q", "r"
    return pd.Series(data, dtype=object)


@given(st.data())
@settings(max_examples=25)
def test_chi2_family_parity(data):
    """Chi2/Cramerv/Tschuprowt batch == scalar for non-2x2 tables (quali x, >=3-class y)."""
    n = data.draw(nrows)
    p = data.draw(st.integers(1, 3))
    # x without NaN so the non-null overlap keeps y's >=3 classes (-> table never 2x2)
    block = pd.DataFrame(
        {
            f"c{j}": data.draw(categorical_column(n, cardinality=data.draw(st.integers(1, 4)), nan_rate=0.0))
            for j in range(p)
        }
    )
    y = data.draw(multi_quali_y(n))
    for cls in (Chi2Measure, CramervMeasure, TschuprowtMeasure):
        _assert_parity(cls, block, y)


@given(st.data())
@settings(max_examples=25)
def test_chi2_2x2_parity(data):
    """Binary x vs binary y: vectorized Chi2 matches the Yates-corrected scalar.

    Regression guard for the previously-missing Yates continuity correction in
    ``chi2_all`` (now applied on 2x2 tables, matching ``scipy.stats.chi2_contingency``).
    """
    n = data.draw(st.integers(20, 60))
    x = pd.Series(
        ["a", "b"] + data.draw(st.lists(st.sampled_from(["a", "b"]), min_size=n - 2, max_size=n - 2)), dtype=object
    )
    y = pd.Series([0, 1] + data.draw(st.lists(st.integers(0, 1), min_size=n - 2, max_size=n - 2)))
    _assert_parity(Chi2Measure, pd.DataFrame({"c0": x}), y)


# --------------------------------------------------------------------------
# Value ranges
# --------------------------------------------------------------------------
@st.composite
def quali_xy(draw):
    """A non-degenerate (x, y) qualitative pair: both with >=2 distinct values."""
    n = draw(st.integers(min_value=20, max_value=60))
    x = draw(categorical_column(n, cardinality=draw(st.integers(2, 4)), nan_rate=0.0))
    y = draw(categorical_column(n, cardinality=draw(st.integers(2, 4)), nan_rate=0.0))
    return x, y


@given(quali_xy())
@settings(max_examples=40)
def test_cramerv_tschuprowt_unit_interval(xy):
    """Cramér's V and Tschuprow's T lie in [0, 1] for non-degenerate inputs."""
    x, y = xy
    for cls in (CramervMeasure, TschuprowtMeasure):
        value = cls().compute_association(x, y)
        assert -1e-9 <= value <= 1.0 + 1e-9


@given(st.data())
@settings(max_examples=40)
def test_chi2_non_negative(data):
    """Raw Chi2 statistic is non-negative."""
    n = data.draw(st.integers(20, 60))
    x = data.draw(categorical_column(n, cardinality=data.draw(st.integers(2, 4)), nan_rate=0.0))
    y = data.draw(categorical_column(n, cardinality=data.draw(st.integers(2, 4)), nan_rate=0.0))
    assert Chi2Measure().compute_association(x, y) >= -1e-9


@given(st.data())
@settings(max_examples=40)
def test_kruskal_effect_sizes_unit_interval(data):
    """Kruskal epsilon²/eta² effect sizes lie in [0, 1] where defined."""
    n = data.draw(st.integers(20, 60))
    x = data.draw(numerical_column(n, ties=data.draw(st.booleans()), nan_rate=0.0))
    y = data.draw(categorical_column(n, cardinality=data.draw(st.integers(2, 4)), nan_rate=0.0))
    for cls in (KruskalEpsilonSquaredMeasure, KruskalEtaSquaredMeasure):
        value = cls().compute_association(x, y)
        if not _is_nan(value):
            assert -1e-9 <= value <= 1.0 + 1e-9


@given(st.data())
@settings(max_examples=40)
def test_correlation_bounded_by_one(data):
    """|Pearson| and |Spearman| are at most 1 where defined."""
    n = data.draw(st.integers(20, 60))
    x = data.draw(numerical_column(n, ties=data.draw(st.booleans()), nan_rate=0.0))
    y = pd.Series(data.draw(st.lists(continuous_floats, min_size=n, max_size=n)))
    for cls in (PearsonMeasure, SpearmanMeasure):
        value = cls().compute_association(x, y)
        if not _is_nan(value):
            assert abs(value) <= 1.0 + 1e-9


@given(st.data())
@settings(max_examples=40)
def test_rmeasure_unit_interval(data):
    """RMeasure (sqrt R²) lies in [0, 1] for a binary target."""
    n = data.draw(st.integers(20, 60))
    x = data.draw(numerical_column(n, ties=data.draw(st.booleans()), nan_rate=0.0))
    y = pd.Series([0, 1] + data.draw(st.lists(st.integers(0, 1), min_size=n - 2, max_size=n - 2)))
    value = RMeasure().compute_association(x, y)
    if not _is_nan(value):
        assert -1e-9 <= value <= 1.0 + 1e-9


@given(st.data())
@settings(max_examples=40)
def test_default_measures_unit_interval(data):
    """NanMeasure and ModeMeasure are fractions in [0, 1] (column has some data)."""
    n = data.draw(st.integers(5, 60))
    x = data.draw(categorical_column(n, cardinality=data.draw(st.integers(1, 4)), nan_rate=0.3))
    assume(x.notna().any())  # ModeMeasure is undefined on an all-NaN column (see below)
    for cls in (NanMeasure, ModeMeasure):
        value = cls().compute_association(x)
        assert 0.0 <= value <= 1.0


@given(st.integers(min_value=1, max_value=30))
def test_mode_measure_all_nan_is_defined(repeat):
    """ModeMeasure on an all-NaN column yields a defined value (NaN), not a crash."""
    x = pd.Series([None] * repeat, dtype=object)
    value = ModeMeasure().compute_association(x)
    assert _is_nan(value) or 0.0 <= value <= 1.0


# --------------------------------------------------------------------------
# Invariances
# --------------------------------------------------------------------------
@given(
    st.data(),
    st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=40)
def test_pearson_affine_invariant(data, a, b, c, d):
    """Pearson's r is invariant under positive affine transforms of x and/or y."""
    n = data.draw(st.integers(20, 60))
    x = data.draw(numerical_column(n, ties=False, nan_rate=0.0))
    y = pd.Series(data.draw(st.lists(continuous_floats, min_size=n, max_size=n)))

    base = PearsonMeasure().compute_association(x, y)
    transformed = PearsonMeasure().compute_association(a * x + b, c * y + d)
    if _is_nan(base):
        assert _is_nan(transformed)
    else:
        assert transformed == pytest.approx(base, rel=1e-6, abs=1e-6)


@given(st.data())
@settings(max_examples=40)
def test_self_correlation_is_one(data):
    """x == y implies Pearson and Spearman equal 1.0 (when x is non-constant)."""
    n = data.draw(st.integers(20, 60))
    x = data.draw(numerical_column(n, ties=data.draw(st.booleans()), nan_rate=0.0))
    if x.nunique() <= 1:
        return  # constant -> undefined, covered elsewhere
    for cls in (PearsonMeasure, SpearmanMeasure):
        assert cls().compute_association(x, x) == pytest.approx(1.0, abs=1e-9)


@given(st.data())
@settings(max_examples=30)
def test_constant_feature_is_nan(data):
    """A constant feature has no defined Pearson/Spearman correlation (NaN)."""
    n = data.draw(st.integers(20, 60))
    x = data.draw(numerical_column(n, constant=True))
    y = pd.Series(data.draw(st.lists(continuous_floats, min_size=n, max_size=n)))
    for cls in (PearsonMeasure, SpearmanMeasure):
        assert _is_nan(cls().compute_association(x, y))


@given(st.data())
@settings(max_examples=30)
def test_default_measures_ignore_y(data):
    """NanMeasure/ModeMeasure depend only on x; NanMeasure == observed NaN fraction."""
    n = data.draw(st.integers(5, 60))
    x = data.draw(categorical_column(n, cardinality=data.draw(st.integers(1, 4)), nan_rate=0.3))
    y1 = pd.Series(data.draw(st.lists(st.integers(0, 1), min_size=n, max_size=n)))
    y2 = pd.Series(data.draw(st.lists(st.integers(0, 5), min_size=n, max_size=n)))

    assert _same(NanMeasure().compute_association(x, y1), NanMeasure().compute_association(x, y2))
    assert _same(ModeMeasure().compute_association(x, y1), ModeMeasure().compute_association(x, y2))
    assert NanMeasure().compute_association(x) == pytest.approx(x.isna().mean())


# --------------------------------------------------------------------------
# validate() contract
# --------------------------------------------------------------------------
@given(
    st.floats(min_value=-2, max_value=2, allow_nan=False),
    st.floats(min_value=0, max_value=1, allow_nan=False),
)
def test_validate_threshold_contract(value, threshold):
    """validate() is value>=threshold (non-absolute) / |value|>=threshold (absolute)."""
    measure = CramervMeasure(threshold)
    measure.value = value
    assert measure.validate() == (value >= threshold)

    absolute = PearsonMeasure(threshold)
    absolute.value = value
    assert absolute.validate() == (abs(value) >= threshold)


@given(st.sampled_from([None, float("nan")]))
def test_validate_false_when_undefined(bad_value):
    """A None/NaN value never validates (for sortable association measures)."""
    measure = CramervMeasure(0.0)
    measure.value = bad_value
    assert measure.validate() is False


# --------------------------------------------------------------------------
# reverse_xy()
# --------------------------------------------------------------------------
def test_reverse_xy_toggles_type_flags():
    """reverse_xy() swaps the x/y qualitative/quantitative orientation flags."""
    measure = KruskalMeasure()
    assert measure.is_x_quantitative and measure.is_y_qualitative

    assert measure.reverse_xy() is True
    assert measure.is_x_qualitative and not measure.is_x_quantitative
    assert measure.is_y_quantitative and not measure.is_y_qualitative


def test_reverse_xy_noop_on_non_reversible():
    """A non-reversible measure reports reverse_xy() as a no-op (False)."""
    assert CramervMeasure().reverse_xy() is False
