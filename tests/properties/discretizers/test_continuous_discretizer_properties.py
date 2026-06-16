"""Property-based tests for the continuous discretizer's quantile kernel.

Source: ``AutoCarver.discretizers.quantitatives.continuous_discretizer`` —
``find_quantiles`` and the fitted ``ContinuousDiscretizer``. The quantile cut
points are sorted, finite and bounded by ``q``; after fit every quantitative
feature's value order ends at ``+inf`` with sorted finite edges.
"""

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st
from strategies import numerical_column

from AutoCarver.discretizers.quantitatives.continuous_discretizer import ContinuousDiscretizer, find_quantiles
from AutoCarver.features import Features

q_strat = st.integers(min_value=2, max_value=20)


@st.composite
def values_array(draw, *, allow_nan=True):
    """A float ``ndarray`` (heavy ties / continuous / NaN), possibly degenerate."""
    n = draw(st.integers(min_value=0, max_value=150))
    if n == 0:
        return np.array([], dtype=float)
    series = draw(numerical_column(n, ties=draw(st.booleans()), nan_rate=0.3 if allow_nan else 0.0))
    return series.to_numpy(dtype=float)


# --------------------------------------------------------------------------
# find_quantiles
# --------------------------------------------------------------------------
@given(values_array(), q_strat)
@settings(max_examples=80)
def test_find_quantiles_sorted_finite_bounded(values, q):
    """Cut points are ascending, finite, and number at most ``q`` (+1 tolerance)."""
    result = find_quantiles(values, q)
    assert result == sorted(result)
    assert all(np.isfinite(x) for x in result)
    # the kernel produces at most ~q cuts (segment quantiles + frequent values)
    assert len(result) <= q + 1


@given(q_strat)
def test_find_quantiles_all_nan_is_empty(q):
    """An all-NaN input yields no quantiles."""
    assert find_quantiles(np.array([np.nan, np.nan, np.nan]), q) == []


@given(
    st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    st.integers(min_value=1, max_value=30),
    q_strat,
)
def test_find_quantiles_constant_input(value, repeat, q):
    """A single repeated value collapses to that one cut point."""
    result = find_quantiles(np.full(repeat, value), q)
    assert result == [value]


# --------------------------------------------------------------------------
# ContinuousDiscretizer.fit
# --------------------------------------------------------------------------
@given(st.data(), st.sampled_from([0.05, 0.1, 0.2, 0.25]))
@settings(max_examples=30, deadline=None)
def test_fitted_values_end_at_inf_and_sorted(data, min_freq):
    """After fit each feature's order ends at +inf, finite edges sort ascending,
    and the number of values is bounded by ``q = round(1 / min_freq)``."""
    n = data.draw(st.integers(min_value=30, max_value=120))
    n_cols = data.draw(st.integers(min_value=1, max_value=3))
    columns = {
        f"num{j}": data.draw(numerical_column(n, ties=data.draw(st.booleans()), nan_rate=0.2)) for j in range(n_cols)
    }
    X = pd.DataFrame(columns)
    features = Features(numericals=list(columns))

    discretizer = ContinuousDiscretizer(features, min_freq)
    discretizer.fit(X)

    q = discretizer.q
    for feature in features:
        values = list(feature.values)
        assert len(values) >= 1
        # last edge is +inf
        assert np.isinf(values[-1]) and values[-1] > 0
        # finite edges are strictly the non-inf prefix and sorted ascending
        finite = values[:-1]
        assert all(np.isfinite(x) for x in finite)
        assert finite == sorted(finite)
        # bounded by the requested number of quantiles (+inf bucket, +1 tolerance)
        assert len(values) <= q + 1
