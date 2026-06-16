"""Property-based tests for the vectorized association kernels.

Source: ``AutoCarver.selectors.measures._vectorized`` — ``one_hot``,
``_tie_factors``, ``kruskal_h`` and ``chi2_all``. These check structural
invariants (shapes, NaN placement, value ranges) that must hold for every
generated block, independent of the scalar reference implementations (parity
with those lives in ``test_measures_properties.py``).
"""

import numpy as np
import pandas as pd
from hypothesis import given
from hypothesis import strategies as st
from strategies import categorical_column, numerical_column

from AutoCarver.selectors.measures._vectorized import _tie_factors, chi2_all, kruskal_h, one_hot

nrows = st.integers(min_value=2, max_value=60)


@st.composite
def numeric_block(draw, *, min_cols: int = 1, max_cols: int = 4, nan: bool = False):
    """A quantitative block ``(N, P)`` of float columns (heavy ties allowed)."""
    n = draw(nrows)
    p = draw(st.integers(min_value=min_cols, max_value=max_cols))
    nan_rate = 0.2 if nan else 0.0
    cols = {f"q{j}": draw(numerical_column(n, ties=draw(st.booleans()), nan_rate=nan_rate)) for j in range(p)}
    return pd.DataFrame(cols)


@st.composite
def quali_block(draw, *, min_cols: int = 1, max_cols: int = 4):
    """A qualitative (object) block ``(N, P)``."""
    n = draw(nrows)
    p = draw(st.integers(min_value=min_cols, max_value=max_cols))
    cols = {f"c{j}": draw(categorical_column(n, nan_rate=0.2)) for j in range(p)}
    return pd.DataFrame(cols)


# --------------------------------------------------------------------------
# one_hot
# --------------------------------------------------------------------------
@given(st.data())
def test_one_hot_shape_and_rows(data):
    """``one_hot`` returns an ``(N, K)`` one-hot matrix; NaN rows are all-zero,
    non-NaN rows sum to exactly 1, and ``K`` equals the distinct non-null count."""
    n = data.draw(nrows)
    groups = data.draw(categorical_column(n, nan_rate=0.3))

    matrix, k = one_hot(groups)

    assert matrix.shape == (n, k)
    assert k == groups.dropna().nunique()

    row_sums = matrix.sum(axis=1)
    na_mask = groups.isna().to_numpy()
    # NaN rows -> all-zero; populated rows -> exactly one hot
    assert np.allclose(row_sums[na_mask], 0.0)
    assert np.allclose(row_sums[~na_mask], 1.0)
    # entries are strictly 0/1
    assert set(np.unique(matrix)).issubset({0.0, 1.0})


# --------------------------------------------------------------------------
# _tie_factors
# --------------------------------------------------------------------------
@given(numeric_block(nan=True))
def test_tie_factors_length_and_range(block):
    """One factor per column, each in ``[0, 1]`` (1 = tie-free, 0 = fully tied)."""
    factors = _tie_factors(block)
    assert factors.shape == (block.shape[1],)
    assert np.all(factors >= -1e-12)
    assert np.all(factors <= 1.0 + 1e-12)


@given(st.integers(min_value=2, max_value=40), st.integers(min_value=2, max_value=40))
def test_tie_factors_tie_free_is_one(n, m):
    """A column of all-distinct values (no ties) yields a tie factor of 1.0."""
    # n distinct values guarantees no ties
    block = pd.DataFrame({"q": np.arange(n, dtype=float)})
    factors = _tie_factors(block)
    assert factors[0] == 1.0
    _ = m


# --------------------------------------------------------------------------
# kruskal_h
# --------------------------------------------------------------------------
@given(st.data())
def test_kruskal_h_structure(data):
    """``kruskal_h`` lengths/observation counts match; defined H is non-negative
    and only defined when both ``n_obs > 1`` and ``n_groups > 1``."""
    block = data.draw(numeric_block(nan=True))
    n = block.shape[0]
    # the kernel assumes the grouping target carries no missing values
    groups = data.draw(categorical_column(n, nan_rate=0.0))

    h, n_obs, n_groups = kruskal_h(block, groups)
    p = block.shape[1]
    assert h.shape == (p,) and n_obs.shape == (p,) and n_groups.shape == (p,)

    for j, col in enumerate(block.columns):
        assert n_obs[j] == block[col].notna().sum()
        # undefined (NaN) whenever fewer than 2 obs or fewer than 2 groups
        if n_obs[j] <= 1 or n_groups[j] <= 1:
            assert np.isnan(h[j])
        # where defined, H is non-negative and the guards held
        if not np.isnan(h[j]):
            assert h[j] >= -1e-9
            assert n_obs[j] > 1 and n_groups[j] > 1


# --------------------------------------------------------------------------
# chi2_all
# --------------------------------------------------------------------------
@given(st.data())
def test_chi2_all_structure(data):
    """``chi2_all`` lengths match; chi² is non-negative where defined; ``n_mod_y``
    is the (constant) distinct-``y`` count and ``n_obs`` the pairwise non-NaN count."""
    block = data.draw(quali_block())
    n = block.shape[0]
    y = data.draw(categorical_column(n, cardinality=data.draw(st.integers(1, 4)), nan_rate=0.2))

    chi2, n_obs, n_mod_x, n_mod_y = chi2_all(block, y)
    p = block.shape[1]
    assert chi2.shape == (p,) and n_obs.shape == (p,)
    assert n_mod_x.shape == (p,) and n_mod_y.shape == (p,)

    expected_n_y = y.dropna().nunique()
    assert np.all(n_mod_y == float(expected_n_y))

    for j, col in enumerate(block.columns):
        both = block[col].notna().to_numpy() & y.notna().to_numpy()
        assert n_obs[j] == both.sum()
        assert n_mod_x[j] == block[col].dropna().nunique()
        if not np.isnan(chi2[j]):
            assert chi2[j] >= -1e-9
