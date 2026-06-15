"""Property-based tests for nan_unique."""

import numpy as np
import pandas as pd
from hypothesis import given
from hypothesis import strategies as st

from AutoCarver.features.qualitatives.qualitative_feature import nan_unique

value_lists = st.lists(
    st.sampled_from(["a", "b", "c", "d", "e", "f"]),
    min_size=0,
    max_size=30,
)


@given(value_lists, st.booleans())
def test_nan_unique_no_nan_and_unique(values, sort):
    """Result contains only non-NaN values, with no duplicates, all from input."""
    series = pd.Series(values, dtype=object)
    result = nan_unique(series, sort=sort)
    assert all(pd.notna(v) for v in result)
    assert len(set(result)) == len(result)
    assert set(result) <= set(values)


@given(value_lists, st.booleans())
def test_nan_unique_ignores_injected_nans(values, sort):
    """Injecting NaNs into the series does not change the set of returned values."""
    clean = pd.Series(values, dtype=object)
    noisy = pd.Series([*values, np.nan, None, np.nan], dtype=object)
    assert set(nan_unique(clean, sort=sort)) == set(nan_unique(noisy, sort=sort))
