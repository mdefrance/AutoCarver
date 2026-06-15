"""Property-based tests for the Wilson-score frequency helpers."""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from AutoCarver.discretizers.utils.frequency_ci import is_significantly_below, wilson_upper_bound

alphas = st.sampled_from([0.01, 0.05, 0.1])
nobs_strat = st.integers(min_value=1, max_value=100_000)


@given(st.data(), nobs_strat, alphas)
def test_bound_in_unit_interval(data, nobs, alpha):
    """Wilson upper bound is always a valid probability in [0, 1]."""
    count = data.draw(st.integers(min_value=0, max_value=nobs))
    bound = wilson_upper_bound(count, nobs, alpha)
    assert 0.0 <= bound <= 1.0


@given(st.data(), nobs_strat, alphas)
def test_bound_above_point_estimate(data, nobs, alpha):
    """The upper bound never falls below the observed proportion count/nobs."""
    count = data.draw(st.integers(min_value=0, max_value=nobs))
    bound = wilson_upper_bound(count, nobs, alpha)
    assert bound >= count / nobs - 1e-12


@given(st.data(), nobs_strat, alphas)
def test_bound_monotone_in_count(data, nobs, alpha):
    """For fixed nobs/alpha, the bound is non-decreasing in count."""
    c1 = data.draw(st.integers(min_value=0, max_value=nobs))
    c2 = data.draw(st.integers(min_value=0, max_value=nobs))
    lo, hi = sorted((c1, c2))
    assert wilson_upper_bound(lo, nobs, alpha) <= wilson_upper_bound(hi, nobs, alpha) + 1e-12


@given(alphas)
def test_zero_nobs_returns_one(alpha):
    """Empty samples are treated as non-significant (bound = 1.0)."""
    assert wilson_upper_bound(5, 0, alpha) == 1.0


@given(st.lists(st.integers(min_value=0, max_value=1000), min_size=1, max_size=10), nobs_strat, alphas)
def test_vectorized_matches_scalar(counts, nobs, alpha):
    """Array input yields the elementwise scalar results, same shape."""
    counts = [min(c, nobs) for c in counts]
    arr = np.asarray(counts)
    vec = wilson_upper_bound(arr, nobs, alpha)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == arr.shape
    for c, v in zip(counts, vec):
        assert np.isclose(v, wilson_upper_bound(c, nobs, alpha))


@given(st.data(), nobs_strat, alphas, st.floats(min_value=0.001, max_value=0.5))
def test_is_significantly_below_matches_definition(data, nobs, alpha, min_freq):
    """is_significantly_below == (wilson_upper_bound < min_freq), scalar bool out."""
    count = data.draw(st.integers(min_value=0, max_value=nobs))
    result = is_significantly_below(count, nobs, min_freq, alpha)
    assert isinstance(result, bool)
    assert result == (wilson_upper_bound(count, nobs, alpha) < min_freq)


@given(st.data(), nobs_strat, alphas, st.floats(min_value=0.001, max_value=0.5))
def test_not_below_when_point_estimate_meets_threshold(data, nobs, alpha, min_freq):
    """If count/nobs >= min_freq, the modality can't be significantly below it."""
    count = data.draw(st.integers(min_value=0, max_value=nobs))
    if count / nobs >= min_freq:
        assert is_significantly_below(count, nobs, min_freq, alpha) is False
