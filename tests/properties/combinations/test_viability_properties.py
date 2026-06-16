"""Property-based tests for combination viability tests."""

import numpy as np
import pandas as pd
from hypothesis import given
from hypothesis import strategies as st

from AutoCarver.combinations.utils.testing import (
    _test_distinct_target_rates_between_modalities,
    _test_minimum_frequency_per_modality,
    _test_modality_ordering,
)
from AutoCarver.combinations.utils.testing import test_viability as compute_viability
from AutoCarver.discretizers.utils.frequency_ci import is_significantly_below

alphas = st.sampled_from([0.01, 0.05, 0.1])


# --------------------------------------------------------------------------
# distinct target rates
# --------------------------------------------------------------------------
@given(st.lists(st.floats(min_value=0, max_value=1, allow_nan=False), min_size=2, max_size=10))
def test_distinct_rates_matches_consecutive_isclose(values):
    """True iff no consecutive pair is np.isclose."""
    series = pd.Series(values)
    expected = not any(np.isclose(values[i], values[i - 1]) for i in range(1, len(values)))
    assert _test_distinct_target_rates_between_modalities(series) == expected


@given(st.floats(min_value=0, max_value=1, allow_nan=False), st.integers(min_value=2, max_value=8))
def test_constant_rates_are_not_distinct(value, n):
    """A constant series of length >= 2 always fails the distinctness test."""
    series = pd.Series([value] * n)
    assert _test_distinct_target_rates_between_modalities(series) is False


# --------------------------------------------------------------------------
# minimum frequency
# --------------------------------------------------------------------------
@given(st.lists(st.integers(min_value=0, max_value=1000), min_size=1, max_size=8), alphas)
def test_min_freq_none_always_passes(counts, alpha):
    """min_freq=None disables the test."""
    series = pd.Series(counts)
    nobs = max(int(series.sum()), 1)
    assert _test_minimum_frequency_per_modality(series, nobs, None, alpha) is True


@given(
    st.lists(st.integers(min_value=1, max_value=1000), min_size=1, max_size=8),
    alphas,
    st.floats(min_value=0.001, max_value=0.5),
)
def test_min_freq_consistent_with_is_significantly_below(counts, alpha, min_freq):
    """Passes iff no modality is significantly below min_freq."""
    series = pd.Series(counts)
    nobs = int(series.sum())
    result = _test_minimum_frequency_per_modality(series, nobs, min_freq, alpha)
    expected = not bool(np.any(is_significantly_below(series.values, nobs, min_freq, alpha)))
    assert result == expected


# --------------------------------------------------------------------------
# modality ordering
# --------------------------------------------------------------------------
@given(st.lists(st.floats(min_value=0, max_value=1, allow_nan=False), min_size=1, max_size=8, unique=True))
def test_modality_ordering_reflexive(values):
    """A series ranks identically against itself."""
    series = pd.Series(values, index=[f"m{i}" for i in range(len(values))])
    assert _test_modality_ordering(series, series) is True


# --------------------------------------------------------------------------
# test_viability composite
# --------------------------------------------------------------------------
@st.composite
def rates_frame(draw):
    """A per-modality rates frame with 'count', 'frequency' and a target rate col."""
    n = draw(st.integers(min_value=2, max_value=6))
    counts = draw(st.lists(st.integers(min_value=1, max_value=500), min_size=n, max_size=n))
    target = draw(st.lists(st.floats(min_value=0, max_value=1, allow_nan=False), min_size=n, max_size=n))
    total = sum(counts)
    frame = pd.DataFrame(
        {
            "count": counts,
            "frequency": [c / total for c in counts],
            "target_rate": target,
        },
        index=[f"m{i}" for i in range(n)],
    )
    return frame


@given(rates_frame(), st.floats(min_value=0.001, max_value=0.4), alphas)
def test_viability_keys_and_logic_train(frame, min_freq, alpha):
    """On train (no train_target_rate) result carries the test keys and
    VIABLE == MIN_FREQ and DISTINCT_RATES."""
    out = compute_viability(frame, min_freq, "target_rate", alpha)
    train = out["train"]
    assert "viable" in train and "info" in train
    # recompute the components
    min_freq_ok = _test_minimum_frequency_per_modality(frame["count"], int(frame["count"].sum()), min_freq, alpha)
    distinct_ok = _test_distinct_target_rates_between_modalities(frame["target_rate"])
    assert out["viable"] == (min_freq_ok and distinct_ok)


@given(rates_frame(), st.floats(min_value=0.001, max_value=0.4), alphas)
def test_viability_dev_includes_ranking(frame, min_freq, alpha):
    """When train_target_rate is provided, VIABLE also folds in the ranking test."""
    train_rate = frame["target_rate"]
    out = compute_viability(frame, min_freq, "target_rate", alpha, train_target_rate=train_rate)
    dev = out["dev"]
    min_freq_ok = _test_minimum_frequency_per_modality(frame["count"], int(frame["count"].sum()), min_freq, alpha)
    distinct_ok = _test_distinct_target_rates_between_modalities(frame["target_rate"])
    ranking_ok = _test_modality_ordering(train_rate, frame["target_rate"])
    assert dev["viable"] == (min_freq_ok and distinct_ok and ranking_ok)
