"""Property-based tests for the redundancy filters.

Source: ``AutoCarver.selectors.filters``. Every ``filter(X, ranks)`` is a
subset-selecting, order-preserving, idempotent operation. The validity filter
keeps a feature iff its recorded measures are all valid; the correlation filters
drop a feature that is too correlated with a higher-ranked one (threshold < 1)
and keep everything at the default threshold of 1.0.
"""

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from AutoCarver.features import CategoricalFeature, QuantitativeFeature, get_versions
from AutoCarver.selectors.filters import (
    CramervFilter,
    PearsonFilter,
    SpearmanFilter,
    TschuprowtFilter,
    ValidFilter,
)


# --------------------------------------------------------------------------
# ValidFilter
# --------------------------------------------------------------------------
@st.composite
def features_with_validity(draw):
    """A list of features, each with 0+ recorded measures of random validity."""
    n = draw(st.integers(min_value=1, max_value=6))
    features = []
    for i in range(n):
        feature = QuantitativeFeature(f"q{i}")
        n_measures = draw(st.integers(min_value=0, max_value=3))
        feature.measures = {f"M{j}": {"valid": draw(st.booleans())} for j in range(n_measures)}
        features.append(feature)
    return features


@given(features_with_validity())
def test_valid_filter_keeps_iff_all_valid(features):
    """ValidFilter keeps a feature iff it has no measures or all are valid;
    output is an order-preserving subset of the input."""
    X = pd.DataFrame()
    result = ValidFilter().filter(X, features)

    expected = [f for f in features if len(f.measures) == 0 or all(m["valid"] for m in f.measures.values())]
    assert result == expected
    # subset + order preserved among survivors
    assert all(f in features for f in result)
    assert result == [f for f in features if f in result]


@given(features_with_validity())
def test_valid_filter_idempotent(features):
    """Applying ValidFilter twice equals applying it once."""
    X = pd.DataFrame()
    once = ValidFilter().filter(X, features)
    twice = ValidFilter().filter(X, once)
    assert twice == once


# --------------------------------------------------------------------------
# Correlation filters — quantitative
# --------------------------------------------------------------------------
@st.composite
def quant_features_and_X(draw):
    """A list of quantitative features and a matching numeric DataFrame."""
    n_features = draw(st.integers(min_value=2, max_value=4))
    nrows = draw(st.integers(min_value=20, max_value=60))
    features = [QuantitativeFeature(f"q{i}") for i in range(n_features)]
    columns = {
        f.version: draw(
            st.lists(st.floats(-1e2, 1e2, allow_nan=False, allow_infinity=False), min_size=nrows, max_size=nrows)
        )
        for f in features
    }
    return features, pd.DataFrame(columns)


@given(quant_features_and_X())
@settings(max_examples=40)
def test_quantitative_filter_subset_and_order(features_X):
    """PearsonFilter output is an order-preserving subset of the ranks."""
    features, X = features_X
    result = PearsonFilter().filter(X, features)
    assert all(f in features for f in result)
    assert result == [f for f in features if f in result]


@given(quant_features_and_X())
@settings(max_examples=40)
def test_quantitative_filter_default_threshold_keeps_all(features_X):
    """At the default threshold of 1.0 no feature is dropped."""
    features, X = features_X
    # default threshold = 1.0; |corr| > 1.0 is impossible -> nothing dropped
    assert PearsonFilter().filter(X, features) == features
    assert SpearmanFilter().filter(X, features) == features


@given(st.integers(min_value=20, max_value=60))
def test_quantitative_filter_drops_perfectly_correlated(nrows):
    """A lower-ranked, perfectly correlated feature is dropped below threshold 1."""
    a = QuantitativeFeature("a")
    b = QuantitativeFeature("b")
    base = np.linspace(-5, 5, nrows)
    X = pd.DataFrame({"a": base, "b": 2 * base + 1})  # b is an affine image of a

    # a is ranked first (more associated with target) -> b is the redundant one
    kept = PearsonFilter(threshold=0.5).filter(X, [a, b])
    assert kept == [a]


# --------------------------------------------------------------------------
# Correlation filters — qualitative
# --------------------------------------------------------------------------
@given(st.integers(min_value=20, max_value=60))
def test_qualitative_filter_drops_perfectly_associated(nrows):
    """A lower-ranked, identical qualitative feature is dropped below threshold 1."""
    a = CategoricalFeature("a")
    b = CategoricalFeature("b")
    pattern = ["x" if i % 2 else "y" for i in range(nrows)]
    X = pd.DataFrame({"a": pattern, "b": pattern})  # identical -> Cramér's V == 1

    for filter_cls in (CramervFilter, TschuprowtFilter):
        kept = filter_cls(threshold=0.5).filter(X, [a, b])
        assert kept == [a]


@given(st.integers(min_value=20, max_value=60))
def test_qualitative_filter_default_threshold_keeps_all(nrows):
    """At the default threshold of 1.0 even identical qualitative features survive."""
    a = CategoricalFeature("a")
    b = CategoricalFeature("b")
    pattern = ["x" if i % 2 else "y" for i in range(nrows)]
    X = pd.DataFrame({"a": pattern, "b": pattern})

    assert get_versions(CramervFilter().filter(X, [a, b])) == ["a", "b"]
