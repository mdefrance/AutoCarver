"""Property-based tests for the ordinal & categorical discretizers.

Source: ``ordinal_discretizer`` / ``categorical_discretizer``. OrdinalDiscretizer
only merges *adjacent* modalities (user order is never reordered); merged groups
are contiguous. CategoricalDiscretizer rolls modalities that are significantly
below ``min_freq`` (Wilson CI) into the feature's default, leaving every standalone
survivor above that floor.
"""

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st
from strategies import binary_target

from AutoCarver.discretizers.qualitatives.categorical_discretizer import CategoricalDiscretizer
from AutoCarver.discretizers.qualitatives.ordinal_discretizer import OrdinalDiscretizer
from AutoCarver.discretizers.utils.frequency_ci import is_significantly_below
from AutoCarver.features import Features

min_freqs = st.sampled_from([0.1, 0.15, 0.2, 0.3])


# --------------------------------------------------------------------------
# OrdinalDiscretizer
# --------------------------------------------------------------------------
@given(st.data(), min_freqs)
@settings(max_examples=40, deadline=None)
def test_ordinal_preserves_order_and_contiguous_groups(data, min_freq):
    """Fitted leaders are a subsequence of the user order; each group is a
    contiguous run of the user order; every original value is covered once."""
    cardinality = data.draw(st.integers(min_value=2, max_value=6))
    order = [f"o{j}" for j in range(cardinality)]
    n = data.draw(st.integers(min_value=30, max_value=120))
    X = pd.DataFrame({"f": data.draw(st.lists(st.sampled_from(order), min_size=n, max_size=n))})
    y = data.draw(binary_target(n))

    features = Features(ordinals={"f": order})
    OrdinalDiscretizer(features.ordinals, min_freq).fit(X, y)

    feature = features("f")
    content = feature.content
    pos = {value: i for i, value in enumerate(order)}

    # leaders appear in strictly increasing user-order position (no reordering)
    leader_positions = [pos[leader] for leader in feature.values]
    assert leader_positions == sorted(leader_positions)

    # every original value covered exactly once across groups
    members = [m for group in content.values() for m in group]
    assert sorted(members) == sorted(order)

    # each group is a contiguous block of the user order
    for group in content.values():
        indices = sorted(pos[m] for m in group)
        assert indices == list(range(indices[0], indices[-1] + 1))


# --------------------------------------------------------------------------
# CategoricalDiscretizer
# --------------------------------------------------------------------------
@given(st.data(), min_freqs)
@settings(max_examples=40, deadline=None)
def test_categorical_survivors_above_floor(data, min_freq):
    """Every standalone survivor is not significantly below ``min_freq``; any
    rare modality is rolled into the feature's default."""
    cardinality = data.draw(st.integers(min_value=2, max_value=8))
    pool = [chr(ord("a") + j) for j in range(cardinality)]
    n = data.draw(st.integers(min_value=30, max_value=120))
    raw = data.draw(st.lists(st.sampled_from(pool), min_size=n, max_size=n))
    X = pd.DataFrame({"f": pd.Series(raw, dtype=object)})
    y = data.draw(binary_target(n))

    features = Features(categoricals=["f"])
    feature = features("f")
    alpha = 0.05  # ProcessingConfig default min_freq_alpha

    nobs = len(X)
    counts = X["f"].value_counts()
    rare = {value for value, count in counts.items() if is_significantly_below(count, nobs, min_freq, alpha)}

    CategoricalDiscretizer(features.categoricals, min_freq).fit(X, y)

    # standalone survivors (excluding default / nan) are above the floor
    for leader in feature.values:
        if leader in (feature.default, feature.nan):
            continue
        count = int(counts.get(leader, 0))
        assert not is_significantly_below(count, nobs, min_freq, alpha)

    # rare modalities were rolled into the default group
    if rare:
        assert feature.has_default
        for value in rare:
            assert feature.label_per_value.get(value) == feature.label_per_value.get(feature.default)
