"""Property-based tests for GroupedList and is_equal.

Asserts the structural invariants the rest of AutoCarver relies on:
self-membership, disjointness, value conservation under mutation, and the
NaN-aware equality contract.
"""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from strategies import distinct_labels, grouped_list_dict

from AutoCarver.features import GroupedList
from AutoCarver.features.utils.grouped_list import is_equal


# --------------------------------------------------------------------------
# is_equal
# --------------------------------------------------------------------------
@given(st.integers() | st.text() | st.floats(allow_nan=False))
def test_is_equal_reflexive(x):
    """is_equal(x, x) is always True for non-NaN values."""
    assert is_equal(x, x)


@given(
    st.integers() | st.text() | st.floats(allow_nan=False),
    st.integers() | st.text() | st.floats(allow_nan=False),
)
def test_is_equal_symmetric(a, b):
    """is_equal is symmetric."""
    assert is_equal(a, b) == is_equal(b, a)


@given(st.sampled_from([np.nan, float("nan"), None]))
def test_is_equal_nan_aware(nan_value):
    """All NaN-likes compare equal to each other; a NaN is never equal to a number."""
    assert is_equal(nan_value, np.nan) is True
    assert is_equal(nan_value, 1.0) is False


# --------------------------------------------------------------------------
# Construction & round-trip
# --------------------------------------------------------------------------
@given(grouped_list_dict())
def test_constructed_from_dict_satisfies_invariants(content):
    """Building from a valid {leader: members} dict never violates sanity_check."""
    gl = GroupedList(content)
    gl.sanity_check()  # raises on violation
    # disjointness: no duplicate values across groups
    assert len(set(gl.values)) == len(gl.values)


@given(distinct_labels())
def test_list_constructor_preserves_order(order):
    """A list initializer yields singleton groups in the same leader order."""
    gl = GroupedList(order)
    assert list(gl) == order
    assert gl.values == order


@given(grouped_list_dict())
def test_copy_constructor_independent(content):
    """The copy constructor produces an equal-but-independent instance."""
    gl = GroupedList(content)
    copy = GroupedList(gl)
    assert list(copy) == list(gl)
    # mutating the copy must not touch the original
    leader = list(copy)[0]
    copy.content[leader].append("___sentinel___")
    assert "___sentinel___" not in gl.values


# --------------------------------------------------------------------------
# Mutations conserve invariants
# --------------------------------------------------------------------------
@given(distinct_labels(min_size=2), st.data())
def test_group_conserves_values_and_invariants(order, data):
    """group(discarded, kept) loses no value and keeps the structure valid."""
    gl = GroupedList(order)
    before = sorted(gl.values)
    # pick two distinct leaders
    kept = data.draw(st.sampled_from(order))
    candidates = [v for v in order if v != kept]
    discarded = data.draw(st.sampled_from(candidates))

    gl.group(discarded, kept)

    gl.sanity_check()
    assert sorted(gl.values) == before  # multiset of values unchanged
    assert len(gl) == len(order) - 1  # one fewer leader
    assert discarded not in gl  # discarded is no longer a leader
    assert gl.get_group(discarded) == kept  # but is reachable under kept


@given(distinct_labels())
def test_append_then_remove_roundtrip(order):
    """Appending a fresh leader then removing it restores the structure."""
    gl = GroupedList(order)
    new = "".join(order) + "Z"  # guaranteed not already present
    gl.append(new)
    assert new in gl
    gl.sanity_check()
    gl.remove(new)
    assert list(gl) == order


@given(grouped_list_dict())
def test_get_group_returns_leader_containing_value(content):
    """For any tracked value, get_group returns a leader whose group holds it."""
    gl = GroupedList(content)
    for value in gl.values:
        leader = gl.get_group(value)
        assert leader in gl
        assert value in gl.get(leader)


# --------------------------------------------------------------------------
# Ordering
# --------------------------------------------------------------------------
@given(distinct_labels())
def test_sort_is_permutation_of_leaders(order):
    """sort() returns the same leaders (as a set), just reordered."""
    gl = GroupedList(order)
    sorted_gl = gl.sort()
    assert set(sorted_gl) == set(gl)
    assert len(sorted_gl) == len(gl)


@given(distinct_labels(min_size=1))
def test_sort_by_permutation_roundtrip(order):
    """sort_by(perm) reorders leaders to match the given permutation."""
    gl = GroupedList(order)
    perm = list(reversed(order))
    reordered = gl.sort_by(perm)
    assert list(reordered) == perm
