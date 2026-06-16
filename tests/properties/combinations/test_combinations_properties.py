"""Property-based tests for combination enumeration & application utilities."""

from hypothesis import given
from hypothesis import strategies as st
from strategies import contiguous_partition, distinct_labels

from AutoCarver.combinations.utils.combinations import (
    combination_formatter,
    consecutive_combinations,
    nan_combinations,
    order_apply_combination,
)
from AutoCarver.features import GroupedList, OrdinalFeature


# --------------------------------------------------------------------------
# consecutive_combinations
# --------------------------------------------------------------------------
@given(distinct_labels(min_size=1, max_size=7), st.integers(min_value=1, max_value=7))
def test_consecutive_combinations_are_contiguous_partitions(order, max_group_size):
    """Every yielded combination partitions `order` into >=2 contiguous groups."""
    for combination in consecutive_combinations(order, max_group_size):
        assert len(combination) >= 2
        # all groups non-empty
        assert all(len(group) >= 1 for group in combination)
        # concatenation reproduces order exactly (contiguous + complete + ordered)
        flat = [v for group in combination for v in group]
        assert flat == order


@given(distinct_labels(min_size=1, max_size=6), st.integers(min_value=1, max_value=6))
def test_consecutive_combinations_respect_group_count_cap(order, max_group_size):
    """No yielded combination has more groups than max_group_size."""
    for combination in consecutive_combinations(order, max_group_size):
        assert len(combination) <= max_group_size


# --------------------------------------------------------------------------
# combination_formatter
# --------------------------------------------------------------------------
@given(contiguous_partition())
def test_combination_formatter_maps_each_value_to_its_leader(order_combo):
    """Each value maps to the first element of its own group; keys == all values."""
    _, combination = order_combo
    mapping = combination_formatter(combination)
    flat = [v for group in combination for v in group]
    assert set(mapping) == set(flat)
    for group in combination:
        for value in group:
            assert mapping[value] == group[0]


# --------------------------------------------------------------------------
# order_apply_combination
# --------------------------------------------------------------------------
@given(contiguous_partition(min_groups=1))
def test_order_apply_combination_valid_and_conservative(order_combo):
    """Result is a valid GroupedList, one group per combination entry, and
    conserves the full set of original values."""
    order, combination = order_combo
    # skip degenerate empty-group combinations (documented to raise IndexError)
    if any(len(group) == 0 for group in combination):
        return
    result = order_apply_combination(GroupedList(order), combination)
    result.sanity_check()
    assert len(result) == len(combination)
    assert sorted(result.values) == sorted(order)
    # each result leader is the first element of a combination group
    assert list(result) == [group[0] for group in combination]


# --------------------------------------------------------------------------
# nan_combinations
# --------------------------------------------------------------------------
@given(distinct_labels(min_size=1, max_size=5), st.integers(min_value=1, max_value=5))
def test_nan_combinations_place_nan_exactly_once(labels, max_n_mod):
    """Every nan combination contains the nan label exactly once and conserves
    the non-nan labels."""
    str_nan = "__NaN__"
    # labels are unique; nan token is distinct from them
    feature = OrdinalFeature("f", values=list(labels))
    feature.nan = str_nan
    feature.has_nan = True
    feature.dropna = True

    non_nan = set(labels)
    for combination in nan_combinations(feature, max_n_mod):
        flat = [v for group in combination for v in group]
        assert flat.count(str_nan) == 1
        assert set(flat) - {str_nan} == non_nan


@given(distinct_labels(min_size=1, max_size=5), st.integers(min_value=1, max_value=5))
def test_nan_combinations_last_is_all_vs_nan(labels, max_n_mod):
    """The final yielded combination is the all-modalities vs. nan partition."""
    str_nan = "__NaN__"
    feature = OrdinalFeature("f", values=list(labels))
    feature.nan = str_nan
    feature.has_nan = True
    feature.dropna = True

    combinations = list(nan_combinations(feature, max_n_mod))
    last = combinations[-1]
    assert last == [list(labels), [str_nan]]
