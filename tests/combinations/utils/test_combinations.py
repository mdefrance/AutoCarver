"""
Unit tests for the combination utilities in the AutoCarver project.
"""

from pandas import DataFrame
from pytest import raises

from AutoCarver.combinations.utils.combinations import (
    combination_formatter,
    combinations_at_index,
    consecutive_combinations,
    format_combinations,
    nan_combinations,
    order_apply_combination,
    xagg_apply_combination,
)
from AutoCarver.features import GroupedList, OrdinalFeature


def test_combinations_at_index_basic():
    """Test combinations_at_index with a basic order and default start index."""
    order = [1, 2, 3, 4]
    result = list(combinations_at_index(0, order, 2))
    expected = [([1], 1, 1), ([1, 2], 2, 1), ([1, 2, 3], 3, 1), ([1, 2, 3, 4], 4, 1)]
    assert result == expected


def test_combinations_at_index_with_start_idx():
    """Test combinations_at_index with a specified start index."""
    order = [1, 2, 3, 4]
    result = list(combinations_at_index(1, order, 2))
    expected = [([2], 2, 1), ([2, 3], 3, 1), ([2, 3, 4], 4, 1)]
    assert result == expected


def test_combinations_at_index_with_nb_remaining_groups():
    """Test combinations_at_index with a specified number of remaining groups."""
    order = [1, 2, 3, 4]
    result = list(combinations_at_index(0, order, 1))
    expected = [([1, 2, 3, 4], 4, 0)]
    assert result == expected


def test_combinations_at_index_empty_order():
    """Test combinations_at_index with an empty order."""
    order = []
    result = list(combinations_at_index(0, order, 2))
    expected = []
    assert result == expected


def test_combinations_at_index_start_idx_out_of_bounds():
    """Test combinations_at_index with a start index out of bounds."""
    order = [1, 2, 3, 4]
    result = list(combinations_at_index(5, order, 2))
    expected = []
    assert result == expected


def test_consecutive_combinations_basic():
    """Test consecutive_combinations with a basic order and max group size."""
    raw_order = [1, 2, 3, 4]
    max_group_size = 2
    result = consecutive_combinations(raw_order, max_group_size)
    print(result)
    expected = [[[1], [2, 3, 4]], [[1, 2], [3, 4]], [[1, 2, 3], [4]]]
    assert result == expected


def test_consecutive_combinations_with_larger_group_size():
    """Test consecutive_combinations with a larger max group size."""
    raw_order = [1, 2, 3, 4]
    max_group_size = 3
    result = consecutive_combinations(raw_order, max_group_size)
    print(result)
    expected = [
        [[1], [2], [3, 4]],
        [[1], [2, 3], [4]],
        [[1], [2, 3, 4]],
        [[1, 2], [3], [4]],
        [[1, 2], [3, 4]],
        [[1, 2, 3], [4]],
    ]
    assert result == expected


def test_consecutive_combinations_with_single_element():
    """Test consecutive_combinations with a single element in the order."""
    raw_order = [1]
    max_group_size = 1
    result = consecutive_combinations(raw_order, max_group_size)
    expected = []
    assert result == expected


def test_consecutive_combinations_with_empty_list():
    """Test consecutive_combinations with an empty order list."""
    raw_order = []
    max_group_size = 2
    result = consecutive_combinations(raw_order, max_group_size)
    expected = []
    assert result == expected


def test_consecutive_combinations_with_non_default_start_index():
    """Test consecutive_combinations with a non-default start index."""
    raw_order = [1, 2, 3, 4]
    max_group_size = 2
    result = consecutive_combinations(raw_order, max_group_size, next_index=1)
    print(result)
    expected = [[[2], [3, 4]], [[2, 3], [4]]]
    assert result == expected


def test_nan_combinations_basic():
    """Test nan_combinations with a basic feature and max_n_mod."""
    str_nan = "NaN_"
    feature = OrdinalFeature("test", values=["A", "B", "C", "D"], nan=str_nan)
    feature.has_nan = True
    feature.dropna = True
    max_n_mod = 2
    result = nan_combinations(feature, max_n_mod)
    print(result)
    expected = [
        [["A", str_nan], ["B", "C", "D"]],
        [["A"], ["B", "C", "D", str_nan]],
        [["A", "B", str_nan], ["C", "D"]],
        [["A", "B"], ["C", "D", str_nan]],
        [["A", "B", "C", str_nan], ["D"]],
        [["A", "B", "C"], ["D", str_nan]],
        [["A", "B", "C", "D"], [str_nan]],
    ]

    assert result == expected


def test_nan_combinations_with_single_label():
    """Test nan_combinations with a single label in the feature."""
    str_nan = "NaN_"
    feature = OrdinalFeature("test", values=["A"], nan=str_nan)
    feature.has_nan = True
    feature.dropna = True
    max_n_mod = 1
    result = nan_combinations(feature, max_n_mod)
    print(result)
    expected = [[["A"], [str_nan]]]
    assert result == expected


def test_nan_combinations_with_max_n_mod_greater_than_labels():
    """Test nan_combinations with max_n_mod greater than the number of labels."""
    str_nan = "NaN_"
    feature = OrdinalFeature("test", values=["A", "B"], nan=str_nan)
    feature.has_nan = True
    feature.dropna = True
    max_n_mod = 3
    result = nan_combinations(feature, max_n_mod)
    print(result)
    expected = [
        [["A", "NaN_"], ["B"]],
        [["A"], ["B", "NaN_"]],
        [["A"], ["B"], ["NaN_"]],
        [["A", "B"], ["NaN_"]],
    ]

    assert result == expected


def test_nan_combinations_with_large_combination():
    """Test nan_combinations with a large combination of labels."""
    str_nan = "NaN_"
    feature = OrdinalFeature("test", values=["A", "B", "C", "D"], nan=str_nan)
    feature.has_nan = True
    feature.dropna = True
    max_n_mod = 4
    result = nan_combinations(feature, max_n_mod)
    print(result)
    expected = [
        [["A", "NaN_"], ["B"], ["C"], ["D"]],
        [["A"], ["B", "NaN_"], ["C"], ["D"]],
        [["A"], ["B"], ["C", "NaN_"], ["D"]],
        [["A"], ["B"], ["C"], ["D", "NaN_"]],
        [["A", "NaN_"], ["B"], ["C", "D"]],
        [["A"], ["B", "NaN_"], ["C", "D"]],
        [["A"], ["B"], ["C", "D", "NaN_"]],
        [["A"], ["B"], ["C", "D"], ["NaN_"]],
        [["A", "NaN_"], ["B", "C"], ["D"]],
        [["A"], ["B", "C", "NaN_"], ["D"]],
        [["A"], ["B", "C"], ["D", "NaN_"]],
        [["A"], ["B", "C"], ["D"], ["NaN_"]],
        [["A", "NaN_"], ["B", "C", "D"]],
        [["A"], ["B", "C", "D", "NaN_"]],
        [["A"], ["B", "C", "D"], ["NaN_"]],
        [["A", "B", "NaN_"], ["C"], ["D"]],
        [["A", "B"], ["C", "NaN_"], ["D"]],
        [["A", "B"], ["C"], ["D", "NaN_"]],
        [["A", "B"], ["C"], ["D"], ["NaN_"]],
        [["A", "B", "NaN_"], ["C", "D"]],
        [["A", "B"], ["C", "D", "NaN_"]],
        [["A", "B"], ["C", "D"], ["NaN_"]],
        [["A", "B", "C", "NaN_"], ["D"]],
        [["A", "B", "C"], ["D", "NaN_"]],
        [["A", "B", "C"], ["D"], ["NaN_"]],
        [["A", "B", "C", "D"], ["NaN_"]],
    ]

    assert result == expected


def test_nan_combinations_with_low_max_n_mod():
    """Test nan_combinations with a low max_n_mod."""
    str_nan = "NaN_"
    feature = OrdinalFeature("test", values=["A", "B", "C", "D"], nan=str_nan)
    feature.has_nan = True
    feature.dropna = True
    max_n_mod = 2
    result = nan_combinations(feature, max_n_mod)
    print(result)
    expected = [
        [["A", "NaN_"], ["B", "C", "D"]],
        [["A"], ["B", "C", "D", "NaN_"]],
        [["A", "B", "NaN_"], ["C", "D"]],
        [["A", "B"], ["C", "D", "NaN_"]],
        [["A", "B", "C", "NaN_"], ["D"]],
        [["A", "B", "C"], ["D", "NaN_"]],
        [["A", "B", "C", "D"], ["NaN_"]],
    ]

    assert result == expected


def test_combination_formatter_basic():
    """Test combination_formatter with a basic combination."""
    combination = [["A", "B"], ["C", "D"]]
    result = combination_formatter(combination)
    expected = {"A": "A", "B": "A", "C": "C", "D": "C"}
    assert result == expected


def test_combination_formatter_single_group():
    """Test combination_formatter with a single group."""
    combination = [["A", "B", "C"]]
    result = combination_formatter(combination)
    expected = {"A": "A", "B": "A", "C": "A"}
    assert result == expected


def test_combination_formatter_single_element_groups():
    """Test combination_formatter with single element groups."""
    combination = [["A"], ["B"], ["C"]]
    result = combination_formatter(combination)
    expected = {"A": "A", "B": "B", "C": "C"}
    assert result == expected


def test_combination_formatter_empty_group():
    """Test combination_formatter with an empty group."""
    combination = [[]]
    result = combination_formatter(combination)
    expected = {}
    assert result == expected


def test_combination_formatter_empty_combination():
    """Test combination_formatter with an empty combination."""
    combination = []
    result = combination_formatter(combination)
    expected = {}
    assert result == expected


def test_combination_formatter_mixed_groups():
    """Test combination_formatter with mixed groups."""
    combination = [["A", "B"], ["C"], ["D", "E", "F"]]
    result = combination_formatter(combination)
    expected = {"A": "A", "B": "A", "C": "C", "D": "D", "E": "D", "F": "D"}
    assert result == expected


def test_order_apply_combination_basic():
    """Test order_apply_combination with a basic order and combination."""
    order = GroupedList(["A", "B", "C", "D"])
    combination = [["A", "B"], ["C", "D"]]
    result = order_apply_combination(order, combination)
    print(result.content)
    expected = {"A": ["B", "A"], "C": ["D", "C"]}
    assert result.content == expected
    assert result == ["A", "C"]


def test_order_apply_combination_single_group():
    """Test order_apply_combination with a single group."""
    order = GroupedList(["A", "B", "C"])
    combination = [["A", "B", "C"]]
    result = order_apply_combination(order, combination)
    print(result.content)
    expected = {"A": ["C", "B", "A"]}
    assert result.content == expected
    assert result == ["A"]


def test_order_apply_combination_single_element_groups():
    """Test order_apply_combination with single element groups."""
    order = GroupedList(["A", "B", "C"])
    combination = [["A"], ["B"], ["C"]]
    result = order_apply_combination(order, combination)
    print(result.content)
    expected = {"A": ["A"], "B": ["B"], "C": ["C"]}
    assert result.content == expected
    assert result == ["A", "B", "C"]


def test_order_apply_combination_empty_group():
    """Test order_apply_combination with an empty group."""
    order = GroupedList(["A", "B", "C"])
    combination = [[]]
    with raises(IndexError):
        order_apply_combination(order, combination)
    combination = []
    result = order_apply_combination(order, combination)
    print(result.content)
    expected = {"A": ["A"], "B": ["B"], "C": ["C"]}
    assert result.content == expected
    assert result == ["A", "B", "C"]


def test_order_apply_combination_mixed_groups():
    """Test order_apply_combination with mixed groups."""
    order = GroupedList(["A", "B", "C", "D", "E"])
    combination = [["A", "B"], ["C"], ["D", "E"]]
    result = order_apply_combination(order, combination)
    print(result.content)
    expected = {"A": ["B", "A"], "C": ["C"], "D": ["E", "D"]}
    assert result.content == expected
    assert result == ["A", "C", "D"]


def test_xagg_apply_combination_basic():
    """Test xagg_apply_combination with a basic xagg and combination."""
    xagg = DataFrame({"A": [1, 2, 3, 4], "B": [5, 6, 7, 8]}, index=["a", "b", "c", "d"])
    order = GroupedList(["a", "b", "c", "d"])
    combination = [["a", "b"], ["c", "d"]]
    order = order_apply_combination(order, combination)
    feature = OrdinalFeature("test", values=GroupedList({"a": ["a", "b"], "c": ["c", "d"]}))
    print(feature.value_per_label)
    result = xagg_apply_combination(xagg, feature)
    print(result)
    expected = DataFrame({"A": [3, 7], "B": [11, 15]}, index=["a", "c"])
    assert result.equals(expected)


def test_xagg_apply_combination_single_group():
    """Test xagg_apply_combination with a single group."""
    xagg = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=["a", "b", "c"])

    order = GroupedList(["a", "b", "c"])
    combination = [["a", "b", "c"]]
    order = order_apply_combination(order, combination)
    feature = OrdinalFeature("test", values=GroupedList({"a": ["a", "b", "c"]}))

    result = xagg_apply_combination(xagg, feature)
    expected = DataFrame({"A": [6], "B": [15]}, index=["a"])
    print(result)
    assert result.equals(expected)


def test_xagg_apply_combination_single_element_groups():
    """Test xagg_apply_combination with single element groups."""
    xagg = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=["a", "b", "c"])

    order = GroupedList(["a", "b", "c"])
    combination = [["a"], ["b"], ["c"]]
    order = order_apply_combination(order, combination)
    feature = OrdinalFeature("test", values=["a", "b", "c"])

    result = xagg_apply_combination(xagg, feature)
    expected = xagg
    print(result)
    assert result.equals(expected)


def test_xagg_apply_combination_empty_xagg():
    """Test xagg_apply_combination with an empty xagg."""
    xagg = None

    order = GroupedList(["a", "b", "c"])
    combination = [["a"], ["b"], ["c"]]
    order = order_apply_combination(order, combination)
    feature = OrdinalFeature("test", values=["a", "b", "c"])

    result = xagg_apply_combination(xagg, feature)
    assert result is None


def test_xagg_apply_combination_empty_order():
    """Test xagg_apply_combination with an empty order."""
    xagg = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=["a", "b", "c"])
    feature = OrdinalFeature("test", values=["a", "b", "c"])
    result = xagg_apply_combination(xagg, feature)
    expected = xagg
    print(result)
    assert result.equals(expected)


def test_xagg_apply_combination_mixed_groups():
    """Test xagg_apply_combination with mixed groups."""
    xagg = DataFrame({"A": [1, 2, 3, 4, 5], "B": [6, 7, 8, 9, 10]}, index=["a", "b", "c", "d", "e"])

    order = GroupedList(["a", "b", "c", "d", "e"])
    combination = [["a", "b"], ["c"], ["d", "e"]]
    order = order_apply_combination(order, combination)
    feature = OrdinalFeature("test", values=GroupedList({"a": ["a", "b"], "c": ["c"], "d": ["d", "e"]}))

    result = xagg_apply_combination(xagg, feature)
    expected = DataFrame({"A": [3, 3, 9], "B": [13, 8, 19]}, index=["a", "c", "d"])
    print(result)
    assert result.equals(expected)


def test_xagg_apply_combination_with_nan():
    """Test xagg_apply_combination with NaN values in the feature."""
    feature = OrdinalFeature("test", values=GroupedList({"a": ["a", "b"], "c": ["c", "d"]}))
    feature.has_nan = True
    feature.dropna = False
    xagg = DataFrame({"A": [1, 2, 3, 4, 5], "B": [5, 6, 7, 8, 10]}, index=["a", "b", "c", "d", feature.nan])
    order = GroupedList(["a", "b", "c", "d"])
    combination = [["a", "b"], ["c", "d"]]
    order = order_apply_combination(order, combination)

    result = xagg_apply_combination(xagg, feature)
    print(result)
    expected = DataFrame({"A": [3, 7, 5], "B": [11, 15, 10]}, index=["a", "c", feature.nan])
    assert result.equals(expected)

    # dropped nan
    feature = OrdinalFeature("test", values=GroupedList({"a": ["a", "b"], "c": ["c", "d"]}))
    feature.has_nan = True
    feature.dropna = True
    xagg = DataFrame({"A": [1, 2, 3, 4, 5], "B": [5, 6, 7, 8, 10]}, index=["a", "b", "c", "d", feature.nan])
    order = GroupedList(["a", "b", "c", "d"])
    combination = [["a", "b"], ["c", "d"]]
    order = order_apply_combination(order, combination)

    result = xagg_apply_combination(xagg, feature)
    print(result)
    expected = DataFrame({"A": [3, 7, 5], "B": [11, 15, 10]}, index=["a", "c", feature.nan])
    assert result.equals(expected)


def test_format_combinations_basic():
    """Test format_combinations with a basic set of combinations."""
    combinations = [[["A", "B"], ["C", "D"]], [["E", "F"], ["G", "H"]]]
    result = format_combinations(combinations)
    expected = [{"A": "A", "B": "A", "C": "C", "D": "C"}, {"E": "E", "F": "E", "G": "G", "H": "G"}]
    assert result == expected


def test_format_combinations_single_group():
    """Test format_combinations with a single group."""
    combinations = [[["A", "B", "C"]]]
    result = format_combinations(combinations)
    expected = [{"A": "A", "B": "A", "C": "A"}]
    assert result == expected


def test_format_combinations_single_element_groups():
    """Test format_combinations with single element groups."""
    combinations = [[["A"], ["B"], ["C"]]]
    result = format_combinations(combinations)
    expected = [{"A": "A", "B": "B", "C": "C"}]
    assert result == expected


def test_format_combinations_empty_group():
    """Test format_combinations with an empty group."""
    combinations = [[[]]]
    result = format_combinations(combinations)
    expected = [{}]
    assert result == expected


def test_format_combinations_empty_combinations():
    """Test format_combinations with empty combinations."""
    combinations = []
    result = format_combinations(combinations)
    expected = []
    assert result == expected


def test_format_combinations_mixed_groups():
    """Test format_combinations with mixed groups."""
    combinations = [[["A", "B"], ["C"], ["D", "E", "F"]]]
    result = format_combinations(combinations)
    expected = [{"A": "A", "B": "A", "C": "C", "D": "D", "E": "D", "F": "D"}]
    assert result == expected
