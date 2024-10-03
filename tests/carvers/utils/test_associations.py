""" set of tests for associations within carvers"""

from pandas import DataFrame, Series
from AutoCarver.carvers.utils.associations import (
    filter_nan,
    CombinationEvaluator,
)


# removing abstract parts of CombinationEvaluator
CombinationEvaluator.__abstractmethods__ = set()


def test_filter_nan_with_dataframe():
    xagg = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=["a", "NaN", "c"])
    str_nan = "NaN"
    result = filter_nan(xagg, str_nan)
    expected = DataFrame({"A": [1, 3], "B": [4, 6]}, index=["a", "c"])
    assert result.equals(expected)


def test_filter_nan_with_series():
    xagg = Series([1, 2, 3], index=["a", "NaN", "c"])
    str_nan = "NaN"
    result = filter_nan(xagg, str_nan)
    expected = Series([1, 3], index=["a", "c"])
    assert result.equals(expected)


def test_filter_nan_no_nan_in_index():
    xagg = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=["a", "b", "c"])
    str_nan = "NaN"
    result = filter_nan(xagg, str_nan)
    expected = xagg.copy()
    assert result.equals(expected)


def test_filter_nan_empty_dataframe():
    xagg = DataFrame()
    str_nan = "NaN"
    result = filter_nan(xagg, str_nan)
    expected = DataFrame()
    assert result.equals(expected)


def test_filter_nan_none_input():
    xagg = None
    str_nan = "NaN"
    result = filter_nan(xagg, str_nan)
    expected = None
    assert result == expected

