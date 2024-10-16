""" set of tests for associations within carvers"""

from pandas import DataFrame, Series
from pytest import fixture

from AutoCarver.combinations.utils.combination_evaluator import filter_nan, AggregatedSample


def test_filter_nan_with_dataframe():
    """Test filter_nan with a DataFrame"""
    xagg = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=["a", "NaN", "c"])
    str_nan = "NaN"
    result = filter_nan(xagg, str_nan)
    expected = DataFrame({"A": [1, 3], "B": [4, 6]}, index=["a", "c"])
    assert result.equals(expected)


def test_filter_nan_with_series():
    """Test filter_nan with a Series"""
    xagg = Series([1, 2, 3], index=["a", "NaN", "c"])
    str_nan = "NaN"
    result = filter_nan(xagg, str_nan)
    expected = Series([1, 3], index=["a", "c"])
    assert result.equals(expected)


def test_filter_nan_no_nan_in_index():
    """Test filter_nan with no NaN in the index"""
    xagg = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=["a", "b", "c"])
    str_nan = "NaN"
    result = filter_nan(xagg, str_nan)
    expected = xagg.copy()
    assert result.equals(expected)


def test_filter_nan_empty_dataframe():
    """Test filter_nan with an empty DataFrame"""
    xagg = DataFrame()
    str_nan = "NaN"
    result = filter_nan(xagg, str_nan)
    expected = DataFrame()
    assert result.equals(expected)


def test_filter_nan_none_input():
    """Test filter_nan with None input"""
    xagg = None
    str_nan = "NaN"
    result = filter_nan(xagg, str_nan)
    expected = None
    assert result == expected


@fixture
def sample_data():
    """Sample data for AggregatedSample"""
    return DataFrame({"target_0": [0, 2, 0], "target_1": [2, 0, 1]}, index=["a", "b", "c"])


@fixture
def aggregated_sample(sample_data):
    """AggregatedSample fixture"""
    return AggregatedSample(xagg=sample_data)


def test_aggregated_sample_post_init(aggregated_sample, sample_data):
    """Test AggregatedSample post init"""
    assert aggregated_sample.raw.equals(sample_data)


def test_aggregated_sample_raw_getter(aggregated_sample, sample_data):
    """Test AggregatedSample raw getter"""
    assert aggregated_sample.raw.equals(sample_data)


def test_aggregated_sample_raw_setter(aggregated_sample):
    """Test AggregatedSample raw setter"""
    new_data = DataFrame({"target_0": [1, 1, 1], "target_1": [1, 1, 1]}, index=["a", "b", "c"])
    aggregated_sample.raw = new_data
    assert aggregated_sample.raw.equals(new_data)
    assert aggregated_sample.xagg.equals(new_data)


def test_aggregated_sample_shape(aggregated_sample):
    """Test AggregatedSample shape"""
    assert aggregated_sample.shape == (3, 2)


def test_aggregated_sample_index(aggregated_sample):
    """Test AggregatedSample index"""
    assert aggregated_sample.index.tolist() == ["a", "b", "c"]


def test_aggregated_sample_columns(aggregated_sample):
    """Test AggregatedSample columns"""
    assert aggregated_sample.columns.tolist() == ["target_0", "target_1"]


def test_aggregated_sample_values(aggregated_sample):
    """Test AggregatedSample values"""
    assert (aggregated_sample.values == aggregated_sample.xagg.values).all()


@fixture
def sample_data2():
    data = {"category": ["A", "A", "B", "B"], "value1": [10, 20, 30, 40], "value2": [1, 2, 3, 4]}
    xagg = DataFrame(data)
    return AggregatedSample(xagg=xagg)


def test_aggregated_sample_groupby_sum(sample_data2):
    """Test AggregatedSample groupby sum"""
    grouped = sample_data2.groupby("category").sum()
    expected_data = {"value1": [30, 70], "value2": [3, 7]}
    expected_index = ["A", "B"]
    expected = DataFrame(expected_data, index=expected_index)
    assert grouped.equals(expected)


def test_aggregated_sample_groupby_mean(sample_data2):
    """Test AggregatedSample groupby mean"""
    grouped = sample_data2.groupby("category").mean()
    expected_data = {"value1": [15.0, 35.0], "value2": [1.5, 3.5]}
    expected_index = ["A", "B"]
    expected = DataFrame(expected_data, index=expected_index)
    assert grouped.equals(expected)
