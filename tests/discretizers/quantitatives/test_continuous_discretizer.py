"""Set of tests for quantitative_discretizers module."""

from numpy import inf, array, allclose, nan
from pandas import DataFrame
from pytest import raises
from AutoCarver.discretizers.quantitatives.continuous_discretizer import (
    ContinuousDiscretizer,
    np_find_quantiles,
    get_needed_quantiles,
    compute_quantiles,
)
from AutoCarver.features import Features


def test_get_needed_quantiles_empty_array():
    """Test get_needed_quantiles with an empty array"""
    df_feature = np.array([])
    q = 4
    len_df = 10
    result = get_needed_quantiles(df_feature, q, len_df)
    assert result == []


def test_get_needed_quantiles_basic():
    """Test get_needed_quantiles with basic input"""
    df_feature = np.array([1, 2, 3, 4, 5])
    q = 2
    len_df = 5
    result = get_needed_quantiles(df_feature, q, len_df)
    expected = [0.5]
    assert np.allclose(result, expected)


def test_get_needed_quantiles_large_q():
    """Test get_needed_quantiles with a large q value"""
    df_feature = np.array([1, 2, 3, 4, 5])
    q = 10
    len_df = 5
    result = get_needed_quantiles(df_feature, q, len_df)
    expected = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    assert np.allclose(result, expected)


def test_get_needed_quantiles_small_len_df():
    """Test get_needed_quantiles with a small len_df value"""
    df_feature = np.array([1, 2, 3, 4, 5])
    q = 2
    len_df = 2
    result = get_needed_quantiles(df_feature, q, len_df)
    expected = [0.25, 0.5, 0.75]
    assert np.allclose(result, expected)


def test_get_needed_quantiles_large_len_df():
    """Test get_needed_quantiles with a large len_df value"""
    df_feature = np.array([1, 2, 3, 4, 5])
    q = 2
    len_df = 10
    result = get_needed_quantiles(df_feature, q, len_df)
    expected = [0.5]
    assert np.allclose(result, expected)


def test_compute_quantiles_empty_array():
    """Test compute_quantiles with an empty array"""
    df_feature = array([])
    with raises(ValueError):
        compute_quantiles(df_feature, 1, 10)


def test_compute_quantiles_basic():
    """Test compute_quantiles with basic input"""
    df_feature = array([1, 2, 3, 4, 5])
    q = 2
    len_df = 5
    result = compute_quantiles(df_feature, q, len_df)
    expected = [2.5]
    assert allclose(result, expected)


def test_compute_quantiles_with_nans():
    """Test compute_quantiles with NaN values"""
    df_feature = array([1, 2, nan, 4, 5])
    q = 2
    len_df = 5
    result = compute_quantiles(df_feature, q, len_df)
    expected = [3.0]
    assert allclose(result, expected, equal_nan=True)


def test_compute_quantiles_large_q():
    """Test compute_quantiles with a large q value"""
    df_feature = array([1, 2, 3, 4, 5])
    q = 10
    len_df = 5
    result = compute_quantiles(df_feature, q, len_df)
    expected = [1.5, 2.5, 3.5, 4.5]
    assert allclose(result, expected)


def test_compute_quantiles_small_len_df():
    """Test compute_quantiles with a small len_df value"""
    df_feature = array([1, 2, 3, 4, 5])
    q = 2
    len_df = 2
    result = compute_quantiles(df_feature, q, len_df)
    expected = [2.5]
    assert allclose(result, expected)


def test_continuous_discretizer(x_train: DataFrame):
    """Tests ContinuousDiscretizer

    Parameters
    ----------
    x_train : DataFrame
        Simulated Train DataFrame
    """

    quantitatives = [
        "Quantitative",
        "Discrete_Quantitative",
        "Discrete_Quantitative_highnan",
        "Discrete_Quantitative_lownan",
        "Discrete_Quantitative_rarevalue",
    ]
    features = Features(quantitatives=quantitatives)
    min_freq = 0.1

    discretizer = ContinuousDiscretizer(
        features,
        min_freq,
        copy=True,
    )
    x_discretized = discretizer.fit(x_train)
    features.dropna = True
    x_discretized = discretizer.transform(x_train)
    features.dropna = False

    assert all(
        x_discretized.Quantitative.value_counts(normalize=True) == min_freq
    ), "Wrong quantiles"

    assert features("Discrete_Quantitative_highnan").values == [
        2.0,
        3.0,
        4.0,
        7.0,
        inf,
    ], "NaNs should not be added to the order"

    assert features("Discrete_Quantitative_highnan").has_nan, "Should have nan"

    assert features("Discrete_Quantitative_lownan").values == [
        1.0,
        2.0,
        3.0,
        4.0,
        6.0,
        inf,
    ], "NaNs should not be grouped whatsoever"

    assert features("Discrete_Quantitative_rarevalue").values == [
        0.5,
        1.0,
        2.0,
        3.0,
        4.0,
        6.0,
        inf,
    ], "Wrongly associated rare values"
