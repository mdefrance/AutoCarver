"""Set of tests for quantitative_discretizers module."""

from numpy import inf, array, allclose, nan
from pandas import DataFrame
from pytest import raises
from AutoCarver.discretizers.quantitatives.continuous_discretizer import (
    ContinuousDiscretizer,
    np_find_quantiles,
    find_quantiles,
    get_remaining_quantiles,
    compute_quantiles,
    fit_feature,
)
from AutoCarver.features import Features, QuantitativeFeature, GroupedList


def test_get_remaining_quantiles_no_remaining():
    """Test get_remaining_quantiles with no remaining length"""
    remaining_len_df = 0
    initial_len_df = 1000
    q = 4
    result = get_remaining_quantiles(remaining_len_df, initial_len_df, q)
    expected = []
    print(expected, result)
    assert len(result) == 0
    assert allclose(result, expected)


def test_get_remaining_quantiles_full_remaining():
    """Test get_remaining_quantiles with full remaining length"""
    # small q
    remaining_len_df = 1000
    initial_len_df = 1000
    q = 4
    result = get_remaining_quantiles(remaining_len_df, initial_len_df, q)
    expected = [0.25, 0.5, 0.75]
    assert len(result) == q - 1
    assert allclose(result, expected)

    # large q
    remaining_len_df = 1000
    initial_len_df = 1000
    q = 20
    result = get_remaining_quantiles(remaining_len_df, initial_len_df, q)
    expected = [
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
    ]
    print(expected, result)
    assert len(result) == q - 1
    assert allclose(result, expected)


def test_get_remaining_quantiles_edge_case():
    """Test get_remaining_quantiles with full remaining length"""
    # at threshold
    remaining_len_df = 375
    initial_len_df = 1000
    q = 4
    result = get_remaining_quantiles(remaining_len_df, initial_len_df, q)
    expected = [0.5]
    print(expected, result)
    assert len(result) == 1
    assert allclose(result, expected)

    # below threshold
    remaining_len_df = 374
    initial_len_df = 1000
    q = 4
    result = get_remaining_quantiles(remaining_len_df, initial_len_df, q)
    expected = []
    print(expected, result)
    assert len(result) == 0
    assert allclose(result, expected)


def test_compute_quantiles_empty_array():
    """Test compute_quantiles with an empty array"""
    df_feature = array([])
    q = 4
    len_df = 10
    with raises(ValueError):
        compute_quantiles(df_feature, q, len_df)


def test_compute_quantiles_one_remaining():
    """Test compute_quantiles with a large q value"""
    # with q = len_df
    df_feature = array([1, 2, 3, 4, 5])
    q = 5
    initial_len_df = 20
    result = compute_quantiles(df_feature, q, initial_len_df)
    expected = [max(df_feature)]
    assert allclose(result, expected)


def test_compute_quantiles_some_remaining():
    """Test compute_quantiles with a large q value"""
    df_feature = array([1, 2, 3, 4, 5, 6])
    q = 10
    initial_len_df = 20
    result = compute_quantiles(df_feature, q, initial_len_df)
    expected = [2, 4]
    assert allclose(result, expected)

    df_feature = array([1, 2, 3, 4, 5])
    result = compute_quantiles(df_feature, q, initial_len_df)
    expected = [3]
    assert allclose(result, expected)


def test_compute_quantiles_full_remaining():
    """Test compute_quantiles with a large q value"""
    # with q = len_df
    df_feature = array([1, 2, 3, 4, 5])
    q = 5
    initial_len_df = len(df_feature)
    result = compute_quantiles(df_feature, q, initial_len_df)
    expected = [1, 2, 3, 4]
    assert allclose(result, expected)

    # with q < len_df
    df_feature = array([1, 2, 3, 4, 5])
    q = 2
    initial_len_df = len(df_feature)
    result = compute_quantiles(df_feature, q, initial_len_df)
    expected = [3]
    assert allclose(result, expected)
    q = 3
    result = compute_quantiles(df_feature, q, initial_len_df)
    expected = [2, 3]
    assert allclose(result, expected)


def test_np_find_quantiles_empty_array():
    """Test np_find_quantiles with an empty array"""
    df_feature = array([])
    q = 4
    initial_len_df = 10
    quantiles = ["test"]
    result = np_find_quantiles(df_feature, q, initial_len_df, quantiles)
    assert result == quantiles  # Should return the initial quantiles list


def test_np_find_quantiles_no_overrepresented_value():
    """Test np_find_quantiles with no over-represented value"""
    df_feature = array([1, 2, 3, 4, 5])
    q = 2
    initial_len_df = 5
    quantiles = ["test"]
    result = np_find_quantiles(df_feature, q, initial_len_df, quantiles)
    expected = ["test", 3]
    assert result == expected


def test_np_find_quantiles_with_overrepresented_value():
    """Test np_find_quantiles with an over-represented value"""
    # with over-represented value at begining
    df_feature = array([1, 1, 1, 1, 1, 1, 2, 3, 4, 5])
    q = 4
    initial_len_df = len(df_feature)
    quantiles = ["test"]
    result = np_find_quantiles(df_feature, q, initial_len_df, quantiles)
    expected = ["test", 3, 1]
    assert result == expected

    # with over-represented value at end
    df_feature = array([2, 3, 4, 5, 1, 1, 1, 1, 1, 1])
    q = 4
    initial_len_df = len(df_feature)
    quantiles = ["test"]
    result = np_find_quantiles(df_feature, q, initial_len_df, quantiles)
    expected = ["test", 3, 1]
    print(expected, result)
    assert result == expected

    # with over-represented value at middle
    df_feature = array(
        [
            2,
            3,
            4,
            5,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            3,
            4,
            5,
            2,
            3,
            4,
            5,
        ]
    )
    q = 4
    initial_len_df = len(df_feature)
    quantiles = ["test"]
    result = np_find_quantiles(df_feature, q, initial_len_df, quantiles)
    expected = ["test", 3, 4, 1]
    print(expected, result)
    assert result == expected


def test_np_find_quantiles_with_multiple_overrepresented_values():
    """Test np_find_quantiles with multiple over-represented values"""
    # with sevveral over-represented value at begining
    df_feature = array(
        [
            2,
            3,
            4,
            5,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            3,
            4,
            5,
            2,
            3,
            4,
            5,
        ]
    )
    q = 4
    initial_len_df = len(df_feature)
    quantiles = []
    result = np_find_quantiles(df_feature, q, initial_len_df, quantiles)
    expected = [4, 1, 2]
    print(expected, result)
    assert result == expected


def test_find_quantiles_no_overrepresented_value():
    """Test find_quantiles with no over-represented value"""
    df_feature = array([1, 2, 3, 4, 5])
    q = 2
    result = find_quantiles(df_feature, q)
    expected = [3]
    assert result == expected


def test_find_quantiles_with_missing_values():
    """Test find_quantiles with missing values"""

    # with missing values
    df_feature = array([1, 2, 3, 4, nan])
    q = 2
    result = find_quantiles(df_feature, q)
    expected = [2]
    assert result == expected

    # with missing values and over-represented value
    df_feature = array([1, 1, 1, 1, 1, 1, 2, 3, 4, 5, nan])
    q = 4
    result = find_quantiles(df_feature, q)
    expected = [1, 5]
    assert result == expected

    # with missing values and over-represented value at end
    df_feature = array([2, 3, 4, 5, 2, 3, 4, 5, 1, 1, 1, 1, 1, 1, nan])
    q = 4
    result = find_quantiles(df_feature, q)
    expected = [1, 3]
    assert result == expected

    # with missing values and over-represented value at middle
    df_feature = array(
        [
            2,
            3,
            4,
            5,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            3,
            4,
            5,
            2,
            3,
            4,
            5,
            nan,
        ]
    )
    q = 4
    result = find_quantiles(df_feature, q)
    expected = [1, 3, 4]
    assert result == expected


def test_find_quantiles_with_overrepresented_value():
    """Test find_quantiles with an over-represented value"""
    # with over-represented value at begining
    df_feature = array([1, 1, 1, 1, 1, 1, 2, 3, 4, 5])
    q = 4
    result = find_quantiles(df_feature, q)
    expected = [1, 3]
    assert result == expected

    # with over-represented value at begining and not enough alue for a quantiles
    df_feature = array([1, 1, 1, 1, 1, 1, 4, 5])
    q = 4
    result = find_quantiles(df_feature, q)
    expected = [1, 5]
    assert result == expected

    # with over-represented value at end
    df_feature = array([2, 3, 4, 5, 1, 1, 1, 1, 1, 1])
    q = 4
    result = find_quantiles(df_feature, q)
    expected = [1, 3]
    print(expected, result)
    assert result == expected

    # with over-represented value at middle
    df_feature = array(
        [
            2,
            3,
            4,
            5,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            3,
            4,
            5,
            2,
            3,
            4,
            5,
        ]
    )
    q = 4
    result = find_quantiles(df_feature, q)
    expected = [1, 3, 4]
    assert result == expected


def test_find_quantiles_with_multiple_overrepresented_values():
    """Test find_quantiles with multiple over-represented values"""
    # with sevveral over-represented value at begining
    df_feature = array(
        [
            2,
            3,
            4,
            5,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            3,
            4,
            5,
            2,
            3,
            4,
            5,
        ]
    )
    q = 4
    result = find_quantiles(df_feature, q)
    expected = [1, 2, 4]
    print(expected, result)
    assert result == expected


def test_fit_feature_no_overrepresented_value():
    """Test fit_feature with no over-represented value"""
    feature = QuantitativeFeature("feature1")
    df_feature = DataFrame({"feature1": array([1, 2, 3, 4, 5])})
    q = 2
    result = fit_feature(feature, df_feature, q)
    expected = [3]
    assert result[0] == feature.version
    assert result[1] == GroupedList(expected + [inf])


def test_fit_feature_with_missing_values():
    """Test fit_feature with missing values"""

    # with missing values
    feature = QuantitativeFeature("feature1")
    df_feature = DataFrame({"feature1": array([1, 2, 3, 4, nan])})
    q = 2
    result = fit_feature(feature, df_feature, q)
    expected = [2]
    assert result[0] == feature.version
    assert result[1] == GroupedList(expected + [inf])

    # with missing values and over-represented value
    feature = QuantitativeFeature("feature1")
    df_feature = DataFrame({"feature1": array([1, 1, 1, 1, 1, 1, 2, 3, 4, 5, nan])})
    q = 4
    result = fit_feature(feature, df_feature, q)
    expected = [1, 5]
    assert result[0] == feature.version
    assert result[1] == GroupedList(expected + [inf])

    # with missing values and over-represented value at end
    feature = QuantitativeFeature("feature1")
    df_feature = DataFrame({"feature1": array([2, 3, 4, 5, 2, 3, 4, 5, 1, 1, 1, 1, 1, 1, nan])})
    q = 4
    result = fit_feature(feature, df_feature, q)
    expected = [1, 3]
    assert result[0] == feature.version
    assert result[1] == GroupedList(expected + [inf])

    # with missing values and over-represented value at middle
    feature = QuantitativeFeature("feature1")
    df_feature = DataFrame(
        {
            "feature1": array(
                [
                    2,
                    3,
                    4,
                    5,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    2,
                    3,
                    4,
                    5,
                    2,
                    3,
                    4,
                    5,
                    nan,
                ]
            )
        }
    )
    q = 4
    result = fit_feature(feature, df_feature, q)
    expected = [1, 3, 4]
    assert result[0] == feature.version
    assert result[1] == GroupedList(expected + [inf])


def test_fit_feature_with_overrepresented_value():
    """Test fit_feature with an over-represented value"""
    # with over-represented value at begining
    feature = QuantitativeFeature("feature1")
    df_feature = DataFrame({"feature1": array([1, 1, 1, 1, 1, 1, 2, 3, 4, 5])})
    q = 4
    result = fit_feature(feature, df_feature, q)
    expected = [1, 3]
    assert result[0] == feature.version
    assert result[1] == GroupedList(expected + [inf])

    # with over-represented value at begining and not enough alue for a quantiles
    feature = QuantitativeFeature("feature1")
    df_feature = DataFrame({"feature1": array([1, 1, 1, 1, 1, 1, 4, 5])})
    q = 4
    result = fit_feature(feature, df_feature, q)
    expected = [1, 5]
    assert result[0] == feature.version
    assert result[1] == GroupedList(expected + [inf])

    # with over-represented value at end
    feature = QuantitativeFeature("feature1")
    df_feature = DataFrame({"feature1": array([2, 3, 4, 5, 1, 1, 1, 1, 1, 1])})
    q = 4
    result = fit_feature(feature, df_feature, q)
    expected = [1, 3]
    print(expected, result)
    assert result[0] == feature.version
    assert result[1] == GroupedList(expected + [inf])

    # with over-represented value at middle
    feature = QuantitativeFeature("feature1")
    df_feature = DataFrame(
        {
            "feature1": array(
                [
                    2,
                    3,
                    4,
                    5,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    2,
                    3,
                    4,
                    5,
                    2,
                    3,
                    4,
                    5,
                ]
            )
        }
    )
    q = 4
    result = fit_feature(feature, df_feature, q)
    expected = [1, 3, 4]
    assert result[0] == feature.version
    assert result[1] == GroupedList(expected + [inf])


def test_fit_feature_with_multiple_overrepresented_values():
    """Test fit_feature with multiple over-represented values"""
    # with sevveral over-represented value at begining
    feature = QuantitativeFeature("feature1")
    df_feature = DataFrame(
        {
            "feature1": array(
                [
                    2,
                    3,
                    4,
                    5,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    2,
                    2,
                    2,
                    2,
                    2,
                    2,
                    3,
                    4,
                    5,
                    2,
                    3,
                    4,
                    5,
                ]
            )
        }
    )
    q = 4
    result = fit_feature(feature, df_feature, q)
    expected = [1, 2, 4]
    print(expected, result)
    assert result[0] == feature.version
    assert result[1] == GroupedList(expected + [inf])


def test_continuous_discretizer_initialization():
    """Test initialization of ContinuousDiscretizer"""
    feature1 = QuantitativeFeature(name="feature1")
    feature2 = QuantitativeFeature(name="feature2")
    quantitatives = [feature1, feature2]
    min_freq = 0.05
    discretizer = ContinuousDiscretizer(quantitatives=quantitatives, min_freq=min_freq)
    assert discretizer.min_freq == min_freq
    assert feature1 in discretizer.features
    assert feature2 in discretizer.features
    assert "feature1" in discretizer.features
    assert "feature2" in discretizer.features

    # test_continuous_discretizer_q
    assert discretizer.q == round(1 / min_freq)


def test_continuous_discretizer_fit():
    """Test fitting the ContinuousDiscretizer"""
    feature1 = QuantitativeFeature(name="feature1")
    feature2 = QuantitativeFeature(name="feature2")
    quantitatives = [feature1, feature2]
    min_freq = 0.2
    discretizer = ContinuousDiscretizer(quantitatives=quantitatives, min_freq=min_freq)

    # Create a sample DataFrame
    data = {
        "feature1": [
            2,
            3,
            4,
            5,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            3,
            4,
            5,
            2,
            3,
            4,
            5,
        ],
        "feature2": [
            nan,
            3,
            4,
            nan,
            1,
            1,
            1,
            1,
            1,
            nan,
            nan,
            2,
            2,
            2,
            2,
            2,
            nan,
            nan,
            5,
            nan,
            3,
            4,
            5,
        ],
    }

    df = DataFrame(data)

    # Fit the discretizer
    discretizer.fit(df)

    # Check if the features have been fitted
    assert feature1.has_nan == False
    assert feature2.has_nan == True
    assert feature1.content == {1: [1], 2: [2], 4: [4], inf: [inf]}
    assert feature2.content == {1.0: [1.0], 2.0: [2.0], 5.0: [5.0], inf: [inf]}

    # Check if the discretizer has been fitted
    transformed_df = discretizer.transform(df)
    print(transformed_df)
    expected = DataFrame(
        {
            "feature1": [
                "1.0e+00 < x <= 2.0e+00",
                "2.0e+00 < x <= 4.0e+00",
                "2.0e+00 < x <= 4.0e+00",
                "4.0e+00 < x",
                "x <= 1.0e+00",
                "x <= 1.0e+00",
                "x <= 1.0e+00",
                "x <= 1.0e+00",
                "x <= 1.0e+00",
                "x <= 1.0e+00",
                "1.0e+00 < x <= 2.0e+00",
                "1.0e+00 < x <= 2.0e+00",
                "1.0e+00 < x <= 2.0e+00",
                "1.0e+00 < x <= 2.0e+00",
                "1.0e+00 < x <= 2.0e+00",
                "1.0e+00 < x <= 2.0e+00",
                "2.0e+00 < x <= 4.0e+00",
                "2.0e+00 < x <= 4.0e+00",
                "4.0e+00 < x",
                "1.0e+00 < x <= 2.0e+00",
                "2.0e+00 < x <= 4.0e+00",
                "2.0e+00 < x <= 4.0e+00",
                "4.0e+00 < x",
            ],
            "feature2": [
                nan,
                "2.0e+00 < x <= 5.0e+00",
                "2.0e+00 < x <= 5.0e+00",
                nan,
                "x <= 1.0e+00",
                "x <= 1.0e+00",
                "x <= 1.0e+00",
                "x <= 1.0e+00",
                "x <= 1.0e+00",
                nan,
                nan,
                "1.0e+00 < x <= 2.0e+00",
                "1.0e+00 < x <= 2.0e+00",
                "1.0e+00 < x <= 2.0e+00",
                "1.0e+00 < x <= 2.0e+00",
                "1.0e+00 < x <= 2.0e+00",
                nan,
                nan,
                "2.0e+00 < x <= 5.0e+00",
                nan,
                "2.0e+00 < x <= 5.0e+00",
                "2.0e+00 < x <= 5.0e+00",
                "2.0e+00 < x <= 5.0e+00",
            ],
        }
    )
    assert transformed_df.equals(expected)


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
