"""Set of tests for quantitative_discretizers module."""

import numpy as np
import pandas as pd

from AutoCarver.discretizers.quantitatives.continuous_discretizer import (
    ContinuousDiscretizer,
    find_quantiles,
    fit_feature,
)
from AutoCarver.discretizers.utils.base_discretizer import ProcessingConfig
from AutoCarver.features import Features, GroupedList, QuantitativeFeature


def test_find_quantiles_no_overrepresented_value():
    """Test find_quantiles with no over-represented value"""
    df_feature = np.array([1, 2, 3, 4, 5])
    q = 2
    result = find_quantiles(df_feature, q)
    expected = [3]
    assert result == expected


def test_find_quantiles_with_missing_values():
    """Test find_quantiles with missing values"""

    # with missing values
    df_feature = np.array([1, 2, 3, 4, np.nan])
    q = 2
    result = find_quantiles(df_feature, q)
    expected = [2]
    assert result == expected

    # with missing values and over-represented value
    df_feature = np.array([1, 1, 1, 1, 1, 1, 2, 3, 4, 5, np.nan])
    q = 4
    result = find_quantiles(df_feature, q)
    expected = [1, 5]
    assert result == expected

    # with missing values and over-represented value at end
    df_feature = np.array([2, 3, 4, 5, 2, 3, 4, 5, 1, 1, 1, 1, 1, 1, np.nan])
    q = 4
    result = find_quantiles(df_feature, q)
    expected = [1, 3]
    assert result == expected

    # with missing values and over-represented value at middle
    df_feature = np.array(
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
            np.nan,
        ]
    )
    q = 4
    result = find_quantiles(df_feature, q)
    expected = [1, 3, 4]
    assert result == expected


def test_find_quantiles_with_overrepresented_value():
    """Test find_quantiles with an over-represented value"""
    # with over-represented value at begining
    df_feature = np.array([1, 1, 1, 1, 1, 1, 2, 3, 4, 5])
    q = 4
    result = find_quantiles(df_feature, q)
    expected = [1, 3]
    assert result == expected

    # with over-represented value at begining and not enough alue for a quantiles
    df_feature = np.array([1, 1, 1, 1, 1, 1, 4, 5])
    q = 4
    result = find_quantiles(df_feature, q)
    expected = [1, 5]
    assert result == expected

    # with over-represented value at end
    df_feature = np.array([2, 3, 4, 5, 1, 1, 1, 1, 1, 1])
    q = 4
    result = find_quantiles(df_feature, q)
    expected = [1, 3]
    print(expected, result)
    assert result == expected

    # with over-represented value at middle
    df_feature = np.array(
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
    df_feature = np.array(
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
    df_feature = pd.DataFrame({"feature1": np.array([1, 2, 3, 4, 5])})
    q = 2
    result = fit_feature(feature, df_feature, q)
    expected = [3]
    assert result[0] == feature.version
    assert result[1] == GroupedList(expected + [np.inf])


def test_fit_feature_with_missing_values():
    """Test fit_feature with missing values"""

    # with missing values
    feature = QuantitativeFeature("feature1")
    df_feature = pd.DataFrame({"feature1": np.array([1, 2, 3, 4, np.nan])})
    q = 2
    result = fit_feature(feature, df_feature, q)
    expected = [2]
    assert result[0] == feature.version
    assert result[1] == GroupedList(expected + [np.inf])

    # with missing values and over-represented value
    feature = QuantitativeFeature("feature1")
    df_feature = pd.DataFrame({"feature1": np.array([1, 1, 1, 1, 1, 1, 2, 3, 4, 5, np.nan])})
    q = 4
    result = fit_feature(feature, df_feature, q)
    expected = [1, 5]
    assert result[0] == feature.version
    assert result[1] == GroupedList(expected + [np.inf])

    # with missing values and over-represented value at end
    feature = QuantitativeFeature("feature1")
    df_feature = pd.DataFrame({"feature1": np.array([2, 3, 4, 5, 2, 3, 4, 5, 1, 1, 1, 1, 1, 1, np.nan])})
    q = 4
    result = fit_feature(feature, df_feature, q)
    expected = [1, 3]
    assert result[0] == feature.version
    assert result[1] == GroupedList(expected + [np.inf])

    # with missing values and over-represented value at middle
    feature = QuantitativeFeature("feature1")
    df_feature = pd.DataFrame(
        {
            "feature1": np.array(
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
                    np.nan,
                ]
            )
        }
    )
    q = 4
    result = fit_feature(feature, df_feature, q)
    expected = [1, 3, 4]
    assert result[0] == feature.version
    assert result[1] == GroupedList(expected + [np.inf])


def test_fit_feature_with_overrepresented_value():
    """Test fit_feature with an over-represented value"""
    # with over-represented value at begining
    feature = QuantitativeFeature("feature1")
    df_feature = pd.DataFrame({"feature1": np.array([1, 1, 1, 1, 1, 1, 2, 3, 4, 5])})
    q = 4
    result = fit_feature(feature, df_feature, q)
    expected = [1, 3]
    assert result[0] == feature.version
    assert result[1] == GroupedList(expected + [np.inf])

    # with over-represented value at begining and not enough alue for a quantiles
    feature = QuantitativeFeature("feature1")
    df_feature = pd.DataFrame({"feature1": np.array([1, 1, 1, 1, 1, 1, 4, 5])})
    q = 4
    result = fit_feature(feature, df_feature, q)
    expected = [1, 5]
    assert result[0] == feature.version
    assert result[1] == GroupedList(expected + [np.inf])

    # with over-represented value at end
    feature = QuantitativeFeature("feature1")
    df_feature = pd.DataFrame({"feature1": np.array([2, 3, 4, 5, 1, 1, 1, 1, 1, 1])})
    q = 4
    result = fit_feature(feature, df_feature, q)
    expected = [1, 3]
    print(expected, result)
    assert result[0] == feature.version
    assert result[1] == GroupedList(expected + [np.inf])

    # with over-represented value at middle
    feature = QuantitativeFeature("feature1")
    df_feature = pd.DataFrame(
        {
            "feature1": np.array(
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
    assert result[1] == GroupedList(expected + [np.inf])


def test_fit_feature_with_multiple_overrepresented_values():
    """Test fit_feature with multiple over-represented values"""
    # with sevveral over-represented value at begining
    feature = QuantitativeFeature("feature1")
    df_feature = pd.DataFrame(
        {
            "feature1": np.array(
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
    assert result[1] == GroupedList(expected + [np.inf])


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
            np.nan,
            3,
            4,
            np.nan,
            1,
            1,
            1,
            1,
            1,
            np.nan,
            np.nan,
            2,
            2,
            2,
            2,
            2,
            np.nan,
            np.nan,
            5,
            np.nan,
            3,
            4,
            5,
        ],
    }

    df = pd.DataFrame(data)

    # Fit the discretizer
    discretizer.fit(df)

    # Check if the features have been fitted
    assert feature1.has_nan is False
    assert feature2.has_nan is True
    assert feature1.content == {1: [1], 2: [2], 4: [4], np.inf: [np.inf]}
    assert feature2.content == {1.0: [1.0], 2.0: [2.0], 5.0: [5.0], np.inf: [np.inf]}

    # Check if the discretizer has been fitted
    transformed_df = discretizer.transform(df)
    print(transformed_df)
    expected = pd.DataFrame(
        {
            "feature1": [
                "(1.00e+00, 2.00e+00]",
                "(2.00e+00, 4.00e+00]",
                "(2.00e+00, 4.00e+00]",
                "(4.00e+00, inf)",
                "(-inf, 1.00e+00]",
                "(-inf, 1.00e+00]",
                "(-inf, 1.00e+00]",
                "(-inf, 1.00e+00]",
                "(-inf, 1.00e+00]",
                "(-inf, 1.00e+00]",
                "(1.00e+00, 2.00e+00]",
                "(1.00e+00, 2.00e+00]",
                "(1.00e+00, 2.00e+00]",
                "(1.00e+00, 2.00e+00]",
                "(1.00e+00, 2.00e+00]",
                "(1.00e+00, 2.00e+00]",
                "(2.00e+00, 4.00e+00]",
                "(2.00e+00, 4.00e+00]",
                "(4.00e+00, inf)",
                "(1.00e+00, 2.00e+00]",
                "(2.00e+00, 4.00e+00]",
                "(2.00e+00, 4.00e+00]",
                "(4.00e+00, inf)",
            ],
            "feature2": [
                np.nan,
                "(2.00e+00, 5.00e+00]",
                "(2.00e+00, 5.00e+00]",
                np.nan,
                "(-inf, 1.00e+00]",
                "(-inf, 1.00e+00]",
                "(-inf, 1.00e+00]",
                "(-inf, 1.00e+00]",
                "(-inf, 1.00e+00]",
                np.nan,
                np.nan,
                "(1.00e+00, 2.00e+00]",
                "(1.00e+00, 2.00e+00]",
                "(1.00e+00, 2.00e+00]",
                "(1.00e+00, 2.00e+00]",
                "(1.00e+00, 2.00e+00]",
                np.nan,
                np.nan,
                "(2.00e+00, 5.00e+00]",
                np.nan,
                "(2.00e+00, 5.00e+00]",
                "(2.00e+00, 5.00e+00]",
                "(2.00e+00, 5.00e+00]",
            ],
        }
    )
    assert transformed_df.equals(expected)


def test_continuous_discretizer(x_train: pd.DataFrame):
    """Tests ContinuousDiscretizer

    Parameters
    ----------
    x_train : pd.DataFrame
        Simulated Train DataFrame
    """

    quantitatives = [
        "Quantitative",
        "Discrete_Quantitative",
        "Discrete_Quantitative_highnan",
        "Discrete_Quantitative_lownan",
        "Discrete_Quantitative_rarevalue",
    ]
    features = Features(numericals=quantitatives)
    min_freq = 0.1

    discretizer = ContinuousDiscretizer(
        features,
        min_freq,
        config=ProcessingConfig(copy=True),
    )
    x_discretized = discretizer.fit(x_train)
    features.dropna = True
    x_discretized = discretizer.transform(x_train)
    features.dropna = False

    assert all(x_discretized.Quantitative.value_counts(normalize=True) == min_freq), "Wrong quantiles"

    assert features("Discrete_Quantitative_highnan").values == [
        2.0,
        3.0,
        4.0,
        7.0,
        np.inf,
    ], "NaNs should not be added to the order"

    assert features("Discrete_Quantitative_highnan").has_nan, "Should have np.nan"

    assert features("Discrete_Quantitative_lownan").values == [
        1.0,
        2.0,
        3.0,
        4.0,
        6.0,
        np.inf,
    ], "NaNs should not be grouped whatsoever"

    assert features("Discrete_Quantitative_rarevalue").values == [
        0.5,
        1.0,
        2.0,
        3.0,
        4.0,
        6.0,
        np.inf,
    ], "Wrongly associated rare values"
