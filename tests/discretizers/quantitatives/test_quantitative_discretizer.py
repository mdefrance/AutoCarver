"""Set of tests for discretizers module."""

from numpy import inf, nan
from pandas import DataFrame, Series
from pytest import raises

from AutoCarver.discretizers.quantitatives.quantitative_discretizer import (
    QuantitativeDiscretizer,
    check_frequencies,
    check_quantitative_dtypes,
    min_value_counts,
)
from AutoCarver.discretizers.utils.base_discretizer import Sample
from AutoCarver.features import Features, GroupedList, QuantitativeFeature


def test_check_quantitative_dtypes_all_numeric():
    """Test check_quantitative_dtypes with all numeric columns"""
    df = DataFrame({"feature1": [1, 2, 3], "feature2": [4.0, 5.0, 6.0]})
    feature_versions = ["feature1", "feature2"]
    check_quantitative_dtypes(df, feature_versions, "test")


def test_check_quantitative_dtypes_non_numeric():
    """Test check_quantitative_dtypes with non-numeric columns"""
    df = DataFrame({"feature1": [1, 2, 3], "feature2": ["a", "b", "c"]})
    feature_versions = ["feature1", "feature2"]
    with raises(ValueError):
        check_quantitative_dtypes(df, feature_versions, "test")


def test_check_quantitative_dtypes_mixed_types():
    """Test check_quantitative_dtypes with mixed types"""
    df = DataFrame(
        {"feature1": [1, 2, 3], "feature2": [4.0, 5.0, 6.0], "feature3": [1.0, "b", "c"]}
    )
    feature_versions = ["feature1", "feature2", "feature3"]
    with raises(ValueError):
        check_quantitative_dtypes(df, feature_versions, "test")
    feature_versions = ["feature1", "feature2"]
    check_quantitative_dtypes(df, feature_versions, "test")


def test_min_value_counts_basic():
    """Test min_value_counts with basic input"""
    x = Series([1, 2, 2, 3, 3, 3], name="feature")
    features = Features([QuantitativeFeature(name="feature")])
    result = min_value_counts(x, features)
    assert result == 1 / 6  # Minimum frequency is 1/6


def test_min_value_counts_with_nans():
    """Test min_value_counts with NaN values"""
    # without dropna
    x = Series([1, 2, 2, 3, 3, 3, nan, nan], name="feature")
    features = Features([QuantitativeFeature(name="feature")])
    result = min_value_counts(x, features, dropna=False)
    assert result == 1 / 8  # Minimum frequency is 1/6

    # with dropna
    x = Series([1, 2, 2, 3, 3, 3, nan, nan], name="feature")
    features = Features([QuantitativeFeature(name="feature")])
    result = min_value_counts(x, features, dropna=True)
    assert result == 1 / 6  # Minimum frequency is 1/6


def test_min_value_counts_with_labels():
    """Test min_value_counts with predefined labels"""
    # with missing label
    x = Series(
        [
            "x <= 1.00e+00",
            "1.00e+00 < x <= 2.00e+00",
            "1.00e+00 < x <= 2.00e+00",
            "2.00e+00 < x <= 3.00e+00",
            "2.00e+00 < x <= 3.00e+00",
            "2.00e+00 < x <= 3.00e+00",
        ],
        name="feature",
    )
    feature = QuantitativeFeature(name="feature")
    feature.update(GroupedList([1, 2, 3, 4, inf]))
    features = Features([feature])
    result = min_value_counts(x, features)
    assert result == 0  # Label 4 has a frequency of 0

    # without missing label
    x = Series(
        [
            "x <= 1.00e+00",
            "1.00e+00 < x <= 2.00e+00",
            "1.00e+00 < x <= 2.00e+00",
            "2.00e+00 < x <= 3.00e+00",
            "2.00e+00 < x <= 3.00e+00",
            "3.00e+00 < x",
        ],
        name="feature",
    )
    feature = QuantitativeFeature(name="feature")
    feature.update(GroupedList([1, 2, 3, inf]))
    features = Features([feature])
    result = min_value_counts(x, features)
    assert result == 1 / 6


def test_min_value_counts_normalize_false():
    """Test min_value_counts with normalize set to False"""
    x = Series([1, 2, 2, 3, 3, 3], name="feature")
    features = Features([QuantitativeFeature(name="feature")])
    result = min_value_counts(x, features, normalize=False)
    assert result == 1  # Minimum count is 1


def test_check_frequencies_no_rare_modalities():
    """Test check_frequencies with no rare modalities"""
    df = DataFrame({"feature1": [1, 2, 2, 3, 3, 3], "feature2": [4, 4, 4, 5, 5, 5]})
    features = Features(
        [
            QuantitativeFeature(name="Feature 1", version="feature1"),
            QuantitativeFeature(name="Feature 2", version="feature2"),
        ]
    )
    half_min_freq = 0.1
    result = check_frequencies(df, features, half_min_freq)
    assert result == []  # No rare modalities


def test_check_frequencies_with_rare_modalities():
    """Test check_frequencies with rare modalities"""
    df = DataFrame({"feature1": [1, 1, 1, 1, 1, 2], "feature2": [4, 4, 4, 5, 5, 5]})
    features = Features(
        [
            QuantitativeFeature(name="Feature 1", version="feature1"),
            QuantitativeFeature(name="Feature 2", version="feature2"),
        ]
    )
    half_min_freq = 0.2
    result = check_frequencies(df, features, half_min_freq)
    assert len(result) == 1
    assert result[0].version == "feature1"  # Feature 1 has rare modalities


def test_check_frequencies_with_nans():
    """Test check_frequencies with NaN values"""
    df = DataFrame({"feature1": [1, 2, 2, 3, 3, nan], "feature2": [4, 4, 4, 5, 5, 5]})
    features = Features(
        [
            QuantitativeFeature(name="Feature 1", version="feature1"),
            QuantitativeFeature(name="Feature 2", version="feature2"),
        ]
    )
    half_min_freq = 0.1
    result = check_frequencies(df, features, half_min_freq)
    assert result == []  # No rare modalities even with NaNs


def test_check_frequencies_all_rare_modalities():
    """Test check_frequencies with all rare modalities"""
    df = DataFrame({"feature1": [1, 2, 3, 4, 5, 1], "feature2": [1, 2, 3, 4, 5, 2]})
    features = Features(
        [
            QuantitativeFeature(name="Feature 1", version="feature1"),
            QuantitativeFeature(name="Feature 2", version="feature2"),
        ]
    )
    half_min_freq = 0.5
    result = check_frequencies(df, features, half_min_freq)
    assert len(result) == 2
    assert result[0].version == "feature1"  # Feature 1 has rare modalities
    assert result[1].version == "feature2"  # Feature 2 has rare modalities


def test_quantitative_discretizer_initialization():
    """Tests QuantitativeDiscretizer initialization"""
    feature1 = QuantitativeFeature(name="feature1")
    feature2 = QuantitativeFeature(name="feature2")
    quantitatives = [feature1, feature2]
    min_freq = 0.05
    discretizer = QuantitativeDiscretizer(quantitatives=quantitatives, min_freq=min_freq)
    assert discretizer.min_freq == min_freq
    assert feature1 in discretizer.features
    assert feature2 in discretizer.features
    assert discretizer.half_min_freq == min_freq / 2


def test_quantitative_discretizer_prepare_data():
    """Tests _prepare_data method of QuantitativeDiscretizer"""
    quantitatives = [QuantitativeFeature(name="feature1"), QuantitativeFeature(name="feature2")]
    min_freq = 0.05
    discretizer = QuantitativeDiscretizer(quantitatives=quantitatives, min_freq=min_freq)

    data = {"feature1": [1, 2, 3, 4, 5], "feature2": [5.0, 4.0, 3.0, 2.0, 1.0]}
    df = DataFrame(data)
    y = Series([0, 1, 0, 1, 0])

    sample = discretizer._prepare_data(Sample(df, y))
    prepared_df = sample.X
    assert prepared_df["feature1"].dtype != object
    assert prepared_df["feature1"].tolist() == [1, 2, 3, 4, 5]
    assert prepared_df["feature2"].dtype != object
    assert prepared_df["feature2"].tolist() == [5.0, 4.0, 3.0, 2.0, 1.0]


def test_continuous_discretizer_fit():
    """Test fitting the QuantitativeDiscretizer"""
    feature1 = QuantitativeFeature(name="feature1")
    feature2 = QuantitativeFeature(name="feature2")
    quantitatives = [feature1, feature2]
    min_freq = 0.2
    discretizer = QuantitativeDiscretizer(quantitatives=quantitatives, min_freq=min_freq)
    y = Series([0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1])

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
    discretizer.fit(df, y)

    # Check if the features have been fitted
    assert feature1.has_nan is False
    assert feature2.has_nan is True
    print(feature1.content)
    print(feature2.content)
    assert feature1.content == {1: [1], 2: [2], 4: [4], inf: [inf]}
    assert feature2.content == {1.0: [1.0], 2.0: [2.0], inf: [5.0, inf]}

    # Check if the discretizer has been fitted
    transformed_df = discretizer.transform(df)
    print(transformed_df)
    expected = DataFrame(
        {
            "feature1": [
                "1.00e+00 < x <= 2.00e+00",
                "2.00e+00 < x <= 4.00e+00",
                "2.00e+00 < x <= 4.00e+00",
                "4.00e+00 < x",
                "x <= 1.00e+00",
                "x <= 1.00e+00",
                "x <= 1.00e+00",
                "x <= 1.00e+00",
                "x <= 1.00e+00",
                "x <= 1.00e+00",
                "1.00e+00 < x <= 2.00e+00",
                "1.00e+00 < x <= 2.00e+00",
                "1.00e+00 < x <= 2.00e+00",
                "1.00e+00 < x <= 2.00e+00",
                "1.00e+00 < x <= 2.00e+00",
                "1.00e+00 < x <= 2.00e+00",
                "2.00e+00 < x <= 4.00e+00",
                "2.00e+00 < x <= 4.00e+00",
                "4.00e+00 < x",
                "1.00e+00 < x <= 2.00e+00",
                "2.00e+00 < x <= 4.00e+00",
                "2.00e+00 < x <= 4.00e+00",
                "4.00e+00 < x",
            ],
            "feature2": [
                nan,
                "2.00e+00 < x",
                "2.00e+00 < x",
                nan,
                "x <= 1.00e+00",
                "x <= 1.00e+00",
                "x <= 1.00e+00",
                "x <= 1.00e+00",
                "x <= 1.00e+00",
                nan,
                nan,
                "1.00e+00 < x <= 2.00e+00",
                "1.00e+00 < x <= 2.00e+00",
                "1.00e+00 < x <= 2.00e+00",
                "1.00e+00 < x <= 2.00e+00",
                "1.00e+00 < x <= 2.00e+00",
                nan,
                nan,
                "2.00e+00 < x",
                nan,
                "2.00e+00 < x",
                "2.00e+00 < x",
                "2.00e+00 < x",
            ],
        }
    )
    assert transformed_df.equals(expected)


def test_quantitative_discretizer(x_train: DataFrame, target: str):
    """Tests QuantitativeDiscretizer

    Parameters
    ----------
    x_train : DataFrame
        Simulated Train DataFrame
    target: str
        Target feature
    """

    quantitatives = [
        "Quantitative",
        "Discrete_Quantitative",
        "Discrete_Quantitative_highnan",
        "Discrete_Quantitative_lownan",
        "Discrete_Quantitative_rarevalue",
    ]
    min_freq = 0.1
    features = Features(quantitatives=quantitatives)

    discretizer = QuantitativeDiscretizer(quantitatives=features, min_freq=min_freq)
    x_discretized = discretizer.fit_transform(x_train, x_train[target])

    assert not features("Discrete_Quantitative_lownan").values.contains(
        features("Discrete_Quantitative_lownan").nan
    ), "Missing order should not be grouped with ordinal_discretizer"

    assert all(
        x_discretized["Quantitative"].value_counts(normalize=True) >= min_freq
    ), "Non-nan value was not grouped"

    print(x_train.Discrete_Quantitative_rarevalue.value_counts(dropna=False, normalize=True))

    print(features("Discrete_Quantitative_rarevalue").content)
    assert features("Discrete_Quantitative_rarevalue").values == [
        1.0,
        2.0,
        3.0,
        4.0,
        inf,
    ], (
        "Rare values should be grouped to the closest one and inf should be kept whatsoever "
        "(OrdinalDiscretizer)"
    )
