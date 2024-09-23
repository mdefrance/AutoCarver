"""Set of tests for discretizers module."""

from numpy import inf, nan
from pandas import DataFrame, Series, notna

from AutoCarver.config import DEFAULT
from AutoCarver.discretizers import Discretizer
from AutoCarver.features import CategoricalFeature, Features, OrdinalFeature, QuantitativeFeature


def test_discretizer_initialization():
    """Tests Discretizer initialization"""
    feature1 = QuantitativeFeature("feature1")
    feature2 = QuantitativeFeature("feature2")
    feature3 = CategoricalFeature("feature3")
    feature4 = OrdinalFeature("feature4", values=["a", "b", "c"])
    features = Features(
        quantitatives=[feature1, feature2], categoricals=[feature3], ordinals=[feature4]
    )
    min_freq = 0.05
    discretizer = Discretizer(features=features, min_freq=min_freq)
    assert discretizer.min_freq == min_freq
    assert feature1 in discretizer.features
    assert feature2 in discretizer.features
    assert feature3 in discretizer.features
    assert feature4 in discretizer.features
    assert discretizer.features == features


def test_discretizer_fit():
    """Tests Discretizer fit method"""
    feature1 = QuantitativeFeature("feature1")
    feature2 = QuantitativeFeature("feature2")
    feature3 = CategoricalFeature("feature3")
    feature4 = OrdinalFeature("feature4", values=["a", "b"])
    features = Features(
        quantitatives=[feature1, feature2], categoricals=[feature3], ordinals=[feature4]
    )
    min_freq = 0.3
    discretizer = Discretizer(features=features, min_freq=min_freq)

    data = {
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [5, 4, 3, nan, 1],
        "feature4": ["a", "b", "a", "b", nan],
        "feature3": [1, 4, 3, 4, nan],
    }
    df = DataFrame(data)
    y = Series([0, 1, 0, 1, 0])

    # fitting discretizer
    transformed_df = discretizer.fit_transform(df, y)

    print(transformed_df)

    data = {
        "feature1": [
            "x <= 2.0e+00",
            "x <= 2.0e+00",
            "2.0e+00 < x <= 3.0e+00",
            "3.0e+00 < x",
            "3.0e+00 < x",
        ],
        "feature2": [
            "3.0e+00 < x",
            "3.0e+00 < x",
            "x <= 3.0e+00",
            nan,
            "x <= 3.0e+00",
        ],
        "feature4": ["a", "b", "a", "b", nan],
        "feature3": ["1, 3", "4", "1, 3", "4", nan],
    }
    expected = DataFrame(data)
    assert transformed_df.equals(expected)


def test_discretizer(x_train: DataFrame, x_dev_1: DataFrame, target: str):
    """Tests Discretizer

    Parameters
    ----------
    x_train : DataFrame
        Simulated Train DataFrame
    x_dev_1 : DataFrame
        Simulated Test DataFrame
    target: str
        Target feature
    """

    quantitatives = [
        "Quantitative",
        "Discrete_Quantitative_highnan",
        "Discrete_Quantitative_lownan",
        "Discrete_Quantitative",
        "Discrete_Quantitative_rarevalue",
    ]
    categoricals = [
        "Qualitative",
        "Qualitative_grouped",
        "Qualitative_lownan",
        "Qualitative_highnan",
        "Discrete_Qualitative_noorder",
        "Discrete_Qualitative_lownan_noorder",
        "Discrete_Qualitative_rarevalue_noorder",
    ]
    ordinals = [
        "Qualitative_Ordinal",
        "Qualitative_Ordinal_lownan",
        "Discrete_Qualitative_highnan",
    ]
    ordinal_values = {
        "Qualitative_Ordinal": [
            "Low-",
            "Low",
            "Low+",
            "Medium-",
            "Medium",
            "Medium+",
            "High-",
            "High",
            "High+",
        ],
        "Qualitative_Ordinal_lownan": [
            "Low-",
            "Low",
            "Low+",
            "Medium-",
            "Medium",
            "Medium+",
            "High-",
            "High",
            "High+",
        ],
        "Discrete_Qualitative_highnan": ["1", "2", "3", "4", "5", "6", "7"],
    }
    features = Features(
        categoricals=categoricals,
        quantitatives=quantitatives,
        ordinals=ordinals,
        ordinal_values=ordinal_values,
    )

    # minimum frequency per modality + apply(find_common_modalities) outputs a Series
    min_freq = 0.1

    # discretizing features
    discretizer = Discretizer(min_freq=min_freq, features=features, copy=True)
    x_discretized = discretizer.fit_transform(x_train, x_train[target])
    x_dev_discretized = discretizer.transform(x_dev_1)

    assert all(
        x_discretized["Quantitative"].value_counts(normalize=True) >= min_freq
    ), "Non-nan value were not grouped"

    assert features("Discrete_Quantitative_lownan").values == [
        1.0,
        2.0,
        3.0,
        4.0,
        inf,
    ], "NaNs should not be grouped whatsoever"

    assert features("Discrete_Quantitative_rarevalue").values == [
        1.0,
        2.0,
        3.0,
        4.0,
        inf,
    ], "Rare values should be grouped to the closest one (OrdinalDiscretizer)"

    quali_expected = {
        DEFAULT: ["Category A", "Category D", "Category F", DEFAULT],
        "Category C": ["Category C"],
        "Category E": ["Category E"],
    }
    assert (
        features("Qualitative").content == quali_expected
    ), "Values less frequent than min_freq should be grouped into default_value"

    quali_lownan_expected = {
        DEFAULT: ["Category D", "Category F", DEFAULT],
        "Category C": ["Category C"],
        "Category E": ["Category E"],
    }
    assert (
        features("Qualitative_lownan").content == quali_lownan_expected
    ), "If any, NaN values should be put into str_nan and kept by themselves"

    expected_ordinal = {
        "Low+": ["Low-", "Low", "Low+"],
        "Medium-": ["Medium-"],
        "Medium": ["Medium"],
        "High": ["Medium+", "High-", "High"],
        "High+": ["High+"],
    }
    assert (
        features("Qualitative_Ordinal").content == expected_ordinal
    ), "Values not correctly grouped"

    expected_ordinal_lownan = {
        "Low+": ["Low", "Low-", "Low+"],
        "Medium-": ["Medium-"],
        "Medium": ["Medium"],
        "High": ["Medium+", "High-", "High"],
        "High+": ["High+"],
    }
    assert (
        features("Qualitative_Ordinal_lownan").content == expected_ordinal_lownan
    ), "NaNs should stay by themselves."

    # Testing out qualitative with int/float values inside -> StringDiscretizer
    expected = {
        "2": [2.0, "2"],
        "4": [4.0, "4"],
        "1": [1.0, "1"],
        "3": [3.0, "3"],
        DEFAULT: [0.5, "0.5", 6.0, "6", 5.0, "5", DEFAULT],
    }
    assert features("Discrete_Qualitative_rarevalue_noorder").content == expected, (
        "Qualitative features with float values should be converted to string and there values "
        "stored in the values_orders"
    )
    expected = {
        "2": [2, "2"],
        "4": [4, "4"],
        "1": [1, "1"],
        "3": [3, "3"],
        DEFAULT: [7, "7", 6, "6", 5, "5", DEFAULT],
    }
    assert features("Discrete_Qualitative_noorder").content == expected, (
        "Qualitative features with int values should be converted to string and there values stored"
        " in the values_orders"
    )
    expected = {
        "2": ["1", 2.0, "2"],
        "3": [3.0, "3"],
        "4": [4.0, "4"],
        "5": [6.0, "6", 7.0, "7", 5.0, "5"],
    }
    assert features("Discrete_Qualitative_highnan").content == expected, (
        "Ordinal qualitative features with int or float values that contain nan should be converted"
        " to string and there values stored in the values_orders"
    )

    # checking for inconsistancies in tranform
    for feature in features:
        # removing nans because they don't match
        test_unique = x_dev_discretized[feature.name].unique()
        test_unique = [val for val in test_unique if notna(val)]
        train_unique = x_discretized[feature.name].unique()
        train_unique = [val for val in train_unique if notna(val)]
        assert all(
            value in test_unique for value in train_unique
        ), "Missing value from test (at transform step)"
        assert all(
            value in train_unique for value in test_unique
        ), "Missing value from train (at transform step)"
