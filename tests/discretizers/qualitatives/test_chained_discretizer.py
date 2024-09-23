"""Set of tests for qualitative_discretizers module."""

from numpy import isnan, nan
from pandas import DataFrame
from pytest import raises

from AutoCarver.discretizers import ChainedDiscretizer
from AutoCarver.discretizers.qualitatives.chained_discretizer import (
    ChainedDiscretizer,
    check_frequencies,
    ensure_qualitative_dtypes,
)
from AutoCarver.features import CategoricalFeature, Features, OrdinalFeature


def test_check_frequencies_basic():
    """Test check_frequencies with basic input"""
    features = Features(["feature1", "feature2"])
    df = DataFrame(
        {"feature1": ["A", "B", "A", "C", "B", "A"], "feature2": ["X", "Y", "X", "Z", "Y", "X"]}
    )
    min_freq = 1 / 6
    check_frequencies(features, df, min_freq, "Test")


def test_check_frequencies_no_common_modality():
    """Test check_frequencies with no common modality"""
    features = Features(["feature1"])
    df = DataFrame({"feature1": ["A", "B", "C", "D", "E", "F"]})
    min_freq = 0.2
    with raises(ValueError):
        check_frequencies(features, df, min_freq, "Test")


def test_check_frequencies_too_common_modality():
    """Test check_frequencies with too common modality"""
    features = Features(["feature1"])
    df = DataFrame({"feature1": ["A", "A", "A", "A", "A", "B"]})
    min_freq = 0.2
    with raises(ValueError):
        check_frequencies(features, df, min_freq, "Test")


def test_check_frequencies_edge_case():
    """Test check_frequencies with edge case"""
    features = Features(["feature1"])
    df = DataFrame({"feature1": ["A", "A", "A", "A", "A", "A"]})
    min_freq = 0.2
    with raises(ValueError):
        check_frequencies(features, df, min_freq, "Test")


def test_ensure_qualitative_dtypes_with_conversion():
    """Test ensure_qualitative_dtypes with basic input"""
    features = Features(
        [CategoricalFeature("feature1"), OrdinalFeature("feature2", ["A", "B", "C"])]
    )
    df = DataFrame({"feature1": [1, 2, 3], "feature2": ["A", "B", "C"]})
    result = ensure_qualitative_dtypes(features, df)
    assert result["feature1"].dtype == object
    assert result["feature1"].tolist() == ["1", "2", "3"]
    assert result["feature2"].dtype == object
    assert result["feature2"].tolist() == ["A", "B", "C"]


def test_ensure_qualitative_dtypes_without_conversion():
    """Test ensure_qualitative_dtypes with basic input"""
    features = Features(
        [CategoricalFeature("feature1"), OrdinalFeature("feature2", ["A", "B", "C"])]
    )
    df = DataFrame({"feature1": ["1", "2", "3"], "feature2": ["A", "B", "C"]})
    result = ensure_qualitative_dtypes(features, df)
    assert result["feature1"].dtype == object
    assert result["feature1"].tolist() == ["1", "2", "3"]
    assert result["feature2"].dtype == object
    assert result["feature2"].tolist() == ["A", "B", "C"]


def test_ensure_qualitative_dtypes_with_nans():
    """Test ensure_qualitative_dtypes with NaN values"""
    features = Features(
        [CategoricalFeature("feature1"), OrdinalFeature("feature2", ["A", "B", "C"])]
    )
    df = DataFrame({"feature1": [1, nan, 3], "feature2": ["A", "B", nan]})
    result = ensure_qualitative_dtypes(features, df)

    feature1_list = result["feature1"].tolist()
    feature2_list = result["feature2"].tolist()

    assert feature1_list[0] == "1"
    assert isnan(feature1_list[1])
    assert feature1_list[2] == "3"

    assert feature2_list[0] == "A"
    assert feature2_list[1] == "B"
    assert isnan(feature2_list[2])


def test_ensure_qualitative_dtypes_mixed_types():
    """Test ensure_qualitative_dtypes with mixed data types"""
    features = Features(
        [CategoricalFeature("feature1"), OrdinalFeature("feature2", ["A", "2", "B"])]
    )
    df = DataFrame({"feature1": [1, 3.0, 3.5], "feature2": ["A", 2.0, "B"]})
    result = ensure_qualitative_dtypes(features, df)
    assert result["feature1"].dtype == object
    assert result["feature1"].tolist() == ["1", "3", "3.5"]
    assert result["feature2"].dtype == object
    assert result["feature2"].tolist() == ["A", "2", "B"]


def test_chained_discretizer(x_train: DataFrame) -> None:
    """Tests ChainedDiscretizer

    Parameters
    ----------
    x_train : DataFrame
        Simulated Train DataFrame
    """
    # Simulating some datasets with unknown values
    x_train_wrong_1 = x_train.copy()
    x_train_wrong_2 = x_train.copy()

    # minimum frequency per value
    min_freq = 0.15

    # building raising datasets
    x_train_wrong_1["Qualitative_Ordinal"] = x_train["Qualitative_Ordinal"].replace(
        "Medium", "unknown"
    )
    x_train_wrong_2["Qualitative_Ordinal_lownan"] = x_train["Qualitative_Ordinal_lownan"].replace(
        "Medium", "unknown"
    )

    chained_features = [
        "Qualitative_Ordinal",
        "Qualitative_Ordinal_lownan",
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
            "Low+",
            "Medium-",
            "Medium",
            "Medium+",
            "High-",
            "High",
            "High+",
        ],
    }

    # chained orders
    level0_to_level1 = {
        "Lows": ["Low-", "Low", "Low+", "Lows"],
        "Mediums": ["Medium-", "Medium", "Medium+", "Mediums"],
        "Highs": ["High-", "High", "High+", "Highs"],
    }
    level1_to_level2 = {
        "Worst": ["Lows", "Mediums", "Worst"],
        "Best": ["Highs", "Best"],
    }

    # defining features
    features = Features(
        ordinals=chained_features,
        ordinal_values=ordinal_values,
    )

    # fitting discretizer
    discretizer = ChainedDiscretizer(
        min_freq=min_freq, features=features, chained_orders=[level0_to_level1, level1_to_level2]
    )
    _ = discretizer.fit_transform(x_train)

    expected = {
        "High+": ["High+"],
        "Best": ["High", "High-", "Highs", "Best"],
        "Mediums": ["Medium+", "Medium-", "Mediums"],
        "Medium": ["Medium"],
        "Worst": ["Low+", "Low", "Low-", "Lows", "Worst"],
    }
    feature = "Qualitative_Ordinal"
    assert (
        discretizer.features(feature).content == expected
    ), "Values less frequent than min_freq should be grouped"

    assert discretizer.features(feature).values == [
        "Medium",
        "Mediums",
        "Worst",
        "High+",
        "Best",
    ], "Order of ordinal features is wrong"

    expected = {
        "Medium": ["Medium"],
        "High+": ["High+"],
        "Best": ["High", "High-", "Highs", "Best"],
        "Mediums": ["Medium+", "Medium-", "Mediums"],
        "Worst": ["Low+", "Low", "Low-", "Lows", "Worst"],
    }

    feature = "Qualitative_Ordinal_lownan"
    assert discretizer.features(feature).content == expected, (
        "NaNs should be added to the order and missing values from the values_orders should"
        " be added (from chained_orders)"
    )

    expected = [
        "Medium",
        "Mediums",
        "Worst",
        "High+",
        "Best",
    ]
    assert discretizer.features(feature).values == expected, "Order of ordinal features is wrong"

    # testing to fit when unknown values present
    with raises(ValueError):
        discretizer.fit_transform(x_train_wrong_1)

    with raises(ValueError):
        discretizer.fit_transform(x_train_wrong_2)

    # defining features
    features = Features(categoricals=chained_features)

    # testing with categorical features
    discretizer = ChainedDiscretizer(
        min_freq=min_freq, features=features, chained_orders=[level0_to_level1, level1_to_level2]
    )
    _ = discretizer.fit_transform(x_train)

    expected = {
        "High+": ["High+"],
        "Best": ["High", "High-", "Highs", "Best"],
        "Mediums": ["Medium+", "Medium-", "Mediums"],
        "Medium": ["Medium"],
        "Worst": ["Low+", "Low", "Low-", "Lows", "Worst"],
    }
    feature = "Qualitative_Ordinal"
    assert (
        discretizer.features(feature).content == expected
    ), "Values less frequent than min_freq should be grouped"

    assert discretizer.features(feature).values == [
        "Medium",
        "Mediums",
        "Worst",
        "High+",
        "Best",
    ], "Order of ordinal features is wrong"

    expected = {
        "Medium": ["Medium"],
        "High+": ["High+"],
        "Best": ["High", "High-", "Highs", "Best"],
        "Mediums": ["Medium+", "Medium-", "Mediums"],
        "Worst": ["Low+", "Low", "Low-", "Lows", "Worst"],
    }

    feature = "Qualitative_Ordinal_lownan"
    assert discretizer.features(feature).content == expected, (
        "NaNs should be added to the order and missing values from the values_orders should"
        " be added (from chained_orders)"
    )

    expected = [
        "Medium",
        "Mediums",
        "Worst",
        "High+",
        "Best",
    ]
    assert discretizer.features(feature).values == expected, "Order of ordinal features is wrong"

    # checking that unknown provided levels are correctly kept
    level0_to_level1 = {
        "Lows": ["Low-", "Low", "Low+", "Lows"],
        "Mediums": ["Medium-", "Medium", "Medium+", "Mediums"],
        "Highs": ["High-", "High", "High+", "Highs"],
        "ALONE": ["ALL_ALONE", "ALONE"],
    }
    level1_to_level2 = {
        "Worst": ["Lows", "Mediums", "Worst"],
        "Best": ["Highs", "Best"],
        "BEST": ["ALONE", "BEST"],
    }

    # defining features
    features = Features(
        ordinals=chained_features,
        ordinal_values=ordinal_values,
    )

    # fitting discretizer
    discretizer = ChainedDiscretizer(
        min_freq=min_freq, features=features, chained_orders=[level0_to_level1, level1_to_level2]
    )
    _ = discretizer.fit_transform(x_train)

    feature = "Qualitative_Ordinal_lownan"
    expected = {
        "Medium": ["Medium"],
        "Mediums": ["Medium+", "Medium-", "Mediums"],
        "Worst": ["Low+", "Low", "Low-", "Lows", "Worst"],
        "High+": ["High+"],
        "Best": ["High", "High-", "Highs", "Best"],
        "BEST": ["ALL_ALONE", "ALONE", "BEST"],
    }
    assert discretizer.features(feature).content == expected, "All provided values should be kept."

    feature = "Qualitative_Ordinal"
    expected = {
        "Medium": ["Medium"],
        "Mediums": ["Medium+", "Medium-", "Mediums"],
        "Worst": ["Low+", "Low", "Low-", "Lows", "Worst"],
        "High+": ["High+"],
        "Best": ["High", "High-", "Highs", "Best"],
        "BEST": ["ALL_ALONE", "ALONE", "BEST"],
    }
    assert discretizer.features(feature).content == expected, "All provided values should be kept."

    # testing for good defintion of levels
    with raises(ValueError):
        level0_to_level1 = {
            "Lows": ["Low-", "Low", "Low+", "Lows"],
            "Mediums": ["Medium-", "Medium", "Medium+", "Mediums"],
            "Highs": ["High-", "High", "High+", "Highs"],
        }
        level1_to_level2 = {
            "Worst": ["Lows", "Mediums", "Worst"],
            "Best": ["Highs", "Best"],
            "BEST": ["ALONE", "BEST"],
        }

        # defining features
        features = Features(
            ordinals=chained_features,
            ordinal_values=ordinal_values,
        )

        # fitting discretizer
        discretizer = ChainedDiscretizer(
            min_freq=min_freq,
            features=features,
            chained_orders=[level0_to_level1, level1_to_level2],
        )
        _ = discretizer.fit_transform(x_train)

    # testing for defintion of levels
    with raises(ValueError):
        level0_to_level1 = {
            "Lows": ["Low-", "Low", "Low+", "Lows"],
            "Mediums": ["Medium-", "Medium", "Medium+", "Mediums"],
            "Highs": ["High-", "High", "High+", "Highs"],
        }
        level1_to_level2 = {
            "Worst": ["Lows", "Mediums", "Worst"],
            "Best": ["Highs", "Best"],
            "BEST": ["BEST"],
        }

        # defining features
        features = Features(
            ordinals=chained_features,
            ordinal_values=ordinal_values,
        )

        # fitting discretizer
        discretizer = ChainedDiscretizer(
            min_freq=min_freq,
            features=features,
            chained_orders=[level0_to_level1, level1_to_level2],
        )
        _ = discretizer.fit_transform(x_train)

    # testing that it does not work when there is a val in values_orders missing from chained_orders
    with raises(ValueError):
        ordinal_values = {
            "Qualitative_Ordinal_lownan": [
                "-Low",
                "Low+",
                "Medium-",
                "Medium",
                "Medium+",
                "High-",
                "High",
                "High+",
            ],
            "Qualitative_Ordinal_highnan": [
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
        }

        level0_to_level1 = {
            "Lows": ["Low-", "Low", "Low+", "Lows"],
            "Mediums": ["Medium-", "Medium", "Medium+", "Mediums"],
            "Highs": ["High-", "High", "High+", "Highs"],
        }
        level1_to_level2 = {
            "Worst": ["Lows", "Mediums", "Worst"],
            "Best": ["Highs", "Best"],
        }

        # defining features
        features = Features(
            ordinals=chained_features,
            ordinal_values=ordinal_values,
        )

        # fitting discretizer
        discretizer = ChainedDiscretizer(
            min_freq=min_freq,
            features=features,
            chained_orders=[level0_to_level1, level1_to_level2],
        )
        _ = discretizer.fit_transform(x_train)
