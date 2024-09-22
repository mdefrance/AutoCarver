"""Set of tests for discretizers module."""

from numpy import nan
from pandas import DataFrame, Series

from AutoCarver.features import Features, CategoricalFeature, OrdinalFeature, GroupedList
from AutoCarver.discretizers.qualitatives.qualitative_discretizer import QualitativeDiscretizer


def test_qualitative_discretizer_init():
    """Test initialization of QualitativeDiscretizer"""
    feature1 = CategoricalFeature("feature1")
    feature2 = OrdinalFeature("feature2", ["A", "B", "C"])
    features = [feature1, feature2]
    discretizer = QualitativeDiscretizer(qualitatives=features, min_freq=0.05)
    assert isinstance(discretizer.features, Features)
    assert "feature1" in discretizer.features
    assert "feature2" in discretizer.features
    assert discretizer.min_freq == 0.05


def test_qualitative_discretizer_prepare_data():
    """Test _prepare_X method of QualitativeDiscretizer"""
    features = [CategoricalFeature("feature1"), OrdinalFeature("feature2", ["A", "B", "C"])]
    df = DataFrame(
        {"feature1": [1, 2, 3], "feature2": ["A", "B", "C"], "feature3": [1.0, 2.1, 3.2]}
    )
    discretizer = QualitativeDiscretizer(qualitatives=features, min_freq=0.05)
    prepared_df = discretizer._prepare_data(df)
    assert prepared_df["feature1"].dtype == object
    assert prepared_df["feature1"].tolist() == ["1", "2", "3"]
    assert prepared_df["feature2"].dtype == object
    assert prepared_df["feature2"].tolist() == ["A", "B", "C"]
    assert prepared_df["feature3"].tolist() == [1.0, 2.1, 3.2]


def test_qualitative_discretizer_fit_categorical_features():
    """Test fit method of QualitativeDiscretizer with basic input"""

    # test with binary target
    feature1 = CategoricalFeature("feature1")
    feature2 = CategoricalFeature("feature2")
    feature3 = CategoricalFeature("feature3")
    discretizer = QualitativeDiscretizer([feature1, feature2, feature3], min_freq=0.2)

    X = DataFrame(
        {
            "feature1": ["x", "b", "a", "b", "c", "c", "c", "c", nan],
            "feature2": ["a", "y", "x", "y", "z", "z", "z", "z", nan],
            "feature3": [0, 2, 1, 2, 3, 3.0, "3", 3, nan],
        }
    )
    y = Series([0, 0, 1, 0, 1, 1, 1, 1, 1])

    discretizer.fit(X, y)

    assert feature1.has_default
    assert feature2.has_default
    assert feature3.has_default
    assert feature1.values == ["b", feature1.default, "c"]
    assert feature2.values == ["y", feature2.default, "z"]
    assert feature3.values == ["2", feature3.default, "3"]

    transformed_x = discretizer.transform(X)

    assert transformed_x["feature1"].tolist() == [
        feature1.label_per_value[feature1.default],
        "b",
        feature1.label_per_value[feature1.default],
        "b",
        "c",
        "c",
        "c",
        "c",
        nan,
    ]
    assert transformed_x["feature2"].tolist() == [
        feature2.label_per_value[feature2.default],
        "y",
        feature2.label_per_value[feature2.default],
        "y",
        "z",
        "z",
        "z",
        "z",
        nan,
    ]
    assert transformed_x["feature3"].tolist() == [
        feature3.label_per_value[feature3.default],
        "2",
        feature3.label_per_value[feature3.default],
        "2",
        "3",
        "3",
        "3",
        "3",
        nan,
    ]


def test_qualitative_discretizer_fit_ordinal_features():
    """Test fit method of QualitativeDiscretizer with basic input"""

    feature1 = OrdinalFeature("feature1", ["a", "b", "c", "x"])
    feature2 = OrdinalFeature("feature2", ["a", "x", "y", "z"])
    feature3 = OrdinalFeature("feature3", ["0", "1", "2", "3"])
    discretizer = QualitativeDiscretizer([feature1, feature2, feature3], min_freq=0.2)

    X = DataFrame(
        {
            "feature1": ["x", "b", "a", "b", "c", "c", "c", "c", nan],
            "feature2": ["a", "y", "x", "y", "z", "z", "z", "z", nan],
            "feature3": [0, 2, 1, 2, 3, 3.0, "3", 3, nan],
        }
    )
    y = Series([0, 0, 1, 0, 1, 1, 1, 1, 1])

    discretizer.fit(X, y)

    # Check feature1
    expected_feature1 = GroupedList(["a", "b", "c", "x"])
    expected_feature1.group("x", "c")
    expected_feature1.group("a", "b")
    assert feature1.values == expected_feature1
    assert feature1.content == expected_feature1.content

    # Check feature2
    expected_feature2 = GroupedList(["a", "x", "y", "z"])
    expected_feature2.group("a", "x")
    assert feature2.values == expected_feature2
    assert feature2.content == expected_feature2.content

    # Check feature3
    expected_feature3 = GroupedList(["0", "1", "2", "3"])
    expected_feature3.group("0", "1")
    print("featiure3", feature3.content)
    assert feature3.values == expected_feature3
    assert feature3.content == {"1": [0, "0", 1, "1"], "2": [2, "2"], "3": [3, "3"]}

    # Check transformed data
    transformed = discretizer.transform(X)
    df_expected = DataFrame(
        {
            "feature1": [
                "c to x",
                "a to b",
                "a to b",
                "a to b",
                "c to x",
                "c to x",
                "c to x",
                "c to x",
                nan,
            ],
            "feature2": ["a to x", "y", "a to x", "y", "z", "z", "z", "z", nan],
            "feature3": ["0 to 1", "2", "0 to 1", "2", "3", "3", "3", "3", nan],
        }
    )
    assert transformed.equals(df_expected), "Transformed data does not match expected data"


def test_qualitative_discretizer(x_train: DataFrame, target: str):
    """Tests QualitativeDiscretizer

    Parameters
    ----------
    x_train : DataFrame
        Simulated Train DataFrame
    target: str
        Target feature
    """

    categoricals = [
        "Qualitative",
        "Qualitative_grouped",
        "Qualitative_lownan",
        "Qualitative_highnan",
        "Discrete_Quantitative",
    ]
    ordinals = ["Qualitative_Ordinal", "Qualitative_Ordinal_lownan"]
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
    }

    # defining features
    str_default = "__default_test__"
    features = Features(
        categoricals=categoricals,
        ordinals=ordinals,
        ordinal_values=ordinal_values,
        default=str_default,
    )

    min_freq = 0.1

    discretizer = QualitativeDiscretizer(
        min_freq=min_freq, qualitatives=features.qualitatives, copy=True, verbose=True
    )
    x_discretized = discretizer.fit_transform(x_train, x_train[target])

    quali_expected = {
        str_default: ["Category A", "Category D", "Category F", str_default],
        "Category C": ["Category C"],
        "Category E": ["Category E"],
    }
    assert (
        features("Qualitative").content == quali_expected
    ), "Values less frequent than min_freq should be grouped into default_value"

    quali_lownan_expected = {
        str_default: ["Category D", "Category F", str_default],
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

    feature = "Discrete_Quantitative"
    print(features(feature).labels, x_discretized[feature].unique())
    assert all(
        label in features(feature).labels for label in x_discretized[feature].unique()
    ), "discretizer not taking into account string discretizer"
