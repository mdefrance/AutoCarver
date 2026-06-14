"""Set of tests for discretizers module."""

import numpy as np
import pandas as pd

from AutoCarver.discretizers.qualitatives.qualitative_discretizer import QualitativeDiscretizer
from AutoCarver.discretizers.utils.base_discretizer import ProcessingConfig, Sample
from AutoCarver.features import CategoricalFeature, Features, FeaturesConfig, GroupedList, OrdinalFeature


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


def test_qualitative_discretizer_prepare_sample():
    """Test _prepare_X method of QualitativeDiscretizer"""
    features = [CategoricalFeature("feature1"), OrdinalFeature("feature2", ["A", "B", "C"])]
    df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": ["A", "B", "C"], "feature3": [1.0, 2.1, 3.2]})
    discretizer = QualitativeDiscretizer(qualitatives=features, min_freq=0.05)
    sample = discretizer._prepare_sample(Sample(df))
    prepared_df = sample.X
    assert prepared_df["feature1"].dtype == object
    assert prepared_df["feature1"].tolist() == ["1", "2", "3"]
    assert prepared_df["feature2"].dtype == object
    assert prepared_df["feature2"].tolist() == ["A", "B", "C"]
    assert prepared_df["feature3"].tolist() == [1.0, 2.1, 3.2]


def test_qualitative_discretizer_fit_categorical_features():
    """QualitativeDiscretizer routes categorical features through CategoricalDiscretizer.

    Scaled to n≈1000 so the Wilson CI is tight enough to flag the 5%-frequent
    "x"/"a" modalities as significantly below min_freq=0.10.
    """

    feature1 = CategoricalFeature("feature1")
    feature2 = CategoricalFeature("feature2")
    feature3 = CategoricalFeature("feature3")
    discretizer = QualitativeDiscretizer([feature1, feature2, feature3], min_freq=0.10)

    n = 999
    f_values = ["x"] * 50 + ["a"] * 50 + ["b"] * 300 + ["c"] * (n - 400) + [np.nan]
    f3_values = ["0"] * 50 + ["1"] * 50 + ["2"] * 300 + ["3"] * (n - 400) + [np.nan]
    X = pd.DataFrame({"feature1": f_values, "feature2": f_values, "feature3": f3_values})
    y = pd.Series([1] * 50 + [1] * 50 + [0] * 300 + [1] * (n - 400) + [1])

    discretizer.fit(X, y)

    assert feature1.has_default
    assert feature2.has_default
    assert feature3.has_default
    # near-threshold ("x"/"a" or "0"/"1") merged into default; others survive
    assert set(feature1.values) == {"c", "b", feature1.default}
    assert set(feature2.values) == {"c", "b", feature2.default}
    assert set(feature3.values) == {"3", "2", feature3.default}


def test_qualitative_discretizer_fit_ordinal_features():
    """QualitativeDiscretizer dispatches ordinal features to OrdinalDiscretizer.

    On this n=9 sample with min_freq=3/9 the Wilson CI is too wide to flag the
    singleton modalities as significantly below — they survive untouched. The
    test still verifies the dispatch wiring (no exception, labels preserved).
    """

    feature1 = OrdinalFeature("feature1", ["a", "b", "c", "x"])
    feature2 = OrdinalFeature("feature2", ["a", "x", "y", "z"])
    feature3 = OrdinalFeature("feature3", ["0", "1", "2", "3"])
    discretizer = QualitativeDiscretizer([feature1, feature2, feature3], min_freq=3 / 9)

    X = pd.DataFrame(
        {
            "feature1": ["x", "b", "a", "b", "c", "c", "c", "c", np.nan],
            "feature2": ["a", "y", "x", "y", "z", "z", "z", "z", np.nan],
            "feature3": [0, 2, 1, 2, 3, 3.0, "3", 3, np.nan],
        }
    )
    y = pd.Series([0, 0, 1, 0, 1, 1, 1, 1, 1])

    discretizer.fit(X, y)

    assert feature1.values == GroupedList(["a", "b", "c", "x"])
    assert feature2.values == GroupedList(["a", "x", "y", "z"])
    assert feature3.values == GroupedList(["0", "1", "2", "3"])
    assert feature3.content == {"0": [0, "0"], "1": [1, "1"], "2": [2, "2"], "3": [3, "3"]}


def test_qualitative_discretizer(x_train: pd.DataFrame, target: str):
    """Tests QualitativeDiscretizer

    Parameters
    ----------
    x_train : pd.DataFrame
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
    features = Features(categoricals=categoricals, ordinals=ordinal_values, config=FeaturesConfig(default=str_default))

    min_freq = 0.1

    discretizer = QualitativeDiscretizer(
        min_freq=min_freq,
        qualitatives=features.qualitatives,
        config=ProcessingConfig(copy=True, verbose=True),
    )
    x_discretized = discretizer.fit_transform(x_train, x_train[target])

    quali_expected = {
        str_default: ["Category A", "Category D", "Category F", str_default],
        "Category C": ["Category C"],
        "Category E": ["Category E"],
    }
    assert features("Qualitative").content == quali_expected, (
        "Values less frequent than min_freq should be grouped into default_value"
    )

    quali_lownan_expected = {
        str_default: ["Category D", "Category F", str_default],
        "Category C": ["Category C"],
        "Category E": ["Category E"],
    }
    assert features("Qualitative_lownan").content == quali_lownan_expected, (
        "If any, np.nan values should be put into str_nan and kept by themselves"
    )

    expected_ordinal = {
        "Low+": ["Low-", "Low", "Low+"],
        "Medium-": ["Medium-"],
        "Medium": ["Medium"],
        "High": ["Medium+", "High-", "High"],
        "High+": ["High+"],
    }
    assert features("Qualitative_Ordinal").content == expected_ordinal, "Values not correctly grouped"

    expected_ordinal_lownan = {
        "Low+": ["Low", "Low-", "Low+"],
        "Medium-": ["Medium-"],
        "Medium": ["Medium"],
        "High": ["Medium+", "High-", "High"],
        "High+": ["High+"],
    }
    assert features("Qualitative_Ordinal_lownan").content == expected_ordinal_lownan, "NaNs should stay by themselves."

    feature = "Discrete_Quantitative"
    print(features(feature).labels, x_discretized[feature].unique())
    assert all(label in features(feature).labels for label in x_discretized[feature].unique()), (
        "discretizer not taking into account string discretizer"
    )
