"""Set of tests for discretizers module."""

from numpy import inf
from pandas import DataFrame, notna

from AutoCarver import Features
from AutoCarver.config import DEFAULT
from AutoCarver.discretizers import QualitativeDiscretizer


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
