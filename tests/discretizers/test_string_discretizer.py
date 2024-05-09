"""Set of tests for quantitative_discretizers module."""

from pandas import DataFrame

from AutoCarver.discretizers import StringDiscretizer
from AutoCarver.features import Features


def test_string_discretizer(x_train: DataFrame) -> None:
    """Tests StringDiscretizer

    Parameters
    ----------
    x_train : DataFrame
        Simulated Train DataFrame
    """

    categoricals = [
        "Qualitative",
        "Qualitative_grouped",
        "Qualitative_lownan",
        "Qualitative_highnan",
        "Discrete_Qualitative_noorder",
        "Discrete_Qualitative_lownan_noorder",
        "Discrete_Qualitative_rarevalue_noorder",
    ]
    ordinals = ["Qualitative_Ordinal", "Qualitative_Ordinal_lownan", "Discrete_Qualitative_highnan"]
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
    features = Features(categoricals=categoricals, ordinals=ordinals, ordinal_values=ordinal_values)

    discretizer = StringDiscretizer(features=features)
    _ = discretizer.fit_transform(x_train)

    expected = {
        "2": [2, "2"],
        "4": [4, "4"],
        "3": [3, "3"],
        "7": [7, "7"],
        "1": [1, "1"],
        "5": [5, "5"],
        "6": [6, "6"],
    }
    assert (
        discretizer.features("Discrete_Qualitative_noorder").values.content == expected
    ), "Not correctly converted for qualitative with integers"

    expected = {
        "2": [2.0, "2"],
        "4": [4.0, "4"],
        "3": [3.0, "3"],
        "1": [1.0, "1"],
        "5": [5.0, "5"],
        "6": [6.0, "6"],
        features("Discrete_Qualitative_lownan_noorder").nan: [
            features("Discrete_Qualitative_lownan_noorder").nan
        ],
    }
    assert (
        discretizer.features("Discrete_Qualitative_lownan_noorder").values.content == expected
    ), "Not correctly converted for qualitative with integers and nans"

    expected = {
        "2": [2.0, "2"],
        "4": [4.0, "4"],
        "3": [3.0, "3"],
        "0.5": [0.5, "0.5"],
        "1": [1.0, "1"],
        "5": [5.0, "5"],
        "6": [6.0, "6"],
    }
    assert (
        discretizer.features("Discrete_Qualitative_rarevalue_noorder").values.content == expected
    ), "Not correctly converted for qualitative with integers and floats"

    expected = {
        "Low-": ["Low-"],
        "Low": ["Low"],
        "Low+": ["Low+"],
        "Medium-": ["Medium-"],
        "Medium": ["Medium"],
        "Medium+": ["Medium+"],
        "High-": ["High-"],
        "High": ["High"],
        "High+": ["High+"],
        features("Qualitative_Ordinal_lownan").nan: [features("Qualitative_Ordinal_lownan").nan],
    }
    assert (
        discretizer.features("Qualitative_Ordinal_lownan").values.content == expected
    ), "No conversion for already string features"

    expected = {
        "Low-": ["Low-"],
        "Low": ["Low"],
        "Low+": ["Low+"],
        "Medium-": ["Medium-"],
        "Medium": ["Medium"],
        "Medium+": ["Medium+"],
        "High-": ["High-"],
        "High": ["High"],
        "High+": ["High+"],
    }
    assert (
        discretizer.features("Qualitative_Ordinal").values.content == expected
    ), "No conversion for not specified featues"

    expected = {
        "1": ["1"],
        "2": [2.0, "2"],
        "3": [3.0, "3"],
        "4": [4.0, "4"],
        "5": [5.0, "5"],
        "6": [6.0, "6"],
        "7": [7.0, "7"],
        features("Discrete_Qualitative_highnan").nan: [
            features("Discrete_Qualitative_highnan").nan
        ],
    }
    assert (
        discretizer.features("Discrete_Qualitative_highnan").values.content == expected
    ), "Original order should be kept for ordinal features"
