"""Set of tests for discretizers module."""

from numpy import inf
from pandas import DataFrame, notna

from AutoCarver.config import DEFAULT
from AutoCarver.discretizers import Discretizer, QualitativeDiscretizer, QuantitativeDiscretizer


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

    discretizer = QuantitativeDiscretizer(quantitatives=quantitatives, min_freq=min_freq)
    x_discretized = discretizer.fit_transform(x_train, x_train[target])

    assert not discretizer.features("Discrete_Quantitative_lownan").values.contains(
        discretizer.features("Discrete_Quantitative_lownan").nan
    ), "Missing order should not be grouped with ordinal_discretizer"

    assert all(
        x_discretized["Quantitative"].value_counts(normalize=True) >= min_freq
    ), "Non-nan value was not grouped"

    print(x_train.Discrete_Quantitative_rarevalue.value_counts(dropna=False, normalize=True))

    print(discretizer.features("Discrete_Quantitative_rarevalue").values.content)
    assert discretizer.features("Discrete_Quantitative_rarevalue").values == [
        1.0,
        2.0,
        3.0,
        4.0,
        inf,
    ], (
        "Rare values should be grouped to the closest one and inf should be kept whatsoever "
        "(OrdinalDiscretizer)"
    )


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

    min_freq = 0.1
    str_default = "__default_test__"

    discretizer = QualitativeDiscretizer(
        min_freq=min_freq,
        categoricals=categoricals,
        ordinals=ordinals,
        ordinal_values=ordinal_values,
        default=str_default,
        copy=True,
        verbose=True,
    )
    x_discretized = discretizer.fit_transform(x_train, x_train[target])

    quali_expected = {
        str_default: ["Category A", "Category D", "Category F", str_default],
        "Category C": ["Category C"],
        "Category E": ["Category E"],
    }
    assert (
        discretizer.features("Qualitative").values.content == quali_expected
    ), "Values less frequent than min_freq should be grouped into default_value"

    quali_lownan_expected = {
        str_default: ["Category D", "Category F", str_default],
        "Category C": ["Category C"],
        "Category E": ["Category E"],
    }
    assert (
        discretizer.features("Qualitative_lownan").values.content == quali_lownan_expected
    ), "If any, NaN values should be put into str_nan and kept by themselves"

    expected_ordinal = {
        "Low+": ["Low-", "Low", "Low+"],
        "Medium-": ["Medium-"],
        "Medium": ["Medium"],
        "High": ["Medium+", "High-", "High"],
        "High+": ["High+"],
    }
    assert (
        discretizer.features("Qualitative_Ordinal").values.content == expected_ordinal
    ), "Values not correctly grouped"

    expected_ordinal_lownan = {
        "Low+": ["Low", "Low-", "Low+"],
        "Medium-": ["Medium-"],
        "Medium": ["Medium"],
        "High": ["Medium+", "High-", "High"],
        "High+": ["High+"],
    }
    assert (
        discretizer.features("Qualitative_Ordinal_lownan").values.content == expected_ordinal_lownan
    ), "NaNs should stay by themselves."

    feature = "Discrete_Quantitative"
    assert all(
        value in discretizer.features(feature).values for value in x_discretized[feature].unique()
    ), "discretizer not taking into account string discretizer"


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

    # minimum frequency per modality + apply(find_common_modalities) outputs a Series
    min_freq = 0.1

    # discretizing features
    discretizer = Discretizer(
        min_freq=min_freq,
        quantitatives=quantitatives,
        categoricals=categoricals,
        ordinals=ordinals,
        ordinal_values=ordinal_values,
        copy=True,
    )
    x_discretized = discretizer.fit_transform(x_train, x_train[target])
    x_dev_discretized = discretizer.transform(x_dev_1)

    assert all(
        x_discretized["Quantitative"].value_counts(normalize=True) >= min_freq
    ), "Non-nan value were not grouped"

    assert discretizer.features("Discrete_Quantitative_lownan").values == [
        1.0,
        2.0,
        3.0,
        4.0,
        inf,
    ], "NaNs should not be grouped whatsoever"

    assert discretizer.features("Discrete_Quantitative_rarevalue").values == [
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
        discretizer.features("Qualitative").values.content == quali_expected
    ), "Values less frequent than min_freq should be grouped into default_value"

    quali_lownan_expected = {
        DEFAULT: ["Category D", "Category F", DEFAULT],
        "Category C": ["Category C"],
        "Category E": ["Category E"],
    }
    assert (
        discretizer.features("Qualitative_lownan").values.content == quali_lownan_expected
    ), "If any, NaN values should be put into str_nan and kept by themselves"

    expected_ordinal = {
        "Low+": ["Low-", "Low", "Low+"],
        "Medium-": ["Medium-"],
        "Medium": ["Medium"],
        "High": ["Medium+", "High-", "High"],
        "High+": ["High+"],
    }
    assert (
        discretizer.features("Qualitative_Ordinal").values.content == expected_ordinal
    ), "Values not correctly grouped"

    expected_ordinal_lownan = {
        "Low+": ["Low", "Low-", "Low+"],
        "Medium-": ["Medium-"],
        "Medium": ["Medium"],
        "High": ["Medium+", "High-", "High"],
        "High+": ["High+"],
    }
    assert (
        discretizer.features("Qualitative_Ordinal_lownan").values.content == expected_ordinal_lownan
    ), "NaNs should stay by themselves."

    # Testing out qualitative with int/float values inside -> StringDiscretizer
    expected = {
        "2": [2.0, "2"],
        "4": [4.0, "4"],
        "1": [1.0, "1"],
        "3": [3.0, "3"],
        DEFAULT: [0.5, "0.5", 6.0, "6", 5.0, "5", DEFAULT],
    }
    assert (
        discretizer.features("Discrete_Qualitative_rarevalue_noorder").values.content == expected
    ), (
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
    assert discretizer.features("Discrete_Qualitative_noorder").values.content == expected, (
        "Qualitative features with int values should be converted to string and there values stored"
        " in the values_orders"
    )
    expected = {
        "2": ["1", 2.0, "2"],
        "3": [3.0, "3"],
        "4": [4.0, "4"],
        "5": [6.0, "6", 7.0, "7", 5.0, "5"],
    }
    assert discretizer.features("Discrete_Qualitative_highnan").values.content == expected, (
        "Ordinal qualitative features with int or float values that contain nan should be converted"
        " to string and there values stored in the values_orders"
    )

    # checking for inconsistancies in tranform
    for feature in discretizer.features:
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
