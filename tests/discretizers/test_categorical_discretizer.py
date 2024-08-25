"""Set of tests for qualitative_discretizers module."""

from pandas import DataFrame
from pytest import raises

from AutoCarver.discretizers import CategoricalDiscretizer
from AutoCarver.features import Features, GroupedList


def test_categorical_discretizer(x_train: DataFrame, target: str) -> None:
    """Tests CategoricalDiscretizer

    Parameters
    ----------
    x_train : DataFrame
        Simulated Train DataFrame
    target: str
        Target feature
    """

    # setting new str_default
    str_default = "_DEFAULT_"

    # defining values_orders
    order = ["Category A", "Category B", "Category C", "Category D", "Category E", "Category F"]
    # ordering for base qualitative ordinal feature
    groupedlist = GroupedList(order)

    # ordering for qualitative ordinal feature that contains NaNs
    groupedlist_grouped = GroupedList(order)
    groupedlist_grouped.group("Category A", "Category D")

    # ordering for base qualitative ordinal feature
    order = ["Low-", "Low", "Low+", "Medium-", "Medium", "Medium+", "High-", "High", "High+"]
    groupedlist_ordinal = GroupedList(order)
    groupedlist_ordinal.group(["Low-", "Low"], "Low+")
    groupedlist_ordinal.group(["Medium+", "High-"], "High")

    # storing per feature orders
    values_orders = {
        "Qualitative_grouped": GroupedList(groupedlist_grouped),
        "Qualitative_highnan": GroupedList(groupedlist),
        "Qualitative_lownan": GroupedList(groupedlist),
        "Qualitative_Ordinal": GroupedList(groupedlist_ordinal),
    }

    categoricals = [
        "Qualitative_grouped",
        "Qualitative_lownan",
        "Qualitative_highnan",
        "Qualitative_Ordinal",
    ]
    features = Features(categoricals=categoricals, default=str_default)
    features.update(values_orders, replace=True)

    min_freq = 0.02
    # unwanted value in values_orders
    with raises(ValueError):
        discretizer = CategoricalDiscretizer(categoricals=features, min_freq=min_freq, copy=True)
        _ = discretizer.fit_transform(x_train, x_train[target])

    # correct feature ordering
    features = Features(categoricals=categoricals + ["Qualitative"], default=str_default)

    discretizer = CategoricalDiscretizer(categoricals=features, min_freq=min_freq, copy=True)
    _ = discretizer.fit_transform(x_train, x_train[target])

    assert features("Qualitative_Ordinal").values.get(str_default) == [
        "Low-",
        str_default,
    ], "Non frequent modalities should be grouped with default value."

    quali_expected_order = {
        "binary_target": ["Category D", str_default, "Category F", "Category C", "Category E"],
        "continuous_target": [str_default, "Category C", "Category E", "Category F", "Category D"],
    }
    assert (
        features("Qualitative").values == quali_expected_order[target]
    ), "Incorrect ordering by target rate"

    quali_expected = {
        str_default: ["Category A", str_default],
        "Category C": ["Category C"],
        "Category F": ["Category F"],
        "Category E": ["Category E"],
        "Category D": ["Category D"],
    }
    assert (
        features("Qualitative").content == quali_expected
    ), "Values less frequent than min_freq should be grouped into default_value"

    quali_lownan_expected_order = {
        "binary_target": [
            "Category D",
            "Category F",
            "Category C",
            "Category E",
        ],
        "continuous_target": [
            "Category C",
            "Category E",
            "Category F",
            "Category D",
        ],
    }
    assert (
        features("Qualitative_lownan").values == quali_lownan_expected_order[target]
    ), "Incorrect ordering by target rate"

    quali_lownan_expected = {
        "Category C": ["Category C"],
        "Category F": ["Category F"],
        "Category E": ["Category E"],
        "Category D": ["Category D"],
    }
    assert (
        features("Qualitative_lownan").content == quali_lownan_expected
    ), "If any, NaN values should be put into str_nan and kept by themselves"

    quali_highnan_expected_order = {
        "binary_target": [
            "Category D",
            str_default,
            "Category C",
            "Category E",
        ],
        "continuous_target": [
            str_default,
            "Category C",
            "Category E",
            "Category D",
        ],
    }
    assert (
        features("Qualitative_highnan").values == quali_highnan_expected_order[target]
    ), "Incorrect ordering by target rate"

    quali_highnan_expected = {
        str_default: ["Category A", str_default],
        "Category C": ["Category C"],
        "Category E": ["Category E"],
        "Category D": ["Category D"],
    }
    assert (
        features("Qualitative_highnan").content == quali_highnan_expected
    ), "If any, NaN values should be put into str_nan and kept by themselves"
