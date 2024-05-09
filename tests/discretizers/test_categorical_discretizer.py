"""Set of tests for qualitative_discretizers module."""

from pandas import DataFrame
from pytest import raises

from AutoCarver.discretizers import CategoricalDiscretizer
from AutoCarver.features import GroupedList


def test_categorical_discretizer(x_train: DataFrame, target: str) -> None:
    """Tests CategoricalDiscretizer

    Parameters
    ----------
    x_train : DataFrame
        Simulated Train DataFrame
    target: str
        Target feature
    """

    # defining values_orders
    order = ["Category A", "Category B", "Category C", "Category D", "Category E", "Category F"]
    # ordering for base qualitative ordinal feature
    groupedlist = GroupedList(order)
    groupedlist_lownan = GroupedList(["Category C", "Category D", "Category E", "Category F"])
    groupedlist_highnan = GroupedList(["Category A", "Category C", "Category D", "Category E"])

    # ordering for qualitative ordinal feature that contains NaNs
    groupedlist_grouped = GroupedList(order)
    groupedlist_grouped.group("Category A", "Category D")

    # ordering for base qualitative ordinal feature
    order = ["Low-", "Low", "Low+", "Medium-", "Medium", "Medium+", "High-", "High", "High+"]
    groupedlist_ordinal = GroupedList(order)
    groupedlist_ordinal.group_list(["Low-", "Low"], "Low+")
    groupedlist_ordinal.group_list(["Medium+", "High-"], "High")

    # storing per feature orders
    values_orders = {
        "Qualitative_grouped": groupedlist_grouped,
        "Qualitative_highnan": groupedlist,
        "Qualitative_lownan": groupedlist,
        "Qualitative_Ordinal": groupedlist_ordinal,
    }

    features = ["Qualitative", "Qualitative_grouped", "Qualitative_lownan", "Qualitative_highnan"]

    min_freq = 0.02
    # unwanted value in values_orders
    with raises(AssertionError):
        discretizer = CategoricalDiscretizer(
            qualitative_features=features, min_freq=min_freq, values_orders=values_orders, copy=True
        )
        _ = discretizer.fit_transform(x_train, x_train[target])

    # correct feature ordering
    groupedlist_grouped.group("Category B", "Category D")
    values_orders = {
        "Qualitative_grouped": groupedlist_grouped,
        "Qualitative_highnan": groupedlist_highnan,
        "Qualitative_lownan": groupedlist_lownan,
        "Qualitative_Ordinal": groupedlist_ordinal,
    }

    discretizer = CategoricalDiscretizer(
        qualitative_features=features, min_freq=min_freq, values_orders=values_orders, copy=True
    )
    _ = discretizer.fit_transform(x_train, x_train[target])

    assert (
        discretizer.values_orders["Qualitative_Ordinal"].content == groupedlist_ordinal.content
    ), "Column names of values_orders not provided if features should not be discretized."

    quali_expected_order = {
        "binary_target": ["Category D", DEFAULT, "Category F", "Category C", "Category E"],
        "continuous_target": [DEFAULT, "Category C", "Category E", "Category F", "Category D"],
    }
    assert (
        discretizer.values_orders["Qualitative"] == quali_expected_order[target]
    ), "Incorrect ordering by target rate"

    quali_expected = {
        DEFAULT: ["Category A", DEFAULT],
        "Category C": ["Category C"],
        "Category F": ["Category F"],
        "Category E": ["Category E"],
        "Category D": ["Category D"],
    }
    assert (
        discretizer.values_orders["Qualitative"].content == quali_expected
    ), "Values less frequent than min_freq should be grouped into default_value"

    quali_lownan_expected_order = {
        "binary_target": [
            "Category D",
            "Category F",
            "Category C",
            "Category E",
            NAN,
        ],
        "continuous_target": ["Category C", "Category E", "Category F", "Category D", NAN],
    }
    assert (
        discretizer.values_orders["Qualitative_lownan"] == quali_lownan_expected_order[target]
    ), "Incorrect ordering by target rate"

    quali_lownan_expected = {
        NAN: [NAN],
        "Category C": ["Category C"],
        "Category F": ["Category F"],
        "Category E": ["Category E"],
        "Category D": ["Category D"],
    }
    assert (
        discretizer.values_orders["Qualitative_lownan"].content == quali_lownan_expected
    ), "If any, NaN values should be put into str_nan and kept by themselves"

    quali_highnan_expected_order = {
        "binary_target": [
            "Category D",
            DEFAULT,
            "Category C",
            "Category E",
            NAN,
        ],
        "continuous_target": [DEFAULT, "Category C", "Category E", "Category D", NAN],
    }
    assert (
        discretizer.values_orders["Qualitative_highnan"] == quali_highnan_expected_order[target]
    ), "Incorrect ordering by target rate"

    quali_highnan_expected = {
        DEFAULT: ["Category A", DEFAULT],
        "Category C": ["Category C"],
        NAN: [NAN],
        "Category E": ["Category E"],
        "Category D": ["Category D"],
    }
    assert (
        discretizer.values_orders["Qualitative_highnan"].content == quali_highnan_expected
    ), "If any, NaN values should be put into str_nan and kept by themselves"

    assert (
        discretizer.values_orders["Qualitative_grouped"].content == groupedlist_grouped.content
    ), "Grouped values should keep there group"
