"""Set of tests for qualitative_discretizers module."""

from pandas import DataFrame
from pytest import raises

from AutoCarver.config import DEFAULT, NAN
from AutoCarver.discretizers import (
    CategoricalDiscretizer,
    ChainedDiscretizer,
    GroupedList,
    OrdinalDiscretizer,
)


def test_chained_discretizer(x_train: DataFrame) -> None:
    """Tests CategoricalDiscretizer

    Parameters
    ----------
    x_train : DataFrame
        Simulated Train DataFrame
    """
    # Simulating some datasets with unknown values
    x_train_wrong_1 = x_train.copy()
    x_train_wrong_2 = x_train.copy()

    x_train_wrong_1["Qualitative_Ordinal"] = x_train["Qualitative_Ordinal"].replace(
        "Medium", "unknown"
    )
    x_train_wrong_2["Qualitative_Ordinal_lownan"] = x_train["Qualitative_Ordinal_lownan"].replace(
        "Medium", "unknown"
    )

    chained_features = ["Qualitative_Ordinal", "Qualitative_Ordinal_lownan"]
    values_orders = {
        "Qualitative_Ordinal_lownan": [
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

    min_freq = 0.15

    discretizer = ChainedDiscretizer(
        qualitative_features=chained_features,
        chained_orders=[level0_to_level1, level1_to_level2],
        min_freq=min_freq,
        values_orders=values_orders,
        unknown_handling="raise",
        copy=True,
    )
    _ = discretizer.fit_transform(x_train)

    expected = {
        "High+": ["High+"],
        "Best": ["High", "High-", "Highs", "Best"],
        "Mediums": ["Medium+", "Medium-", "Mediums"],
        "Medium": ["Medium"],
        "Worst": ["Low+", "Low", "Low-", "Lows", "Worst"],
    }
    assert (
        discretizer.values_orders["Qualitative_Ordinal"].content == expected
    ), "Values less frequent than min_freq should be grouped"
    assert discretizer.values_orders["Qualitative_Ordinal"] == [
        "Medium",
        "Mediums",
        "Worst",
        "High+",
        "Best",
    ], "Order of ordinal features is wrong"

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
        NAN: [NAN],
    }
    assert (
        discretizer.values_orders["Qualitative_Ordinal_highnan"].content == expected
    ), "Not specified features should not be modified, expect for there NaNS"

    expected = {
        "Medium": ["Medium"],
        "High+": ["High+"],
        "Best": ["High", "High-", "Highs", "Best"],
        "Mediums": ["Medium+", "Medium-", "Mediums"],
        "Worst": ["Low+", "Low", "Low-", "Lows", "Worst"],
        NAN: [NAN],
    }
    assert discretizer.values_orders["Qualitative_Ordinal_lownan"].content == expected, (
        "NaNs should be added to the order and missing values from the values_orders should"
        " be added (from chained_orders)"
    )
    expected = [
        "Medium",
        "Mediums",
        "Worst",
        "High+",
        "Best",
        NAN,
    ]
    assert (
        discretizer.values_orders["Qualitative_Ordinal_lownan"] == expected
    ), "Order of ordinal features is wrong"

    # testing to fit when unknwon_handling = 'raise'
    with raises(AssertionError):
        discretizer.fit_transform(x_train_wrong_1)
    with raises(AssertionError):
        discretizer.fit_transform(x_train_wrong_2)

    # testing discretization when unknwon_handling = 'drop'
    min_freq = 0.15
    discretizer = ChainedDiscretizer(
        qualitative_features=chained_features,
        chained_orders=[level0_to_level1, level1_to_level2],
        min_freq=min_freq,
        values_orders=values_orders,
        unknown_handling="drop",
        copy=True,
    )

    # Case 1: unknown value but no NaN
    _ = discretizer.fit_transform(x_train_wrong_1)

    expected = {
        "Mediums": ["Medium+", "Medium", "Medium-", "Mediums"],
        "Worst": ["Low+", "Low", "Low-", "Lows", "Worst"],
        "High+": ["High+"],
        "Best": ["High", "High-", "Highs", "Best"],
        NAN: ["unknown", NAN],
    }
    assert (
        discretizer.values_orders["Qualitative_Ordinal"].content == expected
    ), "Values less frequent than min_freq should be grouped"
    expected = [
        "Mediums",
        "Worst",
        "High+",
        "Best",
        NAN,
    ]
    assert (
        discretizer.values_orders["Qualitative_Ordinal"] == expected
    ), "Order of ordinal features is wrong"

    expected = {
        "Medium": ["Medium"],
        "Mediums": ["Medium+", "Medium-", "Mediums"],
        "Worst": ["Low+", "Low", "Low-", "Lows", "Worst"],
        "High+": ["High+"],
        "Best": ["High", "High-", "Highs", "Best"],
        NAN: [NAN],
    }
    assert discretizer.values_orders["Qualitative_Ordinal_lownan"].content == expected, (
        "NaNs should be added to the order and missing values from the values_orders should be"
        " added (from chained_orders)"
    )
    expected = [
        "Medium",
        "Mediums",
        "Worst",
        "High+",
        "Best",
        NAN,
    ]
    assert (
        discretizer.values_orders["Qualitative_Ordinal_lownan"] == expected
    ), "Order of ordinal features is wrong"

    # Case 2: unknown value but with NaN
    min_freq = 0.15
    discretizer = ChainedDiscretizer(
        qualitative_features=chained_features,
        chained_orders=[level0_to_level1, level1_to_level2],
        min_freq=min_freq,
        values_orders=values_orders,
        unknown_handling="drop",
        copy=True,
    )
    _ = discretizer.fit_transform(x_train_wrong_2)

    expected = {
        "Medium": ["Medium"],
        "Mediums": ["Medium+", "Medium-", "Mediums"],
        "Worst": ["Low+", "Low", "Low-", "Lows", "Worst"],
        "High+": ["High+"],
        "Best": ["High", "High-", "Highs", "Best"],
    }
    assert (
        discretizer.values_orders["Qualitative_Ordinal"].content == expected
    ), "Values less frequent than min_freq should be grouped"
    expected = [
        "Medium",
        "Mediums",
        "Worst",
        "High+",
        "Best",
    ]
    assert (
        discretizer.values_orders["Qualitative_Ordinal"] == expected
    ), "Order of ordinal features is wrong"

    expected = {
        "Mediums": ["Medium+", "Medium", "Medium-", "Mediums"],
        "Worst": ["Low+", "Low", "Low-", "Lows", "Worst"],
        "High+": ["High+"],
        "Best": ["High", "High-", "Highs", "Best"],
        NAN: ["unknown", NAN],
    }
    assert discretizer.values_orders["Qualitative_Ordinal_lownan"].content == expected, (
        "NaNs should be added to the order and missing values from the values_orders should be "
        "added (from chained_orders)"
    )
    expected = [
        "Mediums",
        "Worst",
        "High+",
        "Best",
        NAN,
    ]
    assert (
        discretizer.values_orders["Qualitative_Ordinal_lownan"] == expected
    ), "Order of ordinal features is wrong"

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

    min_freq = 0.15

    discretizer = ChainedDiscretizer(
        qualitative_features=chained_features,
        chained_orders=[level0_to_level1, level1_to_level2],
        min_freq=min_freq,
        values_orders=values_orders,
        copy=True,
    )
    _ = discretizer.fit_transform(x_train)

    expected = {
        "Medium": ["Medium"],
        "Mediums": ["Medium+", "Medium-", "Mediums"],
        "Worst": ["Low+", "Low", "Low-", "Lows", "Worst"],
        "High+": ["High+"],
        "Best": ["High", "High-", "Highs", "Best"],
        "BEST": ["ALL_ALONE", "ALONE", "BEST"],
        NAN: [NAN],
    }

    assert (
        discretizer.values_orders["Qualitative_Ordinal_lownan"].content == expected
    ), "All provided values should be kept."

    expected = {
        "Medium": ["Medium"],
        "Mediums": ["Medium+", "Medium-", "Mediums"],
        "Worst": ["Low+", "Low", "Low-", "Lows", "Worst"],
        "High+": ["High+"],
        "Best": ["High", "High-", "Highs", "Best"],
        "BEST": ["ALL_ALONE", "ALONE", "BEST"],
    }

    assert (
        discretizer.values_orders["Qualitative_Ordinal"].content == expected
    ), "All provided values should be kept."

    # testing for defintion of levels
    with raises(AssertionError):
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

        min_freq = 0.15

        discretizer = ChainedDiscretizer(
            qualitative_features=chained_features,
            chained_orders=[level0_to_level1, level1_to_level2],
            min_freq=min_freq,
            values_orders=values_orders,
            unknown_handling="drop",
            copy=True,
        )
        _ = discretizer.fit_transform(x_train)

    # testing for defintion of levels
    with raises(AssertionError):
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

        min_freq = 0.15

        discretizer = ChainedDiscretizer(
            qualitative_features=chained_features,
            chained_orders=[level0_to_level1, level1_to_level2],
            min_freq=min_freq,
            values_orders=values_orders,
            unknown_handling="drop",
            copy=True,
        )
        _ = discretizer.fit_transform(x_train)

    # testing that it does not work when there is a val in values_orders missing from chained_orders
    with raises(AssertionError):
        values_orders = {
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

        min_freq = 0.15

        discretizer = ChainedDiscretizer(
            qualitative_features=chained_features,
            chained_orders=[level0_to_level1, level1_to_level2],
            min_freq=min_freq,
            values_orders=values_orders,
            copy=True,
        )
        _ = discretizer.fit_transform(x_train)


def test_default_discretizer(x_train: DataFrame, target: str) -> None:
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


def test_ordinal_discretizer(x_train: DataFrame, target: str) -> None:
    """Tests OrdinalDiscretizer

    # TODO: add tests for quantitative features

    Parameters
    ----------
    x_train : DataFrame
        Simulated Train DataFrame
    target: str
        Target feature
    """
    # defining values_orders
    order = ["Low-", "Low", "Low+", "Medium-", "Medium", "Medium+", "High-", "High", "High+"]
    # ordering for base qualitative ordinal feature
    groupedlist = GroupedList(order)

    # ordering for qualitative ordinal feature that contains NaNs
    groupedlist_lownan = GroupedList(order)

    # storing per feature orders
    features = ["Qualitative_Ordinal", "Qualitative_Ordinal_lownan"]
    values_orders = {
        "Qualitative_Ordinal": groupedlist,
        "Qualitative_Ordinal_lownan": groupedlist_lownan,
    }

    # minimum frequency per modality + apply(find_common_modalities) outputs a Series
    min_freq = 0.01

    # discretizing features
    discretizer = OrdinalDiscretizer(
        ordinal_features=features, min_freq=min_freq, values_orders=values_orders, copy=True
    )
    discretizer.fit_transform(x_train, x_train[target])

    expected_ordinal_01 = {
        "Low-": ["Low", "Low-"],
        "Low+": ["Low+"],
        "Medium-": ["Medium-"],
        "Medium": ["Medium"],
        "Medium+": ["High-", "Medium+"],
        "High": ["High"],
        "High+": ["High+"],
    }
    expected_ordinal_lownan_01 = {
        "Low+": ["Low", "Low-", "Low+"],
        "Medium-": ["Medium-"],
        "Medium": ["Medium"],
        "Medium+": ["High-", "Medium+"],
        "High": ["High"],
        "High+": ["High+"],
        NAN: [NAN],
    }
    assert (
        discretizer.values_orders["Qualitative_Ordinal"].content == expected_ordinal_01
    ), "Missing value in order not correctly grouped"
    assert (
        discretizer.values_orders["Qualitative_Ordinal_lownan"].content
        == expected_ordinal_lownan_01
    ), "Missing value in order not correctly grouped or introduced nans."

    # minimum frequency per modality + apply(find_common_modalities) outputs a DataFrame
    min_freq = 0.08

    # discretizing features
    discretizer = OrdinalDiscretizer(
        ordinal_features=features, min_freq=min_freq, values_orders=values_orders, copy=True
    )
    discretizer.fit_transform(x_train, x_train[target])

    expected_ordinal_08 = {
        "Low+": ["Low-", "Low", "Low+"],
        "Medium-": ["Medium-"],
        "Medium": ["Medium"],
        "High": ["Medium+", "High-", "High"],
        "High+": ["High+"],
    }
    expected_ordinal_lownan_08 = {
        "Low+": ["Low", "Low-", "Low+"],
        "Medium-": ["Medium-"],
        "Medium": ["Medium"],
        "High": ["Medium+", "High-", "High"],
        "High+": ["High+"],
        NAN: [NAN],
    }
    assert (
        discretizer.values_orders["Qualitative_Ordinal"].content == expected_ordinal_08
    ), "Values not correctly grouped"
    assert (
        discretizer.values_orders["Qualitative_Ordinal_lownan"].content
        == expected_ordinal_lownan_08
    ), "NaNs should stay by themselves."
