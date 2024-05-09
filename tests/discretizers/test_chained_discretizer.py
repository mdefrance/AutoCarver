"""Set of tests for qualitative_discretizers module."""

from pandas import DataFrame
from pytest import raises

from AutoCarver.discretizers import CategoricalDiscretizer, ChainedDiscretizer, OrdinalDiscretizer
from AutoCarver.features import GroupedList


def _chained_discretizer(x_train: DataFrame) -> None:
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
