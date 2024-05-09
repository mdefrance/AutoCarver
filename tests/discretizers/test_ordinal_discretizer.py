"""Set of tests for qualitative_discretizers module."""

from pandas import DataFrame

from AutoCarver.discretizers import OrdinalDiscretizer
from AutoCarver.features import GroupedList


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
