"""Set of tests for qualitative_discretizers module."""

from pandas import DataFrame

from AutoCarver.discretizers import OrdinalDiscretizer
from AutoCarver.features import GroupedList, Features


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
    ordinals = ["Qualitative_Ordinal", "Qualitative_Ordinal_lownan"]
    ordinal_values = {
        "Qualitative_Ordinal": groupedlist,
        "Qualitative_Ordinal_lownan": groupedlist_lownan,
    }
    features = Features(ordinals=ordinals, ordinal_values=ordinal_values)

    # minimum frequency per modality + apply(find_common_modalities) outputs a Series
    min_freq = 0.01

    # discretizing features
    discretizer = OrdinalDiscretizer(ordinals=features, min_freq=min_freq, copy=True)
    x_disc = discretizer.fit_transform(x_train, x_train[target])

    feature = "Qualitative_Ordinal"
    expected_ordinal_01 = {
        "Low-": ["Low", "Low-"],
        "Low+": ["Low+"],
        "Medium-": ["Medium-"],
        "Medium": ["Medium"],
        "Medium+": ["High-", "Medium+"],
        "High": ["High"],
        "High+": ["High+"],
    }
    print(
        discretizer.features(feature).get_content(),
        features(feature).get_content(),
        x_disc[feature].value_counts(dropna=False, normalize=True).round(2),
    )
    assert (
        features(feature).get_content() == expected_ordinal_01
    ), "Missing value in order not correctly grouped"

    expected_ordinal_lownan_01 = {
        "Low+": ["Low", "Low-", "Low+"],
        "Medium-": ["Medium-"],
        "Medium": ["Medium"],
        "Medium+": ["High-", "Medium+"],
        "High": ["High"],
        "High+": ["High+"],
    }
    assert (
        features("Qualitative_Ordinal_lownan").get_content() == expected_ordinal_lownan_01
    ), "Missing value in order not correctly grouped or introduced nans."

    # minimum frequency per modality + apply(find_common_modalities) outputs a DataFrame
    min_freq = 0.08

    # discretizing features
    features = Features(ordinals=ordinals, ordinal_values=ordinal_values)
    discretizer = OrdinalDiscretizer(ordinals=features, min_freq=min_freq, copy=True)
    discretizer.fit_transform(x_train, x_train[target])

    expected_ordinal_08 = {
        "Low+": ["Low-", "Low", "Low+"],
        "Medium-": ["Medium-"],
        "Medium": ["Medium"],
        "High": ["Medium+", "High-", "High"],
        "High+": ["High+"],
    }
    assert (
        features("Qualitative_Ordinal").get_content() == expected_ordinal_08
    ), "Values not correctly grouped"

    expected_ordinal_lownan_08 = {
        "Low+": ["Low", "Low-", "Low+"],
        "Medium-": ["Medium-"],
        "Medium": ["Medium"],
        "High": ["Medium+", "High-", "High"],
        "High+": ["High+"],
    }
    assert (
        features("Qualitative_Ordinal_lownan").get_content() == expected_ordinal_lownan_08
    ), "NaNs should stay by themselves."
