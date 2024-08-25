"""Set of tests for base_discretizers module."""

from numpy import nan
from pandas import DataFrame
from pytest import FixtureRequest, fixture

from AutoCarver.discretizers import BaseDiscretizer
from AutoCarver.features import Features, GroupedList


@fixture(params=[True, False])
def dropna(request: FixtureRequest) -> str:
    return request.param


# TODO: test quantitative discretization
def test_base_discretizer(x_train: DataFrame, dropna: bool) -> None:
    """Tests BaseDiscretizer

    Parameters
    ----------
    x_train : DataFrame
        Simulated Train DataFrame
    """

    # values to input nans
    str_nan = "NAN"
    # dropna = True

    # defining values_orders
    order = ["Low-", "Low", "Low+", "Medium-", "Medium", "Medium+", "High-", "High", "High+"]

    # ordering for base qualitative ordinal feature
    groupedlist = GroupedList(order)
    groupedlist.group(["Low-", "Low"], "Low+")
    groupedlist.group(["Medium+", "High-"], "High")

    # ordering for qualitative ordinal feature that contains NaNs
    groupedlist_lownan = GroupedList(order)
    groupedlist_lownan.group(["Low-", "Low"], "Low+")
    groupedlist_lownan.group(["Medium+", "High-"], "High")

    # storing per feature orders
    ordinal_values = {
        "Qualitative_Ordinal": groupedlist,
        "Qualitative_Ordinal_lownan": groupedlist_lownan,
    }
    ordinals = ["Qualitative_Ordinal", "Qualitative_Ordinal_lownan"]
    features = Features(
        ordinals=ordinals, ordinal_values=ordinal_values, nan=str_nan, dropna=dropna
    )
    features.fit(x_train)
    feature = features("Qualitative_Ordinal_lownan")
    print(feature.has_nan, feature.dropna, feature.content)

    # initiating discretizer
    discretizer = BaseDiscretizer(features=features, dropna=dropna, copy=True)
    x_discretized = discretizer.fit_transform(x_train)

    # testing ordinal qualitative feature discretization
    x_expected = x_train.copy()
    feature = "Qualitative_Ordinal"
    x_expected[feature] = (
        x_expected[feature]
        .replace("Low-", "Low+")
        .replace("Low", "Low+")
        .replace("Low+", "Low+")
        .replace("Medium-", "Medium-")
        .replace("Medium", "Medium")
        .replace("Medium+", "High")
        .replace("High-", "High")
        .replace("High", "High")
        .replace("High+", "High+")
    )
    assert all(x_expected[feature] == x_discretized[feature]), "incorrect discretization"

    # testing ordinal qualitative feature discretization with nans
    feature = "Qualitative_Ordinal_lownan"
    x_expected[feature] = (
        x_expected[feature]
        .replace("Low-", "Low+")
        .replace("Low", "Low+")
        .replace("Low+", "Low+")
        .replace("Medium-", "Medium-")
        .replace("Medium", "Medium")
        .replace("Medium+", "High")
        .replace("High-", "High")
        .replace("High", "High")
        .replace("High+", "High+")
    )
    # replacing nans if requested
    if dropna:
        x_expected[feature] = x_expected[feature].replace(nan, str_nan)

    assert all(x_expected[feature].isna() == x_discretized[feature].isna()), "unexpected NaNs"

    non_nans = x_expected[feature].notna()
    print(x_expected.loc[non_nans, feature].value_counts())
    print(x_discretized.loc[non_nans, feature].value_counts())
    assert all(
        x_expected.loc[non_nans, feature] == x_discretized.loc[non_nans, feature]
    ), "incorrect discretization with nans"

    # checking that other columns are left unchanged
    feature = "Quantitative"
    assert all(x_discretized[feature] == x_discretized[feature]), "Others should not be modified"
