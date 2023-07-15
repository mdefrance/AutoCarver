"""Set of tests for base_discretizers module."""

from numpy import nan
from pandas import DataFrame

from AutoCarver.discretizers.utils.base_discretizers import GroupedListDiscretizer
from AutoCarver.discretizers.utils.grouped_list import GroupedList

# TODO: test quantitative discretization
def test_grouped_list_discretizer(x_train: DataFrame, x_test_1: DataFrame, x_test_2: DataFrame):
    """Tests GroupedListDiscretizer

    Parameters
    ----------
    x_train : DataFrame
        Simulated Train DataFrame
    x_test_1 : DataFrame
        Simulated Test DataFrame
    x_test_2 : DataFrame
        Simulated Test DataFrame
    """

    # values to input nans
    str_nan = "__NAN__"

    # defining values_orders
    order = ["Low-", "Low", "Low+", "Medium-", "Medium", "Medium+", "High-", "High", "High+"]

    # ordering for base qualitative ordinal feature
    groupedlist = GroupedList(order)
    groupedlist.group_list(["Low-", "Low"], "Low+")
    groupedlist.group_list(["Medium+", "High-"], "High")

    # ordering for qualitative ordinal feature that contains NaNs
    groupedlist_lownan = GroupedList(order + [str_nan])
    groupedlist_lownan.group_list(["Low-", "Low"], "Low+")
    groupedlist_lownan.group_list(["Medium+", "High-"], "High")

    # storing per feature orders
    values_orders = {
        "Qualitative_Ordinal": groupedlist,
        "Qualitative_Ordinal_lownan": groupedlist_lownan,
    }
    features = ["Qualitative_Ordinal", "Qualitative_Ordinal_lownan"]

    # initiating discretizer
    discretizer = GroupedListDiscretizer(
        features=features,
        values_orders=values_orders,
        str_nan=str_nan,
        input_dtypes="str",
        copy=True,
    )
    x_discretized = discretizer.fit_transform(x_train)

    # testing ordinal qualitative feature discretization
    x_expected = x_train.copy()
    x_expected["Qualitative_Ordinal"] = (
        x_expected["Qualitative_Ordinal"]
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
    assert all(
        x_expected["Qualitative_Ordinal"] == x_discretized["Qualitative_Ordinal"]
    ), "incorrect discretization"

    # testing ordinal qualitative feature discretization with nans
    x_expected = x_train.copy()
    x_expected["Qualitative_Ordinal_lownan"] = (
        x_expected["Qualitative_Ordinal_lownan"]
        .replace("Low-", "Low+")
        .replace("Low", "Low+")
        .replace("Low+", "Low+")
        .replace("Medium-", "Medium-")
        .replace("Medium", "Medium")
        .replace("Medium+", "High")
        .replace("High-", "High")
        .replace("High", "High")
        .replace("High+", "High+")
        .replace(nan, "__NAN__")
    )
    assert all(
        x_expected["Qualitative_Ordinal_lownan"] == x_discretized["Qualitative_Ordinal_lownan"]
    ), "incorrect discretization with nans"

    # checking that other columns are left unchanged
    assert all(
        x_discretized["Quantitative"] == x_discretized["Quantitative"]
    ), "Other columns should not be modified"
