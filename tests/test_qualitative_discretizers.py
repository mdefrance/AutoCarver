"""Set of tests for qualitative_discretizers module."""

from AutoCarver.discretizers.utils.qualitative_discretizers import *
from pytest import fixture, raises


def test_chained_discretizer():
    """TODO"""
    pass


def test_default_discretizer(x_train: DataFrame):
    """Tests DefaultDiscretizer

    Parameters
    ----------
    x_train : DataFrame
        Simulated Train DataFrame
    """    

    # defining values_orders
    # order = list(x_train["Qualitative"].unique())
    order = ['Category A', 'Category B', 'Category C', 'Category D', 'Category E', 'Category F']
    # ordering for base qualitative ordinal feature
    groupedlist = GroupedList(order)
    groupedlist_lownan = GroupedList(['Category C', 'Category D', 'Category E', 'Category F'])
    groupedlist_highnan = GroupedList(['Category A',  'Category C', 'Category D', 'Category E'])

    # ordering for qualitative ordinal feature that contains NaNs
    groupedlist_grouped = GroupedList(order)
    groupedlist_grouped.group("Category A", "Category D")


    # ordering for base qualitative ordinal feature
    order = ['Low-', 'Low', 'Low+', 'Medium-', 'Medium', 'Medium+', 'High-', 'High', 'High+']
    groupedlist_ordinal = GroupedList(order)
    groupedlist_ordinal.group_list(['Low-', 'Low'], 'Low+')
    groupedlist_ordinal.group_list(['Medium+', 'High-'], 'High')

    # storing per feature orders
    values_orders = {
        "Qualitative_grouped": groupedlist_grouped,
        "Qualitative_highnan": groupedlist,
        "Qualitative_lownan": groupedlist,
        "Qualitative_Ordinal": groupedlist_ordinal,  # TODO: check that it has not been changed
    }

    features = ["Qualitative", "Qualitative_grouped", "Qualitative_lownan", "Qualitative_highnan"]

    # unwanted value in values_orders
    with raises(AssertionError):
        min_freq = 0.02

        discretizer = DefaultDiscretizer(features, min_freq, values_orders=values_orders, copy=True)
        x_discretized = discretizer.fit_transform(x_train, x_train["quali_ordinal_target"])

    # correct feature ordering
    groupedlist_grouped.group("Category B", "Category D")
    values_orders = {
        "Qualitative_grouped": groupedlist_grouped,
        "Qualitative_highnan": groupedlist_highnan,
        "Qualitative_lownan": groupedlist_lownan,
        "Qualitative_Ordinal": groupedlist_ordinal,
    }

    min_freq = 0.02

    discretizer = DefaultDiscretizer(features, min_freq, values_orders=values_orders, copy=True)
    x_discretized = discretizer.fit_transform(x_train, x_train["quali_ordinal_target"])

    assert discretizer.values_orders['Qualitative_Ordinal'].contained == groupedlist_ordinal.contained, "Column names of values_orders not provided if features should not be discretized."
    quali_expected = {
        '__OTHER__': ['Category A', '__OTHER__'],
         'Category C': ['Category C'],
         'Category F': ['Category F'],
         'Category E': ['Category E'],
         'Category D': ['Category D']
    }
    assert discretizer.values_orders['Qualitative'].contained == quali_expected, "Values less frequent than min_freq should be grouped into default_value"
    quali_lownan_expected = {
        '__OTHER__': ['__NAN__', '__OTHER__'],
         'Category C': ['Category C'],
         'Category F': ['Category F'],
         'Category E': ['Category E'],
         'Category D': ['Category D']
    }
    assert discretizer.values_orders['Qualitative_lownan'].contained == quali_lownan_expected, "If any, NaN values should be put into str_nan"
    quali_highnan_expected = {
        '__OTHER__': ['Category A', '__OTHER__'],
        'Category C': ['Category C'],
        '__NAN__': ['__NAN__'],
        'Category E': ['Category E'],
        'Category D': ['Category D']
    }
    assert discretizer.values_orders['Qualitative_highnan'].contained == quali_highnan_expected, "If any, NaN values should be put into str_nan"
    assert discretizer.values_orders['Qualitative_grouped'].contained == groupedlist_grouped.contained, "Grouped values should stay grouped"
