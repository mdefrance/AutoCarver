"""Set of tests for qualitative_discretizers module."""

from AutoCarver.discretizers.utils.qualitative_discretizers import *
from pytest import raises


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
        "Qualitative_Ordinal": groupedlist_ordinal,
    }

    features = ["Qualitative", "Qualitative_grouped", "Qualitative_lownan", "Qualitative_highnan"]

    min_freq = 0.02
    # unwanted value in values_orders
    with raises(AssertionError):

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


def test_ordinal_discretizer(x_train: DataFrame):
    """Tests OrdinalDiscretizer

    # TODO: add tests for quantitative features

    Parameters
    ----------
    x_train : DataFrame
        Simulated Train DataFrame
    x_test_1 : DataFrame
        Simulated Test DataFrame
    x_test_2 : DataFrame
        Simulated Test DataFrame
    """

    # defining values_orders
    order = ['Low-', 'Low', 'Low+', 'Medium-', 'Medium', 'Medium+', 'High-', 'High', 'High+']
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
    discretizer = OrdinalDiscretizer(features, values_orders, min_freq, copy=True)
    discretizer.fit_transform(x_train, x_train["quali_ordinal_target"])

    expected_ordinal_01 = {
        'Low-': ['Low', 'Low-'],
        'Low+': ['Low+'],
        'Medium-': ['Medium-'],
        'Medium': ['Medium'],
        'Medium+': ['High-', 'Medium+'],
        'High': ['High'],
        'High+': ['High+']
    }
    expected_ordinal_lownan_01 = {
        'Low+': ['Low', 'Low-', 'Low+'],
        'Medium-': ['Medium-'],
        'Medium': ['Medium'],
        'Medium+': ['High-', 'Medium+'],
        'High': ['High'],
        'High+': ['High+'],
        '__NAN__': ['__NAN__']
    }
    assert discretizer.values_orders['Qualitative_Ordinal'].contained == expected_ordinal_01, "Missing value in order not correctly grouped"
    assert discretizer.values_orders['Qualitative_Ordinal_lownan'].contained == expected_ordinal_lownan_01, "Missing value in order not correctly grouped or introduced nans."

    # minimum frequency per modality + apply(find_common_modalities) outputs a DataFrame
    min_freq = 0.08

    # discretizing features
    discretizer = OrdinalDiscretizer(features, values_orders, min_freq, copy=True)
    discretizer.fit_transform(x_train, x_train["quali_ordinal_target"])

    expected_ordinal_08 = {
        'Low+': ['Low-', 'Low', 'Low+'],
        'Medium-': ['Medium-'],
        'Medium': ['Medium'],
        'High': ['Medium+', 'High-', 'High'],
        'High+': ['High+']
    }
    expected_ordinal_lownan_08 = {
        'Low+': ['Low', 'Low-', 'Low+'],
        'Medium-': ['Medium-'],
        'Medium': ['Medium'],
        'High': ['Medium+', 'High-', 'High'],
        'High+': ['High+'],
        '__NAN__': ['__NAN__']
    }
    assert discretizer.values_orders['Qualitative_Ordinal'].contained == expected_ordinal_08, "Values not correctly grouped"
    assert discretizer.values_orders['Qualitative_Ordinal_lownan'].contained == expected_ordinal_lownan_08, "NaNs should stay by themselves."
