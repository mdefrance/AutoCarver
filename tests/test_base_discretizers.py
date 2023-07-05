"""Set of tests for base_discretizers module."""

from AutoCarver.discretizers.utils.base_discretizers import *
from pytest import fixture, raises
from pandas import DataFrame
from numpy import nan, random

def test_grouped_list_init():
    """ Tests the initialization of a GroupedList"""
    
    # init by list
    test_list = ['1', '2', '3']
    groupedlist = GroupedList(test_list)
    assert groupedlist == test_list, "When init by list, GroupedList.contained should have only one value per key"
    assert groupedlist.contained == {'1': ['1'], '2': ['2'], '3': ['3']}, "When init by list, GroupedList.contained should have only one value per key: itself"
    
    # init by dict
    test_dict = {'1': ['1', '4'], '2': ['2'], '3': ['3']}
    groupedlist = GroupedList(test_dict)
    assert groupedlist == ['1', '2', '3'], "When init by dict, all keys should be in the list"
    assert groupedlist.contained == {'1': ['1', '4'], '2': ['2'], '3': ['3']},"When init by dict, all values should be in stored in the contained dict"

    test_dict = {'1': ['1', '4'], '2': ['2'], '3': ['3'], '4': []}
    groupedlist = GroupedList(test_dict)
    assert groupedlist == ['1', '2', '3'], "When init by dict, keys that are in another key (group) should be popped"
    assert groupedlist.contained == {'1': ['1', '4'], '2': ['2'], '3': ['3']},"When init by dict, all values should be in stored in the contained dict"

    test_dict = {'1': ['1', '4'], '2': ['2'], '3': ['3'], '4': []}
    groupedlist = GroupedList(test_dict)
    assert groupedlist == ['1', '2', '3'], "When init by dict, keys that are in another key (group) should be popped"
    assert groupedlist.contained == {'1': ['1', '4'], '2': ['2'], '3': ['3']},"When init by dict, all values should be in stored in the contained dict"

    # check that a value can not be in several keys (groups)
    with raises(AssertionError):
        test_dict = {'1': ['1', '4'], '2': ['2'], '3': ['3'], '4': ['4']}
        groupedlist = GroupedList(test_dict)
    
    test_dict = {'1': ['1', '4'], '2': ['2'], '3': ['3'], '4': [], '5': []}
    groupedlist = GroupedList(test_dict)
    assert groupedlist == ['1', '2', '3', '5'], "When init by dict, keys that are in no key (group) should be kept in the list"
    assert groupedlist.contained == {'1': ['1', '4'], '2': ['2'], '3': ['3'], '5': ['5']},"When init by dict, keys that are in no key (group) should be added to themselves"
    
    # init by copy
    groupedlist_copy = GroupedList(groupedlist)
    assert groupedlist_copy == ['1', '2', '3', '5'], "When init by GroupedList, GroupedList should be an exact copy"
    assert groupedlist_copy.contained == {'1': ['1', '4'], '2': ['2'], '3': ['3'], '5': ['5']}, "When init by GroupedList, GroupedList should be an exact copy"



def init_test_df(seed: int, size: int = 1000) -> DataFrame:
    """Initializes a DataFrame used in tests

    Parameters
    ----------
    seed : int
        Seed for the random samples
    size : int
        Generated sample size

    Returns
    -------
    DataFrame
        A DataFrame to perform Discretizers tests
    """    
    
    # Set random seed for reproducibility
    random.seed(seed)

    # Generate random qualitative ordinal features
    qual_ord_features = (
        ['Low-'] * int(1 * 100) + ['Low'] * int(0 * 100) + ['Low+'] * int(12 * 100) +  # 13%
        ['Medium-'] * int(10 * 100) + ['Medium'] * int(24 * 100) + ['Medium+'] * int(6 * 100) +  # 40%
        ['High-'] * int(0 * 100) + ['High'] * int(7 * 100) + ['High+'] * int(40 * 100) # 47 %
    )
    # qual_ord_features = ['Low-', 'Low', 'Low+', 'Medium-', 'Medium', 'Medium+', 'High-', 'High', 'High+']
    ordinal_data = random.choice(qual_ord_features, size=size)
    
    # adding binary target associated to qualitative ordinal feature
    binary = [1 - (qual_ord_features.index(val) / (len(qual_ord_features) - 1)) for val in ordinal_data]

    # Generate random qualitative features
    qual_features = ['Category A', 'Category B', 'Category C']
    qualitative_data = random.choice(qual_features, size=size)

    # Generate random quantitative features
    quantitative_data = random.rand(size) * 100

    # Create DataFrame
    data = {
        'Qualitative_Ordinal': ordinal_data,
        'Qualitative': qualitative_data,
        'Quantitative': quantitative_data,
        'Binary': binary
    }
    df = DataFrame(data)
    
    df["quali_ordinal_target"] = df["Binary"].apply(
        lambda u:
        random.choice([0, 1], p=[1-(u*1/3), (u*1/3)])
    )

    # building specific cases
    df["Qualitative_Ordinal_lownan"] = df["Qualitative_Ordinal"].replace("Low-", nan)
    df["Qualitative_Ordinal_highnan"] = df["Qualitative_Ordinal"].replace("High+", nan)

    return df


def test_grouped_list_functions():
    
    # init a groupedlist
    test_dict = {'1': ['1', '4'], '2': ['2'], '3': ['3'], '4': [], '5': []}
    groupedlist = GroupedList(test_dict)
    
    # test get
    assert groupedlist.get('1') == ['1', '4']
    
    
    # test group
    groupedlist.group('1', '5')
    assert groupedlist == ['2', '3', '5']
    assert groupedlist.contained == {'2': ['2'], '3': ['3'], '5': [ '1', '4', '5']}
    with raises(AssertionError):
        groupedlist.group('7', '5')
        groupedlist.group('5', '8')
    
    # test group_list
    groupedlist.group_list(['2', '3'], '5')
    assert groupedlist == ['5']
    assert groupedlist.contained == {'5': [ '3', '2', '1', '4', '5']}
    
    # test append
    groupedlist.append('0')
    assert groupedlist == ['5', '0']
    assert groupedlist.contained == {'5': [ '3', '2', '1', '4', '5'], '0': ['0']}
    
    # test update
    groupedlist.update({'0': ['0', '7']})
    assert groupedlist == ['5', '0']
    assert groupedlist.contained == {'5': [ '3', '2', '1', '4', '5'], '0': ['0', '7']}
    
    # test sort
    groupedlist = groupedlist.sort()
    assert groupedlist == ['0', '5']
    assert groupedlist.contained == {'0': ['0', '7'], '5': [ '3', '2', '1', '4', '5']}
    
    # test sort_by
    groupedlist = groupedlist.sort_by(['5', '0'])
    assert groupedlist == ['5', '0']
    assert groupedlist.contained == {'5': [ '3', '2', '1', '4', '5'], '0': ['0', '7']}
    with raises(AssertionError):
        groupedlist.sort_by(['5', '0', '1'])
        groupedlist.sort_by(['5'])
    
    # test remove
    groupedlist.append('15')
    groupedlist.remove('15')
    assert groupedlist == ['5', '0']
    assert groupedlist.contained == {'5': [ '3', '2', '1', '4', '5'], '0': ['0', '7']}
    
    # test pop
    groupedlist.append('15')
    groupedlist.pop(len(groupedlist) - 1)
    assert groupedlist == ['5', '0']
    assert groupedlist.contained == {'5': [ '3', '2', '1', '4', '5'], '0': ['0', '7']}
    
    # test get_group
    groupedlist.append('15')
    groupedlist.pop(len(groupedlist) - 1)
    assert groupedlist.get_group('3') == '5'
    assert groupedlist.get_group('0') == '0'
    
    # test values
    assert groupedlist.values() == [ '3', '2', '1', '4', '5', '0', '7']
    
    # test contains
    assert groupedlist.contains('3')
    assert not groupedlist.contains('15')
    
    # test get_repr
    assert groupedlist.get_repr() == ['5 to 3', '7 and 0']

def test_grouped_list_discretizer():
    """ Tests a basic GroupedListDiscretizer"""
    
    # values to input nans
    str_nan = '__NAN__'
    
    # initiating test dataframe
    x_train = init_test_df(123)
    
    # defining values_orders
    order = ['Low-', 'Low', 'Low+', 'Medium-', 'Medium', 'Medium+', 'High-', 'High', 'High+']
    
    # ordering for base qualitative ordinal feature
    groupedlist = GroupedList(order)
    groupedlist.group_list(['Low-', 'Low'], 'Low+')
    groupedlist.group_list(['Medium+', 'High-'], 'High')

    # ordering for qualitative ordinal feature that contains NaNs
    groupedlist_lownan = GroupedList(order + [str_nan])
    groupedlist_lownan.group_list(['Low-', 'Low'], 'Low+')
    groupedlist_lownan.group_list(['Medium+', 'High-'], 'High')

    # storing per feature orders
    values_orders = {
        "Qualitative_Ordinal": groupedlist,
        "Qualitative_Ordinal_lownan": groupedlist_lownan,
    }
    
    # initiating discretizer with output str
    discretizer = GroupedListDiscretizer(values_orders, str_nan=str_nan, output='str', copy=True)
    x_discretized = discretizer.fit_transform(x_train)

    # testing ordinal qualitative feature discretization
    x_expected = x_train.copy()
    x_expected["Qualitative_Ordinal"] = x_expected["Qualitative_Ordinal"].replace('Low-', 'Low+')\
        .replace('Low', 'Low+')\
        .replace('Low+', 'Low+')\
        .replace('Medium-', 'Medium-')\
        .replace('Medium', 'Medium')\
        .replace('Medium+', 'High')\
        .replace('High-', 'High')\
        .replace('High', 'High')\
        .replace('High+', 'High+')
    assert all(x_expected["Qualitative_Ordinal"] == x_discretized["Qualitative_Ordinal"]), "incorrect discretization, for output 'str'"

    # testing ordinal qualitative feature discretization with nans
    x_expected = x_train.copy()
    x_expected["Qualitative_Ordinal_lownan"] = x_expected["Qualitative_Ordinal_lownan"].replace('Low-', 'Low+')\
        .replace('Low', 'Low+')\
        .replace('Low+', 'Low+')\
        .replace('Medium-', 'Medium-')\
        .replace('Medium', 'Medium')\
        .replace('Medium+', 'High')\
        .replace('High-', 'High')\
        .replace('High', 'High')\
        .replace('High+', 'High+')\
        .replace(nan, '__NAN__')
    assert all(x_expected["Qualitative_Ordinal_lownan"] == x_discretized["Qualitative_Ordinal_lownan"]), "incorrect discretization with nans, for output 'str'"

    # checking that other columns are left unchanged
    assert all(x_discretized["Quantitative"] == x_discretized["Quantitative"]), "Other columns should not be modified"

    # initiating discretizer with output float
    discretizer = GroupedListDiscretizer(values_orders, str_nan=str_nan, output='float', copy=True)
    x_discretized = discretizer.fit_transform(x_train)

    # testing ordinal qualitative feature discretization
    x_expected = x_train.copy()
    x_expected["Qualitative_Ordinal"] = x_expected["Qualitative_Ordinal"].replace('Low-', 0)\
        .replace('Low', 0)\
        .replace('Low+', 0)\
        .replace('Medium-', 1)\
        .replace('Medium', 2)\
        .replace('Medium+', 3)\
        .replace('High-', 3)\
        .replace('High', 3)\
        .replace('High+', 4)
    assert all(x_expected["Qualitative_Ordinal"] == x_discretized["Qualitative_Ordinal"]), "incorrect discretization, for output 'float'"
    
    # testing ordinal qualitative feature discretization with nans
    x_expected = x_train.copy()
    x_expected["Qualitative_Ordinal_lownan"] = x_expected["Qualitative_Ordinal_lownan"].replace('Low-', 0)\
        .replace('Low', 0)\
        .replace('Low+', 0)\
        .replace('Medium-', 1)\
        .replace('Medium', 2)\
        .replace('Medium+', 3)\
        .replace('High-', 3)\
        .replace('High', 3)\
        .replace('High+', 4)\
        .replace(nan, 5)
    assert all(x_expected["Qualitative_Ordinal_lownan"] == x_discretized["Qualitative_Ordinal_lownan"]), "incorrect discretization with nans, for output 'float'"

    # testing by adding nan in test set
    x_test = init_test_df(1234)
    discretizer.transform(x_test)
    x_test['Qualitative_Ordinal'] = x_test['Qualitative_Ordinal'].replace('High+', nan)
    with raises(AssertionError):
        discretizer.transform(x_test)

    # testing by adding a new value in test set
    x_test = init_test_df(12345)
    x_test['Qualitative_Ordinal'] = x_test['Qualitative_Ordinal'].replace('High+', 'High++')
    with raises(AssertionError):
        discretizer.transform(x_test)


def test_closest_discretizer():
    """ Tests a ClosestDiscretizer"""
    
    # initiating test dataframe
    x_train = init_test_df(123, 10000)

    # defining values_orders
    order = ['Low-', 'Low', 'Low+', 'Medium-', 'Medium', 'Medium+', 'High-', 'High', 'High+']
    # ordering for base qualitative ordinal feature
    groupedlist = GroupedList(order)

    # ordering for qualitative ordinal feature that contains NaNs
    groupedlist_lownan = GroupedList(order)

    # storing per feature orders
    values_orders = {
        "Qualitative_Ordinal": groupedlist,
        "Qualitative_Ordinal_lownan": groupedlist_lownan,
    }

    # minimum frequency per modality + apply(find_common_modalities) outputs a Series
    min_freq = 0.01

    # discretizing features
    discretizer = ClosestDiscretizer(values_orders, min_freq, copy=True)
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
        'Low+': ['Low-', 'Low','Low+'],
        'Medium-': ['Medium-'],
        'Medium': ['Medium'],
        'Medium+': ['High-', 'Medium+'],
        'High': ['High'],
        'High+': ['High+']
    }
    assert discretizer.values_orders['Qualitative_Ordinal'].contained == expected_ordinal_01, "Missing value in order not correctly grouped"
    assert discretizer.values_orders['Qualitative_Ordinal_lownan'].contained == expected_ordinal_lownan_01, "Missing value in order not correctly grouped or introduced nans."

    # minimum frequency per modality + apply(find_common_modalities) outputs a DataFrame
    min_freq = 0.08

    # discretizing features
    discretizer = ClosestDiscretizer(values_orders, min_freq, copy=True)
    discretizer.fit_transform(x_train, x_train["quali_ordinal_target"])

    expected_ordinal_08 = {
        'Low+': ['Low', 'Low-', 'Low+'],
        'Medium-': ['Medium-'],
        'Medium': ['Medium'],
        'High': ['High-', 'Medium+', 'High'],
        'High+': ['High+']
    }
    expected_ordinal_lownan_08 = {
        'Low+': ['Low-', 'Low', 'Low+'],
        'Medium-': ['Medium-'],
        'Medium': ['Medium'],
        'High': ['High-', 'Medium+', 'High'],
        'High+': ['High+']
    }
    assert discretizer.values_orders['Qualitative_Ordinal'].contained == expected_ordinal_08, "Values not correctly grouped"
    assert discretizer.values_orders['Qualitative_Ordinal_lownan'].contained == expected_ordinal_lownan_08, "Values not correctly grouped or introduced nans."
