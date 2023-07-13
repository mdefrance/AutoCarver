"""Set of tests for base_discretizers module."""

from AutoCarver.discretizers.utils.base_discretizers import *
from pytest import raises
from numpy import nan

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


def test_grouped_list_functions():
    """Tests GroupedList functions"""
    
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
    str_nan = '__NAN__'
    
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
    features = ["Qualitative_Ordinal", "Qualitative_Ordinal_lownan"]
    
    # initiating discretizer 
    discretizer = GroupedListDiscretizer(features=features, values_orders=values_orders, str_nan=str_nan, input_dtypes='str', copy=True)
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
    assert all(x_expected["Qualitative_Ordinal"] == x_discretized["Qualitative_Ordinal"]), "incorrect discretization"

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
    assert all(x_expected["Qualitative_Ordinal_lownan"] == x_discretized["Qualitative_Ordinal_lownan"]), "incorrect discretization with nans"

    # checking that other columns are left unchanged
    assert all(x_discretized["Quantitative"] == x_discretized["Quantitative"]), "Other columns should not be modified"
