"""Set of tests for base_discretizers module."""

from numpy import nan
from pandas import DataFrame
from pytest import raises

from AutoCarver.discretizers.utils.grouped_list import GroupedList


def test_grouped_list_init():
    """Tests the initialization of a GroupedList"""

    # init by list
    test_list = ["1", "2", "3"]
    groupedlist = GroupedList(test_list)
    assert (
        groupedlist == test_list
    ), "When init by list, GroupedList.content should have only one value per key"
    assert groupedlist.content == {
        "1": ["1"],
        "2": ["2"],
        "3": ["3"],
    }, "When init by list, GroupedList.content should have only one value per key: itself"

    # init by dict
    test_dict = {"1": ["1", "4"], "2": ["2"], "3": ["3"]}
    groupedlist = GroupedList(test_dict)
    assert groupedlist == ["1", "2", "3"], "When init by dict, all keys should be in the list"
    assert groupedlist.content == {
        "1": ["1", "4"],
        "2": ["2"],
        "3": ["3"],
    }, "When init by dict, all values should be in stored in the content dict"

    test_dict = {"1": ["1", "4"], "2": ["2"], "3": ["3"], "4": []}
    groupedlist = GroupedList(test_dict)
    assert groupedlist == [
        "1",
        "2",
        "3",
    ], "When init by dict, keys that are in another key (group) should be popped"
    assert groupedlist.content == {
        "1": ["1", "4"],
        "2": ["2"],
        "3": ["3"],
    }, "When init by dict, all values should be in stored in the content dict"

    test_dict = {"1": ["1", "4"], "2": ["2"], "3": ["3"], "4": []}
    groupedlist = GroupedList(test_dict)
    assert groupedlist == [
        "1",
        "2",
        "3",
    ], "When init by dict, keys that are in another key (group) should be popped"
    assert groupedlist.content == {
        "1": ["1", "4"],
        "2": ["2"],
        "3": ["3"],
    }, "When init by dict, all values should be in stored in the content dict"

    # check that a value can not be in several keys (groups)
    with raises(AssertionError):
        test_dict = {"1": ["1", "4"], "2": ["2"], "3": ["3"], "4": ["4"]}
        groupedlist = GroupedList(test_dict)

    test_dict = {"1": ["1", "4"], "2": ["2"], "3": ["3"], "4": [], "5": []}
    groupedlist = GroupedList(test_dict)
    assert groupedlist == [
        "1",
        "2",
        "3",
        "5",
    ], "When init by dict, keys that are in no key (group) should be kept in the list"
    assert groupedlist.content == {
        "1": ["1", "4"],
        "2": ["2"],
        "3": ["3"],
        "5": ["5"],
    }, "When init by dict, keys that are in no key (group) should be added to themselves"

    # init by copy
    groupedlist_copy = GroupedList(groupedlist)
    assert groupedlist_copy == [
        "1",
        "2",
        "3",
        "5",
    ], "When init by GroupedList, GroupedList should be an exact copy"
    assert groupedlist_copy.content == {
        "1": ["1", "4"],
        "2": ["2"],
        "3": ["3"],
        "5": ["5"],
    }, "When init by GroupedList, GroupedList should be an exact copy"


def test_grouped_list_functions():
    """Tests GroupedList functions"""

    # init a groupedlist
    test_dict = {"1": ["1", "4"], "2": ["2"], "3": ["3"], "4": [], "5": []}
    groupedlist = GroupedList(test_dict)

    # test get
    assert groupedlist.get("1") == ["1", "4"]

    # test group
    groupedlist.group("1", "5")
    assert groupedlist == ["2", "3", "5"]
    assert groupedlist.content == {"2": ["2"], "3": ["3"], "5": ["1", "4", "5"]}
    with raises(AssertionError):
        groupedlist.group("7", "5")
        groupedlist.group("5", "8")

    # test group_list
    groupedlist.group_list(["2", "3"], "5")
    assert groupedlist == ["5"]
    assert groupedlist.content == {"5": ["3", "2", "1", "4", "5"]}

    # test append
    groupedlist.append("0")
    assert groupedlist == ["5", "0"]
    assert groupedlist.content == {"5": ["3", "2", "1", "4", "5"], "0": ["0"]}

    # test update
    groupedlist.update({"0": ["0", "7"]})
    assert groupedlist == ["5", "0"]
    assert groupedlist.content == {"5": ["3", "2", "1", "4", "5"], "0": ["0", "7"]}

    # test sort
    groupedlist = groupedlist.sort()
    assert groupedlist == ["0", "5"]
    assert groupedlist.content == {"0": ["0", "7"], "5": ["3", "2", "1", "4", "5"]}

    # test sort_by
    groupedlist = groupedlist.sort_by(["5", "0"])
    assert groupedlist == ["5", "0"]
    assert groupedlist.content == {"5": ["3", "2", "1", "4", "5"], "0": ["0", "7"]}
    with raises(AssertionError):
        groupedlist.sort_by(["5", "0", "1"])
        groupedlist.sort_by(["5"])

    # test remove
    groupedlist.append("15")
    groupedlist.remove("15")
    assert groupedlist == ["5", "0"]
    assert groupedlist.content == {"5": ["3", "2", "1", "4", "5"], "0": ["0", "7"]}

    # test pop
    groupedlist.append("15")
    groupedlist.pop(len(groupedlist) - 1)
    assert groupedlist == ["5", "0"]
    assert groupedlist.content == {"5": ["3", "2", "1", "4", "5"], "0": ["0", "7"]}

    # test get_group
    groupedlist.append("15")
    groupedlist.pop(len(groupedlist) - 1)
    assert groupedlist.get_group("3") == "5"
    assert groupedlist.get_group("0") == "0"

    # test values
    assert groupedlist.values() == ["3", "2", "1", "4", "5", "0", "7"]

    # test contains
    assert groupedlist.contains("3")
    assert not groupedlist.contains("15")

    # test get_repr
    assert groupedlist.get_repr() == ["5 to 3", "7 and 0"]
