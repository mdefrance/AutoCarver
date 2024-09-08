"""Set of tests for base_discretizers module."""

from numpy import array, nan
from pytest import raises

from AutoCarver.features.utils.grouped_list import GroupedList, is_equal


def test_is_equal() -> None:
    """tests function is_equal"""

    assert not is_equal("a", "b")
    assert not is_equal("a", nan)
    assert not is_equal(1, 2)
    assert not is_equal(1, nan)

    assert is_equal("a", "a")
    assert is_equal(1, 1)
    assert is_equal(nan, nan)


def test_grouped_list_init_with_ndarray() -> None:
    arr = array([1, 2, 3])
    gl = GroupedList(arr)
    assert gl == [1, 2, 3]
    assert gl.content == {1: [1], 2: [2], 3: [3]}


def test_grouped_list_init_with_dict() -> None:
    d = {"a": ["b", "c", "a"], "d": ["e", "d"]}
    gl = GroupedList(d)
    assert gl == ["a", "d"]
    assert gl.content == {"a": ["b", "c", "a"], "d": ["e", "d"]}

    d = {"a": ["b", "c"], "d": ["e"]}
    gl = GroupedList(d)
    assert gl == ["a", "d"]
    assert gl.content == {"a": ["b", "c", "a"], "d": ["e", "d"]}

    d = {"a": ["b", "c"], "d": ["e"], "e": []}
    gl = GroupedList(d)
    assert gl == ["a", "d"]
    assert gl.content == {"a": ["b", "c", "a"], "d": ["e", "d"]}

    d = {"a": ["b", "c"], "d": ["e"], "e": [], "f": []}
    gl = GroupedList(d)
    assert gl == ["a", "d", "f"]
    assert gl.content == {"a": ["b", "c", "a"], "d": ["e", "d"], "f": ["f"]}

    # check that a value can not be in several keys (groups)
    with raises(ValueError):
        d = {"1": ["1", "4"], "2": ["2"], "3": ["3"], "4": ["4"]}
        gl = GroupedList(d)


def test_grouped_list_init_with_list() -> None:
    lst = [1, 2, 3]
    gl = GroupedList(lst)
    assert gl == [1, 2, 3]
    assert gl.content == {1: [1], 2: [2], 3: [3]}
    assert gl == lst


def test_grouped_list_init_with_grouped_list() -> None:
    initial = GroupedList([1, 2, 3])
    gl = GroupedList(initial)
    assert gl == [1, 2, 3]
    assert gl == initial
    assert gl.content == initial.content


def test_grouped_list_get_existing_key() -> None:
    gl = GroupedList({"a": ["b", "c"], "d": ["d"]})
    assert gl.get("a") == ["b", "c", "a"]
    assert gl.get("d") == ["d"]
    assert gl.get("d", default="x") == ["d"]
    assert gl.get("d", default=["x"]) == ["d"]


def test_grouped_list_get_non_existing_key() -> None:
    gl = GroupedList({"a": ["b", "c"]})
    assert gl.get("z") == []
    assert gl.get("z", default="x") == ["x"]
    assert gl.get("z", default=["x"]) == ["x"]


def test_grouped_list_group() -> None:
    # trying with a signle value
    gl = GroupedList({"a": ["b"], "c": ["d"]})
    gl.group("a", "c")
    assert gl == ["c"]
    assert gl.content == {"c": ["b", "a", "d", "c"]}

    with raises(ValueError):
        gl.group("c", "d")  # d is not in grouped_list

    with raises(ValueError):
        gl.group("d", "c")  # d is not in grouped_list

    # trying with a list
    gl = GroupedList({"a": ["b"], "c": ["d"]})
    gl.group(["a", "c"], "c")
    assert gl == ["c"]
    assert gl.content == {"c": ["b", "a", "d", "c"]}

    gl = GroupedList({"a": ["b"], "c": ["d"], "e": ["e"]})
    gl.group(["a", "e"], "c")
    assert gl == ["c"]
    assert gl.content == {"c": ["e", "b", "a", "d", "c"]}

    with raises(ValueError):
        gl.group(["a", "e"], "g")  # g is not in grouped_list

    with raises(ValueError):
        gl.group(["a", "g"], "c")  # g is not in grouped_list


def test_grouped_list_append() -> None:
    gl = GroupedList([1, 2, 3])
    gl.append(4)
    assert gl == [1, 2, 3, 4]
    assert gl.content == {1: [1], 2: [2], 3: [3], 4: [4]}

    gl = GroupedList({"a": ["b", "c"]})
    gl.append(4)
    assert gl == ["a", 4]
    assert gl.content == {"a": ["b", "c", "a"], 4: [4]}

    # existing value
    gl = GroupedList([1, 2, 3])
    with raises(ValueError):
        gl.append(2)


def test_grouped_list_update() -> None:
    gl = GroupedList([1, 2])
    gl.update({3: [3, 4], 1: [1, 6]})
    assert gl == [1, 2, 3]
    assert gl.content == {1: [1, 6], 2: [2], 3: [3, 4]}

    gl = GroupedList([1, 2])
    gl.update({3: [4, 5]})
    assert gl == [1, 2, 3]
    assert gl.content == {1: [1], 2: [2], 3: [4, 5, 3]}

    with raises(ValueError):  # not dict input
        gl.update([1, 2, 3])

    with raises(ValueError):  # not dict of list input
        gl.update({3: 4, 1: 6})

    # value in several groups
    gl = GroupedList({1: [1], 2: [6, 2], 3: [4, 5, 3]})
    with raises(ValueError):
        gl.update({3: [6]})


def test_grouped_list_sort() -> None:
    gl = GroupedList({"b": ["x"], "a": ["y"], 1: ["z"]})
    sorted_gl = gl.sort()
    assert sorted_gl == ["a", "b", 1]
    assert sorted_gl.content == {"a": ["y", "a"], "b": ["x", "b"], 1: ["z", 1]}

    gl = GroupedList({"b": ["x"], "a": ["y"], 1: ["z"], 0.5: ["v"]})
    sorted_gl = gl.sort()
    assert sorted_gl == ["a", "b", 0.5, 1]
    assert sorted_gl.content == {"a": ["y", "a"], "b": ["x", "b"], 0.5: ["v", 0.5], 1: ["z", 1]}


def test_grouped_list_sort_by() -> None:
    gl = GroupedList({"b": ["x"], "a": ["y"], 1: ["z"]})
    sorted_gl = gl.sort_by([1, "a", "b"])
    assert sorted_gl == [1, "a", "b"]
    assert sorted_gl.content == {"a": ["y", "a"], "b": ["x", "b"], 1: ["z", 1]}

    gl = GroupedList({"b": ["x"], "a": ["y"], 1: ["z"], 0.5: ["v"]})
    sorted_gl = gl.sort_by([0.5, "a", 1, "b"])
    assert sorted_gl == [0.5, "a", 1, "b"]
    assert sorted_gl.content == {0.5: ["v", 0.5], "a": ["y", "a"], 1: ["z", 1], "b": ["x", "b"]}

    # unexpected value in provided order
    with raises(ValueError):
        gl.sort_by([0.5, "a", 1, "x"])

    # missing value in provided order
    with raises(ValueError):
        gl.sort_by([0.5, "a", 1])


def test_grouped_list_remove() -> None:
    gl = GroupedList([1, 2, 3])
    gl.remove(2)
    assert gl == [1, 3]
    assert gl.content == {1: [1], 3: [3]}

    # value not in list
    gl = GroupedList({"a": ["b"], "c": ["d"]})
    with raises(ValueError):
        gl.remove(2)
    with raises(ValueError):
        gl.remove("d")


def test_grouped_list_pop() -> None:
    gl = GroupedList([1, 2, 3])
    gl.pop(1)
    assert gl == [1, 3]
    assert gl.content == {1: [1], 3: [3]}

    # value not in list
    gl = GroupedList({"a": ["b"], "c": ["d"]})
    with raises(IndexError):
        gl.pop(4)
    with raises(TypeError):
        gl.pop("d")


def test_grouped_list_get_group() -> None:
    gl = GroupedList({"a": ["b", "c"]})
    assert gl.get_group("b") == "a"
    assert gl.get_group("z") == "z"


def test_grouped_list_get_values() -> None:
    gl = GroupedList({"a": ["b", "c"], "d": ["e"]})
    assert gl.values == ["b", "c", "a", "e", "d"]


def test_grouped_list_contains() -> None:
    gl = GroupedList({"a": ["b", "c"]})
    assert gl.contains("a")
    assert gl.contains("b")
    assert not gl.contains("z")


def test_grouped_list_replace_group_leader() -> None:
    gl = GroupedList({"a": ["b", "c"], "f": []})
    gl.replace_group_leader("a", "b")
    assert gl == ["b", "f"]
    assert gl.content == {"b": ["b", "c", "a"], "f": ["f"]}

    with raises(ValueError):  # group_member not in group of group_leader
        gl.replace_group_leader("b", "z")

    with raises(ValueError):  # group_member not in group of group_leader
        gl.replace_group_leader("b", "f")


def test_grouped_list_sanity_check():
    gl = GroupedList({"a": ["b"], "c": ["c", "d"]})
    gl.sanity_check()  # should not raise any error

    # missing group leader
    gl.content = {"a": ["b"], "c": ["c", "d"]}
    with raises(ValueError):
        gl.sanity_check()

    # value in several groups
    gl.content = {"a": ["a", "b"], "c": ["b", "c"]}
    with raises(ValueError):
        gl.sanity_check()

    # key missing from content
    gl.content = {"c": ["b", "c"]}
    with raises(ValueError):
        gl.sanity_check()

    # inconsistent ordering
    gl.content = {"c": ["c", "d"], "a": ["a", "b"]}
    with raises(ValueError):
        gl.sanity_check()
