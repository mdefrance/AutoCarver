""" set of tests for attribute.py """

from pytest import raises

from AutoCarver.utils import get_bool_attribute


def test_get_bool_attribute() -> None:
    """test get_bool_attribute checks"""

    kwargs = {"true": True, "false": False, "wrong": "value"}
    with raises(ValueError):
        get_bool_attribute(kwargs, "wrong", True)
    assert get_bool_attribute(kwargs, "true", True)
    assert get_bool_attribute(kwargs, "true", False)
    assert not get_bool_attribute(kwargs, "false", True)
    assert not get_bool_attribute(kwargs, "false", False)
    assert get_bool_attribute(kwargs, "missing", True)
    assert not get_bool_attribute(kwargs, "missing", False)
