""" set of tests for BaseFeature"""

from pytest import raises
from pandas import DataFrame
from AutoCarver.config import DEFAULT, NAN
from AutoCarver.features import BaseFeature, GroupedList


def test_base_feature_initialization() -> None:
    """test method __init__"""
    feature = BaseFeature(name="test_feature")

    assert feature.name == "test_feature"
    assert not feature.has_nan
    assert feature.nan == NAN
    assert not feature.has_default
    assert feature.default == DEFAULT
    assert not feature.dropna
    assert not feature.is_fitted
    assert feature.values is None
    assert feature.labels is None
    assert isinstance(feature.label_per_value, dict) and len(feature.label_per_value) == 0
    assert isinstance(feature.value_per_label, dict) and len(feature.value_per_label) == 0
    assert not feature.is_ordinal
    assert not feature.is_categorical
    assert not feature.is_qualitative
    assert not feature.is_quantitative
    assert isinstance(feature.statistics, dict) and len(feature.statistics) == 0
    assert isinstance(feature.history, list) and len(feature.history) == 0
    assert isinstance(feature.raw_order, list) and len(feature.raw_order) == 0
    assert feature.version == "test_feature"
    assert feature.version_tag == "test_feature"


def test_base_feature_fit() -> None:
    """test method fit"""
    # test fitting
    data = DataFrame({"test_feature": [1, 2, 3]})
    feature = BaseFeature(name="test_feature")
    feature.fit(data)
    assert feature.is_fitted
    assert not feature.has_nan

    # test fitting twice
    with raises(RuntimeError):
        feature.fit(data)

    # test nan
    data = DataFrame({"test_feature": [1, 2, 3, None]})
    feature = BaseFeature(name="test_feature")
    feature.fit(data)
    assert feature.is_fitted
    assert feature.has_nan


def test_base_feature_check_values() -> None:
    """test method check_values"""

    data = DataFrame({"test_feature": [1, 2, 3, None]})
    feature = BaseFeature(name="test_feature")
    feature.fit(data)
    feature.check_values(data)  # Should not raise any error

    data = DataFrame({"test_feature": [1, 2, 3]})
    feature = BaseFeature(name="test_feature")
    feature.fit(data)
    feature.check_values(data)  # Should not raise any error

    # test nan
    data = DataFrame({"test_feature": [1, 2, 3, None]})
    with raises(ValueError):
        feature.check_values(data)


def test_base_feature_update_values() -> None:
    """test method update_values"""
    feature = BaseFeature(name="test_feature")

    # testing sorting values
    feature.values = GroupedList({"a": ["a", "b"], "c": ["c", "d"]})
    feature.update(["c", "a"], sorted_values=True)
    assert feature.values == ["c", "a"]
    assert feature.values.content == {"c": ["c", "d"], "a": ["a", "b"]}

    # input should be a groupedlist
    with raises(ValueError):
        feature.update(["wrong", "input"])
    with raises(ValueError):
        feature.update(["wrong", "input"], convert_labels=True)

    # testing replace values
    feature.update(GroupedList({1: ["a", "b"], 2: ["c", "d"]}), replace=True)
    assert feature.values == [1, 2]
    assert feature.values.content == {1: ["a", "b", 1], 2: ["c", "d", 2]}

    # testing convert_labels=False and feature.values=None
    feature = BaseFeature(name="test_feature")
    feature.update(GroupedList({"a": ["a", "b"], "c": ["c", "d"]}), convert_labels=False)
    assert feature.values == ["a", "c"]
    assert feature.values.content == {"a": ["a", "b"], "c": ["c", "d"]}

    # testing convert_labels=False and feature.values not None
    with raises(ValueError):
        feature.update(GroupedList({"e": ["f", "g"], "a": ["a", "b"]}), convert_labels=False)
    feature.update(GroupedList({"a": ["a", "c"]}), convert_labels=False)
    assert feature.values == ["a"]
    assert feature.values.content == {"a": ["c", "d", "a", "b"]}

    feature = BaseFeature(name="test_feature")
    with raises(AttributeError):
        feature.update(GroupedList({"a": ["a", "b"], "c": ["c", "d"]}), convert_labels=True)

    # TODO
    # TODO
    # TODO
    # TODO
    # updated_values = GroupedList({"a": ["a", "b"], "c": ["c", "d", "e"]})

    # feature.update(initial_values)
    # assert feature.values == initial_values

    # feature.update(updated_values, replace=True)
    # assert feature.values == updated_values


def test_base_feature_set_has_default() -> None:
    """test method set_has_default"""
    feature = BaseFeature(name="test_feature")
    feature.values = GroupedList(["a", "b"])
    feature.default = "default"

    # not setting default
    feature.set_has_default(has_default=False)
    assert not feature.has_default
    assert "default" not in feature.values
    assert feature.values.content == {"a": ["a"], "b": ["b"]}

    # setting default
    feature.set_has_default(has_default=True)
    assert feature.has_default
    assert "default" in feature.values
    assert feature.values.content == {"a": ["a"], "b": ["b"], "default": ["default"]}


def test_base_feature_set_dropna() -> None:
    """test method set_dropna"""

    # activate feature does not have nan
    feature = BaseFeature(name="test_feature")
    feature.values = GroupedList(["a", "b"])
    feature.nan = "nan_value"
    feature.set_dropna(dropna=True)
    assert feature.dropna
    assert "nan_value" not in feature.values
    assert feature.values.content == {"a": ["a"], "b": ["b"]}

    # activate feature has nan
    feature = BaseFeature(name="test_feature")
    feature.values = GroupedList(["a", "b"])
    feature.nan = "nan_value"
    feature.has_nan = True
    feature.set_dropna(dropna=True)
    assert feature.dropna
    assert "nan_value" in feature.values
    assert feature.values.content == {"a": ["a"], "b": ["b"], "nan_value": ["nan_value"]}

    # deactivate feature already merged nans
    feature = BaseFeature(name="test_feature")
    feature.values = GroupedList({"a": ["a"], "b": ["b"], "nan_value": ["nan_value", "c"]})
    feature.nan = "nan_value"
    feature.has_nan = True
    feature.set_dropna(dropna=True)  # Activate dropna
    assert "nan_value" in feature.values
    assert feature.values.content == {"a": ["a"], "b": ["b"], "nan_value": ["nan_value", "c"]}

    with raises(RuntimeError):
        feature.set_dropna(dropna=False)  # Attempt to deactivate dropna with merged nans

    # deactivate feature not merged nans
    feature = BaseFeature(name="test_feature")
    feature.values = GroupedList({"a": ["a"], "b": ["b"]})
    feature.nan = "nan_value"
    feature.has_nan = True
    feature.set_dropna(dropna=True)  # Activate dropna
    assert feature.values.content == {"a": ["a"], "b": ["b"], "nan_value": ["nan_value"]}
    feature.set_dropna(dropna=False)  # Deactivate dropna without merged nans
    assert feature.values.content == {"a": ["a"], "b": ["b"]}
    assert not feature.dropna


# def test_base_feature_to_json() -> None:
#     feature = BaseFeature(name="test_feature")
#     feature.values = GroupedList(["a", "b", "c"])
#     feature.is_ordinal = True
#     feature.is_categorical = True

#     json_output = feature.to_json()

#     assert json_output["name"] == "test_feature"
#     assert json_output["is_ordinal"]
#     assert json_output["is_categorical"]
#     assert json_output["values"] == feature.values


# def test_base_feature_load() -> None:
#     feature_data = {
#         "name": "test_feature",
#         "version": "1.0",
#         "has_nan": True,
#         "nan": "nan_value",
#         "has_default": True,
#         "default": "default_value",
#         "dropna": True,
#         "is_ordinal": True,
#         "is_categorical": True,
#         "values": {"a": ["a", "b"], "c": ["c", "d"]},
#     }

#     loaded_feature = BaseFeature.load(feature_data, ordinal_encoding=True)

#     assert loaded_feature.name == "test_feature"
#     assert loaded_feature.is_ordinal
#     assert loaded_feature.is_categorical
#     assert loaded_feature.has_nan
#     assert loaded_feature.nan == "nan_value"
#     assert loaded_feature.values == GroupedList({"a": ["a", "b"], "c": ["c", "d"]})
