""" set of tests for BaseFeature"""

from pytest import raises
from pandas import DataFrame
from AutoCarver.config import DEFAULT, NAN
from AutoCarver.features import BaseFeature, GroupedList

# removing abstractmethods for tests
BaseFeature.__abstractmethods__ = set()


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


def test_base_feature_content() -> None:
    """test property content"""
    feature = BaseFeature(name="test_feature")
    feature.values = GroupedList(["a", "d"])

    assert feature.content == {"a": ["a"], "d": ["d"]}


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


def test_base_feature_make_labels() -> None:
    """test method make_labels"""
    feature = BaseFeature(name="test_feature")

    # testing sorting values
    feature.values = GroupedList({"a": ["a", "b"], "c": ["c", "d"]})
    values = feature.make_labels()
    assert values == ["a", "c"]
    assert values.content == {"a": ["a", "b"], "c": ["c", "d"]}


def test_base_feature_update_labels_ordinal_encoding() -> None:
    """test method update_labels and ordinal_encoding"""
    feature = BaseFeature(name="test_feature")

    # testing without ordinal encoding
    feature.values = GroupedList({"a": ["a", "b"], "c": ["c", "d"]})
    feature.update_labels()
    assert feature.labels == ["a", "c"]
    with raises(AttributeError):
        _ = feature.labels.content
    assert feature.label_per_value == {"a": "a", "b": "a", "c": "c", "d": "c"}
    assert feature.value_per_label == {"a": "a", "c": "c"}

    # testing with ordinal encoding existing labels
    feature.ordinal_encoding = True  # automatically updates labels
    assert feature.labels == [0, 1]
    assert feature.label_per_value == {"a": 0, "b": 0, "c": 1, "d": 1}
    assert feature.value_per_label == {0: "a", 1: "c"}

    # testing turning off ordinal encoding existing labels
    feature.ordinal_encoding = False  # automatically updates labels
    assert feature.labels == ["a", "c"]
    assert feature.label_per_value == {"a": "a", "b": "a", "c": "c", "d": "c"}
    assert feature.value_per_label == {"a": "a", "c": "c"}

    # testing with ordinal encoding non-existing labels
    feature = BaseFeature(name="test_feature")
    feature.values = GroupedList({"a": ["a", "b"], "c": ["c", "d"]})
    feature.ordinal_encoding = True  # automatically updates labels
    assert feature.labels == [0, 1]
    assert feature.label_per_value == {"a": 0, "b": 0, "c": 1, "d": 1}
    assert feature.value_per_label == {0: "a", 1: "c"}


def tset_base_feature_update() -> None:
    """test method update"""
    feature = BaseFeature(name="test_feature")

    # testing sorting values
    feature.values = GroupedList({"a": ["a", "b"], "c": ["c", "d"]})
    feature.update(["c", "a"], sorted_values=True)
    assert feature.values == ["c", "a"]
    assert feature.content == {"c": ["c", "d"], "a": ["a", "b"]}

    # input should be a groupedlist
    with raises(ValueError):
        feature.update(["wrong", "input"])
    with raises(ValueError):
        feature.update(["wrong", "input"], convert_labels=True)

    # testing replace values
    feature.update(GroupedList({1: ["a", "b"], 2: ["c", "d"]}), replace=True)
    assert feature.values == [1, 2]
    assert feature.content == {1: ["a", "b", 1], 2: ["c", "d", 2]}


def test_base_feature_has_default() -> None:
    """test property has_default"""
    feature = BaseFeature(name="test_feature")
    feature.update(GroupedList(["a", "b"]), replace=True)
    feature.default = "default"

    # not setting default
    feature.has_default = False
    assert not feature.has_default
    assert "default" not in feature.values
    assert feature.content == {"a": ["a"], "b": ["b"]}
    assert feature.labels == ["a", "b"]

    # setting default
    feature.has_default = True
    assert feature.has_default
    assert "default" in feature.values
    assert feature.content == {"a": ["a"], "b": ["b"], "default": ["default"]}
    assert feature.labels == ["a", "b", "default"]

    # resetting set_has_default
    with raises(RuntimeError):
        feature.has_default = False


def test_base_feature_set_dropna() -> None:
    """test method set_dropna"""

    # activate feature does not have nan
    feature = BaseFeature(name="test_feature")
    feature.update(GroupedList(["a", "b"]), replace=True)
    feature.nan = "nan_value"
    feature.has_nan = False
    feature.dropna = True
    assert feature.dropna
    assert "nan_value" not in feature.values
    assert feature.content == {"a": ["a"], "b": ["b"]}
    assert feature.labels == ["a", "b"]

    # activate feature has nan
    feature = BaseFeature(name="test_feature")
    feature.update(GroupedList(["a", "b"]), replace=True)
    feature.nan = "nan_value"
    feature.has_nan = True
    feature.dropna = True
    assert feature.dropna
    assert "nan_value" in feature.values
    assert feature.content == {"a": ["a"], "b": ["b"], "nan_value": ["nan_value"]}
    assert feature.labels == ["a", "b", "nan_value"]

    # deactivate feature already merged nans
    feature = BaseFeature(name="test_feature")
    feature.update(
        GroupedList({"a": ["a"], "b": ["b"], "nan_value": ["nan_value", "c"]}), replace=True
    )
    feature.nan = "nan_value"
    feature.has_nan = True
    feature.dropna = True  # Activate dropna
    assert "nan_value" in feature.values
    assert feature.content == {"a": ["a"], "b": ["b"], "nan_value": ["nan_value", "c"]}
    with raises(RuntimeError):
        feature.dropna = False  # Attempt to deactivate dropna with merged nans

    # deactivate feature not merged nans
    feature = BaseFeature(name="test_feature")
    feature.update(GroupedList({"a": ["a"], "b": ["b"], "nan_value": ["nan_value"]}), replace=True)
    feature.nan = "nan_value"
    feature.has_nan = True
    feature.dropna = True  # Activate dropna
    assert feature.content == {"a": ["a"], "b": ["b"], "nan_value": ["nan_value"]}
    feature.dropna = False  # Deactivate dropna without merged nans
    assert feature.content == {"a": ["a"], "b": ["b"]}
    assert not feature.dropna

    # activate feature does not have nan
    feature = BaseFeature(name="test_feature")
    feature.update(GroupedList(["a", "b"]), replace=True)
    feature.nan = "nan_value"
    feature.has_nan = False
    feature.dropna = True
    feature.ordinal_encoding = True
    assert feature.dropna
    assert "nan_value" not in feature.values
    assert feature.content == {"a": ["a"], "b": ["b"]}
    assert feature.labels == [0, 1]

    # activate feature has nan
    feature = BaseFeature(name="test_feature")
    feature.update(GroupedList(["a", "b"]), replace=True)
    feature.nan = "nan_value"
    feature.has_nan = True
    feature.dropna = True
    feature.ordinal_encoding = True
    assert feature.dropna
    assert "nan_value" in feature.values
    assert feature.content == {"a": ["a"], "b": ["b"], "nan_value": ["nan_value"]}
    assert feature.labels == [0, 1, 2]


def test_base_feature_group_list() -> None:
    """test method group_list"""
    feature = BaseFeature(name="test_feature")
    feature.values = GroupedList(["a", "b", "c", "d"])
    feature.group_list(["b", "c"], "a")

    assert feature.values == ["a", "d"]
    assert feature.content == {"a": ["c", "b", "a"], "d": ["d"]}


def test_base_feature_to_json() -> None:
    """test method to_json"""

    # ordinal encoding false
    feature = BaseFeature(name="test_feature")
    feature.values = GroupedList(["a", "b", "c"])
    feature.is_ordinal = True
    feature.is_categorical = True
    feature.update_labels()

    json_output = feature.to_json(light_mode=True)

    assert json_output["name"] == "test_feature"
    assert json_output["is_ordinal"]
    assert json_output["is_categorical"]
    assert json_output["values"] == feature.values
    assert not json_output["ordinal_encoding"]
    assert "statistics" not in json_output
    assert "history" not in json_output

    json_output = feature.to_json(light_mode=False)
    assert "statistics" in json_output
    assert "history" in json_output

    # ordinal encoding true
    feature = BaseFeature(name="test_feature")
    feature.values = GroupedList(["a", "b", "c"])
    feature.is_ordinal = True
    feature.is_categorical = True
    feature.ordinal_encoding = True

    json_output = feature.to_json(light_mode=True)

    assert json_output["name"] == "test_feature"
    assert json_output["is_ordinal"]
    assert json_output["is_categorical"]
    assert json_output["values"] == feature.values
    assert json_output["ordinal_encoding"]
    assert "statistics" not in json_output
    assert "history" not in json_output

    json_output = feature.to_json(light_mode=False)
    assert "statistics" in json_output
    assert "history" in json_output


def test_base_feature_load() -> None:
    """test classmethod load"""

    # ordinal encoding false
    feature = BaseFeature(name="test_feature")
    feature.values = GroupedList({"a": ["a", "b"], "c": ["c", "d"]})
    feature.is_ordinal = True
    feature.is_categorical = True
    feature.nan = "nan_value"
    feature.has_nan = True
    feature.update_labels()
    feature.statistics = {"test": "value"}
    feature.history = [
        {"combination": [["value1"], ["value2"]]},
        {"combination": [["value1"], ["value3"]]},
    ]
    feature.raw_order = [1, 2, 3]

    feature_data = feature.to_json(light_mode=False)

    loaded_feature = BaseFeature.load(feature_data)

    assert loaded_feature.name == feature.name
    assert loaded_feature.version == feature.version
    assert loaded_feature.version_tag == feature.version_tag
    assert loaded_feature.is_ordinal == feature.is_ordinal
    assert loaded_feature.is_categorical == feature.is_categorical
    assert loaded_feature.has_nan == feature.has_nan
    assert loaded_feature.nan == feature.nan
    assert loaded_feature.values == feature.values[:]
    assert loaded_feature.content == feature.content
    assert loaded_feature.labels == feature.labels
    assert loaded_feature.label_per_value == feature.label_per_value
    assert loaded_feature.value_per_label == feature.value_per_label
    assert loaded_feature.raw_order == feature.raw_order
    assert loaded_feature.statistics == feature.statistics
    assert loaded_feature.history == feature.history
    assert not loaded_feature.ordinal_encoding

    # ordinal encoding true
    feature = BaseFeature(name="test_feature")
    feature.values = GroupedList({"a": ["a", "b"], "c": ["c", "d"]})
    feature.is_ordinal = True
    feature.is_categorical = True
    feature.nan = "nan_value"
    feature.has_nan = True
    feature.ordinal_encoding = True
    feature.statistics = {"test": "value"}
    feature.history = [
        {"combination": [["value1"], ["value2"]]},
        {"combination": [["value1"], ["value3"]]},
    ]
    feature.raw_order = [1, 2, 3]

    feature_data = feature.to_json(light_mode=True)

    loaded_feature = BaseFeature.load(feature_data)

    assert loaded_feature.name == feature.name
    assert loaded_feature.version == feature.version
    assert loaded_feature.version_tag == feature.version_tag
    assert loaded_feature.is_ordinal == feature.is_ordinal
    assert loaded_feature.is_categorical == feature.is_categorical
    assert loaded_feature.has_nan == feature.has_nan
    assert loaded_feature.nan == feature.nan
    assert loaded_feature.values == feature.values[:]
    assert loaded_feature.content == feature.content
    assert loaded_feature.labels == feature.labels
    assert loaded_feature.label_per_value == feature.label_per_value
    assert loaded_feature.value_per_label == feature.value_per_label
    assert loaded_feature.raw_order == feature.raw_order
    assert len(loaded_feature.statistics) == 0
    assert len(loaded_feature.history) == 0
    assert loaded_feature.ordinal_encoding
