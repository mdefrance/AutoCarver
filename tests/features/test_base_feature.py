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


def test_base_feature_get_content() -> None:
    """test method get_content"""
    feature = BaseFeature(name="test_feature")
    feature.values = GroupedList(["a", "d"])

    assert feature.get_content() == {"a": ["a"], "d": ["d"]}


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


def test_base_feature_get_labels() -> None:
    """test method get_labels"""
    feature = BaseFeature(name="test_feature")

    # testing sorting values
    feature.values = GroupedList({"a": ["a", "b"], "c": ["c", "d"]})
    values = feature.get_labels()
    assert values == ["a", "c"]
    assert values.content == {"a": ["a", "b"], "c": ["c", "d"]}


def test_base_feature_update_labels() -> None:
    """test method update_labels"""
    feature = BaseFeature(name="test_feature")

    # testing without ordinal encoding
    feature.values = GroupedList({"a": ["a", "b"], "c": ["c", "d"]})
    feature.update_labels(ordinal_encoding=False)
    assert feature.labels == ["a", "c"]
    with raises(AttributeError):
        feature.labels.content
    assert feature.label_per_value == {"a": "a", "b": "a", "c": "c", "d": "c"}
    assert feature.value_per_label == {"a": "a", "c": "c"}

    # testing with ordinal encoding existing labels
    feature.update_labels(ordinal_encoding=True)
    assert feature.labels == [0, 1]
    assert feature.label_per_value == {"a": 0, "b": 0, "c": 1, "d": 1}
    assert feature.value_per_label == {0: "a", 1: "c"}

    # testing with ordinal encoding non-existing labels
    feature = BaseFeature(name="test_feature")
    feature.values = GroupedList({"a": ["a", "b"], "c": ["c", "d"]})
    feature.update_labels(ordinal_encoding=True)
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
    assert feature.get_content() == {"c": ["c", "d"], "a": ["a", "b"]}

    # input should be a groupedlist
    with raises(ValueError):
        feature.update(["wrong", "input"])
    with raises(ValueError):
        feature.update(["wrong", "input"], convert_labels=True)

    # testing replace values
    feature.update(GroupedList({1: ["a", "b"], 2: ["c", "d"]}), replace=True)
    assert feature.values == [1, 2]
    assert feature.get_content() == {1: ["a", "b", 1], 2: ["c", "d", 2]}

    # testing convert_labels=False and feature.values=None
    feature = BaseFeature(name="test_feature")
    feature.update(GroupedList({"a": ["a", "b"], "c": ["c", "d"]}), convert_labels=False)
    assert feature.values == ["a", "c"]
    assert feature.get_content() == {"a": ["a", "b"], "c": ["c", "d"]}

    # testing convert_labels=False and feature.values not None
    with raises(ValueError):
        feature.update(GroupedList({"e": ["f", "g"], "a": ["a", "b"]}), convert_labels=False)
    feature.update(GroupedList({"a": ["a", "c"]}), convert_labels=False)
    assert feature.values == ["a"]
    assert feature.get_content() == {"a": ["c", "d", "a", "b"]}


def test_base_feature_update_qualitative() -> None:
    """test method update convert_labels for qualitative features"""

    # testing convert_labels=True and feature.values=None
    feature = BaseFeature(name="test_feature")
    with raises(AttributeError):  # only for already valued features
        feature.update(GroupedList({"a": ["a", "b"], "c": ["c", "d"]}), convert_labels=True)

    # testing convert_labels=True and feature.values not None without ordinal_encoding
    feature = BaseFeature(name="test_feature")
    ordinal_encoding = False
    feature.values = GroupedList({"a": ["a", "b"], "c": ["c", "d"], "e": ["e", "f", "g"]})
    feature.update_labels(ordinal_encoding=ordinal_encoding)  # setting up labels
    feature.is_quantitative = False
    feature.update(
        GroupedList({"a": ["a", "c", "b"]}),
        convert_labels=True,
        ordinal_encoding=ordinal_encoding,
    )
    assert feature.values == ["a", "e"]
    assert feature.get_content() == {"a": ["c", "d", "a", "b"], "e": ["e", "f", "g"]}
    assert feature.labels == ["a", "e"]
    assert feature.label_per_value == {
        "a": "a",
        "b": "a",
        "c": "a",
        "d": "a",
        "e": "e",
        "f": "e",
        "g": "e",
    }
    assert feature.value_per_label == {"a": "a", "e": "e"}
    # test updating nothing
    feature.update(
        GroupedList({"a": ["a", "c", "b"]}),
        convert_labels=True,
        ordinal_encoding=ordinal_encoding,
    )
    assert feature.values == ["a", "e"]
    assert feature.get_content() == {"a": ["c", "d", "a", "b"], "e": ["e", "f", "g"]}
    assert feature.labels == ["a", "e"]
    assert feature.label_per_value == {
        "a": "a",
        "b": "a",
        "c": "a",
        "d": "a",
        "e": "e",
        "f": "e",
        "g": "e",
    }
    assert feature.value_per_label == {"a": "a", "e": "e"}

    # testing convert_labels=True and feature.values not None with ordinal_encoding
    feature = BaseFeature(name="test_feature")
    ordinal_encoding = True
    feature.values = GroupedList({"a": ["a", "b"], "c": ["c", "d"], "e": ["e", "f", "g"]})
    feature.update_labels(ordinal_encoding=ordinal_encoding)  # setting up labels
    feature.is_quantitative = False
    feature.update(
        GroupedList({"a": ["a", "c", "b"]}),
        convert_labels=True,
        ordinal_encoding=ordinal_encoding,
    )
    assert feature.values == ["a", "e"]
    assert feature.get_content() == {"a": ["c", "d", "a", "b"], "e": ["e", "f", "g"]}
    assert feature.labels == [0, 1]
    assert feature.label_per_value == {
        "a": 0,
        "b": 0,
        "c": 0,
        "d": 0,
        "e": 1,
        "f": 1,
        "g": 1,
    }
    assert feature.value_per_label == {0: "a", 1: "e"}


def test_base_feature_update_quantitative() -> None:
    """test method update convert_labels for quantitative features"""

    # testing convert_labels=True and feature.values=None
    feature = BaseFeature(name="test_feature")
    with raises(AttributeError):  # only for already valued features
        feature.update(GroupedList({0: [0], 1: [1], 5: [5]}), convert_labels=True)

    # testing convert_labels=True and feature.values not None without ordinal_encoding
    feature = BaseFeature(name="test_feature")
    ordinal_encoding = True
    feature.values = GroupedList({0: [0], 1: [1], 5: [5]})
    # setting up labels
    feature.labels = ["0<x", "0<x<1", "1<x<5"]
    feature.label_per_value = {0: "0<x", 1: "0<x<1", 5: "1<x<5"}
    feature.value_per_label = {"0<x": 0, "0<x<1": 1, "1<x<5": 5}

    feature.is_quantitative = True
    feature.update(
        GroupedList({"0<x": ["0<x", "0<x<1"]}),
        convert_labels=True,
        ordinal_encoding=ordinal_encoding,
    )
    assert feature.values == [1, 5]
    assert feature.get_content() == {1: [0, 1], 5: [5]}
    # setting up labels
    feature.labels = ["0<x<1", "1<x<5"]
    feature.label_per_value = {0: "0<x<1", 1: "0<x<1", 5: "1<x<5"}
    feature.value_per_label = {"0<x<1": 1, "1<x<5": 5}

    # test updating nothing
    feature.update(
        GroupedList({"0<x<1": ["0<x<1"]}),
        convert_labels=True,
        ordinal_encoding=ordinal_encoding,
    )
    assert feature.values == [1, 5]
    assert feature.get_content() == {1: [0, 1], 5: [5]}

    # testing convert_labels=True and feature.values not None with ordinal_encoding
    feature = BaseFeature(name="test_feature")
    ordinal_encoding = True
    feature.values = GroupedList({0: [0], 1: [1], 5: [5]})
    # setting up labels
    feature.labels = ["0<x", "0<x<1", "1<x<5"]
    feature.label_per_value = {0: "0<x", 1: "0<x<1", 5: "1<x<5"}
    feature.value_per_label = {"0<x": 0, "0<x<1": 1, "1<x<5": 5}

    feature.is_quantitative = True
    feature.update(
        GroupedList({"0<x": ["0<x", "0<x<1"]}),
        convert_labels=True,
        ordinal_encoding=ordinal_encoding,
    )
    assert feature.values == [1, 5]
    assert feature.get_content() == {1: [0, 1], 5: [5]}
    assert feature.labels == [0, 1]
    assert feature.label_per_value == {0: 0, 1: 0, 5: 1}
    assert feature.value_per_label == {0: 1, 1: 5}


def test_base_feature_set_has_default() -> None:
    """test method set_has_default"""
    feature = BaseFeature(name="test_feature")
    feature.values = GroupedList(["a", "b"])
    feature.default = "default"

    # not setting default
    feature.set_has_default(has_default=False)
    assert not feature.has_default
    assert "default" not in feature.values
    assert feature.get_content() == {"a": ["a"], "b": ["b"]}

    # setting default
    feature.set_has_default(has_default=True)
    assert feature.has_default
    assert "default" in feature.values
    assert feature.get_content() == {"a": ["a"], "b": ["b"], "default": ["default"]}


def test_base_feature_set_dropna() -> None:
    """test method set_dropna"""

    # activate feature does not have nan
    feature = BaseFeature(name="test_feature")
    feature.values = GroupedList(["a", "b"])
    feature.nan = "nan_value"
    feature.set_dropna(dropna=True)
    assert feature.dropna
    assert "nan_value" not in feature.values
    assert feature.get_content() == {"a": ["a"], "b": ["b"]}

    # activate feature has nan
    feature = BaseFeature(name="test_feature")
    feature.values = GroupedList(["a", "b"])
    feature.nan = "nan_value"
    feature.has_nan = True
    feature.set_dropna(dropna=True)
    assert feature.dropna
    assert "nan_value" in feature.values
    assert feature.get_content() == {"a": ["a"], "b": ["b"], "nan_value": ["nan_value"]}

    # deactivate feature already merged nans
    feature = BaseFeature(name="test_feature")
    feature.values = GroupedList({"a": ["a"], "b": ["b"], "nan_value": ["nan_value", "c"]})
    feature.nan = "nan_value"
    feature.has_nan = True
    feature.set_dropna(dropna=True)  # Activate dropna
    assert "nan_value" in feature.values
    assert feature.get_content() == {"a": ["a"], "b": ["b"], "nan_value": ["nan_value", "c"]}

    with raises(RuntimeError):
        feature.set_dropna(dropna=False)  # Attempt to deactivate dropna with merged nans

    # deactivate feature not merged nans
    feature = BaseFeature(name="test_feature")
    feature.values = GroupedList({"a": ["a"], "b": ["b"]})
    feature.nan = "nan_value"
    feature.has_nan = True
    feature.set_dropna(dropna=True)  # Activate dropna
    assert feature.get_content() == {"a": ["a"], "b": ["b"], "nan_value": ["nan_value"]}
    feature.set_dropna(dropna=False)  # Deactivate dropna without merged nans
    assert feature.get_content() == {"a": ["a"], "b": ["b"]}
    assert not feature.dropna


def test_base_feature_group_list() -> None:
    """test method group_list"""
    feature = BaseFeature(name="test_feature")
    feature.values = GroupedList(["a", "b", "c", "d"])
    feature.group_list(["b", "c"], "a")

    assert feature.values == ["a", "d"]
    assert feature.get_content() == {"a": ["c", "b", "a"], "d": ["d"]}


def test_base_feature_to_json() -> None:
    """test method to_json"""

    # ordinal encoding false
    feature = BaseFeature(name="test_feature")
    feature.values = GroupedList(["a", "b", "c"])
    feature.is_ordinal = True
    feature.is_categorical = True
    ordinal_encoding = False
    feature.update_labels(ordinal_encoding=ordinal_encoding)

    json_output = feature.to_json(light_mode=True)

    assert json_output["name"] == "test_feature"
    assert json_output["is_ordinal"]
    assert json_output["is_categorical"]
    assert json_output["values"] == feature.values
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
    ordinal_encoding = True
    feature.update_labels(ordinal_encoding=ordinal_encoding)

    json_output = feature.to_json(light_mode=True)

    assert json_output["name"] == "test_feature"
    assert json_output["is_ordinal"]
    assert json_output["is_categorical"]
    assert json_output["values"] == feature.values
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
    ordinal_encoding = False
    feature.update_labels(ordinal_encoding=ordinal_encoding)
    feature.statistics = {"test": "value"}
    feature.history = [
        {"combination": [["value1"], ["value2"]]},
        {"combination": [["value1"], ["value3"]]},
    ]
    feature.raw_order = [1, 2, 3]

    feature_data = feature.to_json(light_mode=False)

    loaded_feature = BaseFeature.load(feature_data, ordinal_encoding=ordinal_encoding)

    assert loaded_feature.name == feature.name
    assert loaded_feature.version == feature.version
    assert loaded_feature.version_tag == feature.version_tag
    assert loaded_feature.is_ordinal == feature.is_ordinal
    assert loaded_feature.is_categorical == feature.is_categorical
    assert loaded_feature.has_nan == feature.has_nan
    assert loaded_feature.nan == feature.nan
    assert loaded_feature.values == feature.values[:]
    assert loaded_feature.get_content() == feature.get_content()
    assert loaded_feature.labels == feature.labels
    assert loaded_feature.label_per_value == feature.label_per_value
    assert loaded_feature.value_per_label == feature.value_per_label
    assert loaded_feature.raw_order == feature.raw_order
    assert loaded_feature.statistics == feature.statistics
    assert loaded_feature.history == feature.history


    # ordinal encoding true
    feature = BaseFeature(name="test_feature")
    feature.values = GroupedList({"a": ["a", "b"], "c": ["c", "d"]})
    feature.is_ordinal = True
    feature.is_categorical = True
    feature.nan = "nan_value"
    feature.has_nan = True
    ordinal_encoding = True
    feature.update_labels(ordinal_encoding=ordinal_encoding)
    feature.statistics = {"test": "value"}
    feature.history = [
        {"combination": [["value1"], ["value2"]]},
        {"combination": [["value1"], ["value3"]]},
    ]
    feature.raw_order = [1, 2, 3]

    feature_data = feature.to_json(light_mode=True)

    loaded_feature = BaseFeature.load(feature_data, ordinal_encoding=ordinal_encoding)

    assert loaded_feature.name == feature.name
    assert loaded_feature.version == feature.version
    assert loaded_feature.version_tag == feature.version_tag
    assert loaded_feature.is_ordinal == feature.is_ordinal
    assert loaded_feature.is_categorical == feature.is_categorical
    assert loaded_feature.has_nan == feature.has_nan
    assert loaded_feature.nan == feature.nan
    assert loaded_feature.values == feature.values[:]
    assert loaded_feature.get_content() == feature.get_content()
    assert loaded_feature.labels == feature.labels
    assert loaded_feature.label_per_value == feature.label_per_value
    assert loaded_feature.value_per_label == feature.value_per_label
    assert loaded_feature.raw_order == feature.raw_order
    assert len(loaded_feature.statistics) == 0
    assert len(loaded_feature.history) == 0
