"""set of tests for quantitative features"""

from numpy import nan
from pandas import DataFrame
from pytest import fixture, raises

from AutoCarver.config import Constants
from AutoCarver.features.qualitatives import OrdinalFeature
from AutoCarver.features.utils.base_feature import BaseFeature
from AutoCarver.features.utils.grouped_list import GroupedList

BaseFeature.__abstractmethods__ = set()


@fixture
def sample_ordinal_feature() -> OrdinalFeature:
    """Create a sample OrdinalFeature for testing"""

    feature = OrdinalFeature("test_feature", values=["1", "2", "3", "4", "5", "a", "b", "c", "d", "e", "f"])
    feature.update(
        GroupedList({"a": ["1", "2", "3", "4", "5", "a"], "b": ["b"], "c": ["c"], "d": ["e", "f"]}),
        replace=True,
    )
    return feature


def test_ordinal_feature_type(sample_ordinal_feature: OrdinalFeature) -> None:
    """testing type"""

    assert not sample_ordinal_feature.is_quantitative
    assert sample_ordinal_feature.is_qualitative
    assert sample_ordinal_feature.is_ordinal
    assert not sample_ordinal_feature.is_categorical


def test_ordinal_feature_format_modalities(sample_ordinal_feature: OrdinalFeature) -> None:
    """testing _format_modalities"""

    # with large max_n_chars
    group = "b"
    content = sample_ordinal_feature.values.get(group)
    result = sample_ordinal_feature._format_modalities(group, content)
    assert result == "b"

    # with large max_n_chars
    group = "d"
    content = sample_ordinal_feature.values.get(group)
    result = sample_ordinal_feature._format_modalities(group, content)
    assert result == "d to f"

    # with large max_n_chars
    group = "a"
    content = sample_ordinal_feature.values.get(group)
    result = sample_ordinal_feature._format_modalities(group, content)
    assert result == "1 to a"

    # with smaller max_n_chars
    sample_ordinal_feature.max_n_chars = 4
    result = sample_ordinal_feature._format_modalities(group, content)
    assert result == "1 to a"

    # adding nans
    # with smaller max_n_chars
    sample_ordinal_feature.max_n_chars = 4
    result = sample_ordinal_feature._format_modalities(group, content + [sample_ordinal_feature.nan])
    assert result == f"1 to a, {sample_ordinal_feature.nan}"

    # with smaller max_n_chars
    sample_ordinal_feature.max_n_chars = 30
    result = sample_ordinal_feature._format_modalities(group, content + [sample_ordinal_feature.nan])
    assert result == f"1 to a, {sample_ordinal_feature.nan}"

    # empty content
    result = sample_ordinal_feature._format_modalities(group, [])
    assert result == group

    # with default
    sample_ordinal_feature.has_default = True
    sample_ordinal_feature.max_n_chars = 30
    result = sample_ordinal_feature._format_modalities(group, content)
    assert result == "1 to a"


def test_ordinal_feature_make_labels(sample_ordinal_feature: OrdinalFeature) -> None:
    """test make_labels method"""

    # without ordinal encoding
    result = sample_ordinal_feature.make_labels()
    assert result == ["1 to a", "b", "c", "d to f"]

    # with ordinal encoding
    sample_ordinal_feature.ordinal_encoding = True
    result = sample_ordinal_feature.make_labels()
    assert result == ["1 to a", "b", "c", "d to f"]
    assert sample_ordinal_feature.value_per_label == {0: "1 to a", 1: "b", 2: "c", 3: "d to f"}
    assert sample_ordinal_feature.label_per_value == {
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "a": 0,
        "b": 1,
        "c": 2,
        "d": 3,
        "e": 3,
        "f": 3,
    }


def test_ordinal_feature_update_ordinal_encoding(
    sample_ordinal_feature: OrdinalFeature,
) -> None:
    """testing update"""

    with raises(AttributeError):  # only already known labels
        sample_ordinal_feature.update(GroupedList({0: [0], 1: [1], "test": ["tests", "test2"]}), convert_labels=True)

    # fitting some nans
    sample_ordinal_feature.fit(DataFrame({sample_ordinal_feature.version: ["a", "c", "f", "1", nan]}))
    sample_ordinal_feature.ordinal_encoding = True
    sample_ordinal_feature.update(GroupedList({2: [2, 3]}), convert_labels=True)
    print(sample_ordinal_feature.content)
    assert sample_ordinal_feature.values == ["a", "b", "c"]
    assert sample_ordinal_feature.content == {
        "a": ["1", "2", "3", "4", "5", "a"],
        "b": ["b"],
        "c": ["e", "f", "d", "c"],
    }
    assert sample_ordinal_feature.labels == [0, 1, 2]
    assert sample_ordinal_feature.value_per_label == {
        0: "1 to a",
        1: "b",
        2: "c to f",
    }
    print(sample_ordinal_feature.label_per_value)
    assert sample_ordinal_feature.label_per_value == {
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "a": 0,
        "b": 1,
        "c": 2,
        "e": 2,
        "f": 2,
        "d": 2,
    }

    # adding nans
    sample_ordinal_feature.dropna = True
    print(sample_ordinal_feature.content)
    sample_ordinal_feature.update(GroupedList({3: [3, 1]}), convert_labels=True)
    assert sample_ordinal_feature.values == ["a", "c", Constants.NAN]
    assert sample_ordinal_feature.content == {
        "a": ["1", "2", "3", "4", "5", "a"],
        "c": ["e", "f", "d", "c"],
        Constants.NAN: ["b", Constants.NAN],
    }
    assert sample_ordinal_feature.labels == [0, 1, 2]
    assert sample_ordinal_feature.value_per_label == {
        0: "1 to a",
        1: "c to f",
        2: "b, __NAN__",
    }
    print(sample_ordinal_feature.label_per_value)
    assert sample_ordinal_feature.label_per_value == {
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "a": 0,
        "c": 1,
        "e": 1,
        "f": 1,
        "d": 1,
        Constants.NAN: 2,
        "b": 2,
    }


def test_ordinal_feature_update_no_ordinal_encoding(
    sample_ordinal_feature: OrdinalFeature,
) -> None:
    """testing update"""

    with raises(AttributeError):  # only already known labels
        sample_ordinal_feature.update(GroupedList({0: [0], 1: [1], "test": ["tests", "test2"]}), convert_labels=True)

    # fitting some nans
    sample_ordinal_feature.fit(DataFrame({sample_ordinal_feature.version: ["a", "c", "f", "1", nan]}))
    sample_ordinal_feature.ordinal_encoding = False
    sample_ordinal_feature.update(GroupedList({"d to f": ["c", "d to f"]}), convert_labels=True)
    print(sample_ordinal_feature.content)
    assert sample_ordinal_feature.values == ["a", "b", "d"]
    assert sample_ordinal_feature.content == {
        "a": ["1", "2", "3", "4", "5", "a"],
        "b": ["b"],
        "d": ["c", "e", "f", "d"],
    }
    assert sample_ordinal_feature.labels == ["1 to a", "b", "c to f"]
    print(sample_ordinal_feature.value_per_label)
    assert sample_ordinal_feature.value_per_label == {
        "1 to a": "a",
        "b": "b",
        "c to f": "d",
    }
    print(sample_ordinal_feature.label_per_value)
    assert sample_ordinal_feature.label_per_value == {
        "1": "1 to a",
        "2": "1 to a",
        "3": "1 to a",
        "4": "1 to a",
        "5": "1 to a",
        "a": "1 to a",
        "b": "b",
        "c": "c to f",
        "e": "c to f",
        "f": "c to f",
        "d": "c to f",
    }

    # adding nans
    sample_ordinal_feature.dropna = True
    print(sample_ordinal_feature.content)
    sample_ordinal_feature.update(GroupedList({Constants.NAN: [Constants.NAN, "b"]}), convert_labels=True)
    assert sample_ordinal_feature.values == ["a", "d", Constants.NAN]
    print(sample_ordinal_feature.content)
    assert sample_ordinal_feature.content == {
        "a": ["1", "2", "3", "4", "5", "a"],
        "d": ["c", "e", "f", "d"],
        Constants.NAN: ["b", Constants.NAN],
    }
    assert sample_ordinal_feature.labels == ["1 to a", "c to f", f"b, {Constants.NAN}"]
    assert sample_ordinal_feature.value_per_label == {
        "1 to a": "a",
        "c to f": "d",
        f"b, {Constants.NAN}": Constants.NAN,
    }
    print(sample_ordinal_feature.label_per_value)
    assert sample_ordinal_feature.label_per_value == {
        "1": "1 to a",
        "2": "1 to a",
        "3": "1 to a",
        "4": "1 to a",
        "5": "1 to a",
        "a": "1 to a",
        "c": "c to f",
        "e": "c to f",
        "f": "c to f",
        "d": "c to f",
        Constants.NAN: f"b, {Constants.NAN}",
        "b": f"b, {Constants.NAN}",
    }


def test_ordinal_feature_get_summary(sample_ordinal_feature: OrdinalFeature) -> None:
    """test function get_summary"""

    summary = sample_ordinal_feature.summary

    expected_summary = [
        {
            "feature": "Ordinal('test_feature')",
            "label": "1 to a",
            "content": ["1", "2", "3", "4", "5", "a"],
        },
        {"feature": "Ordinal('test_feature')", "label": "b", "content": "b"},
        {"feature": "Ordinal('test_feature')", "label": "c", "content": "c"},
        {"feature": "Ordinal('test_feature')", "label": "d to f", "content": ["e", "f", "d"]},
    ]
    assert summary == expected_summary

    sample_ordinal_feature.ordinal_encoding = True
    summary = sample_ordinal_feature.summary

    expected_summary = [
        {
            "feature": "Ordinal('test_feature')",
            "label": 0,
            "content": ["1", "2", "3", "4", "5", "a"],
        },
        {"feature": "Ordinal('test_feature')", "label": 1, "content": "b"},
        {"feature": "Ordinal('test_feature')", "label": 2, "content": "c"},
        {"feature": "Ordinal('test_feature')", "label": 3, "content": ["e", "f", "d"]},
    ]

    assert summary == expected_summary

    # fitting some nans
    sample_ordinal_feature.fit(DataFrame({sample_ordinal_feature.version: ["a", "b", nan]}))
    sample_ordinal_feature.dropna = True
    sample_ordinal_feature.ordinal_encoding = True
    summary = sample_ordinal_feature.summary

    expected_summary = [
        {
            "feature": "Ordinal('test_feature')",
            "label": 0,
            "content": ["1", "2", "3", "4", "5", "a"],
        },
        {"feature": "Ordinal('test_feature')", "label": 1, "content": "b"},
        {"feature": "Ordinal('test_feature')", "label": 2, "content": "c"},
        {"feature": "Ordinal('test_feature')", "label": 3, "content": ["e", "f", "d"]},
        {"feature": "Ordinal('test_feature')", "label": 4, "content": Constants.NAN},
    ]
    print(summary)
    assert summary == expected_summary

    # adding some default
    sample_ordinal_feature.has_default = True
    sample_ordinal_feature.ordinal_encoding = True
    summary = sample_ordinal_feature.summary

    expected_summary = [
        {
            "feature": "Ordinal('test_feature')",
            "label": 0,
            "content": ["1", "2", "3", "4", "5", "a"],
        },
        {"feature": "Ordinal('test_feature')", "label": 1, "content": "b"},
        {"feature": "Ordinal('test_feature')", "label": 2, "content": "c"},
        {"feature": "Ordinal('test_feature')", "label": 3, "content": ["e", "f", "d"]},
        {"feature": "Ordinal('test_feature')", "label": 4, "content": Constants.NAN},
        {"feature": "Ordinal('test_feature')", "label": 5, "content": Constants.DEFAULT},
    ]
    assert summary == expected_summary


def test_ordinal_feature_fit_initial_values(sample_ordinal_feature: OrdinalFeature) -> None:
    """test fit method"""

    data = DataFrame({"test_feature": ["a", "c", "f", "1"]})
    sample_ordinal_feature.fit(data)

    # Check that values have been set
    assert sample_ordinal_feature.values == ["a", "b", "c", "d"]

    # Check that raw_order has not been modified
    assert sample_ordinal_feature.raw_order == [
        "1",
        "2",
        "3",
        "4",
        "5",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
    ]


def test_ordinal_feature_fit_with_nan(sample_ordinal_feature: OrdinalFeature) -> None:
    """test fit method"""

    data = DataFrame({"test_feature": ["a", "c", "f", "1", nan]})

    sample_ordinal_feature.fit(data)

    # Check that values have been set
    assert sample_ordinal_feature.values == ["a", "b", "c", "d"]

    # Check that raw_order has not been modified
    assert sample_ordinal_feature.raw_order == [
        "1",
        "2",
        "3",
        "4",
        "5",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
    ]
