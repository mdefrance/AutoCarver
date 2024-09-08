""" set of tests for quantitative features"""

from numpy import inf, nan
from pandas import DataFrame, Series
from pytest import fixture, raises

from AutoCarver.config import DEFAULT, NAN
from AutoCarver.features.qualitative_features import (
    CategoricalFeature,
    OrdinalFeature,
    get_categorical_features,
    get_ordinal_features,
    get_qualitative_features,
    nan_unique,
)
from AutoCarver.features.utils.base_feature import BaseFeature
from AutoCarver.features.utils.grouped_list import GroupedList

BaseFeature.__abstractmethods__ = set()


# TODO rewrite from QualitativeFeature
def test_get_qualitative_features() -> None:
    """test function get_qualitative_features"""
    # no value
    result = get_qualitative_features([])
    assert result == []

    # no quantitative
    feature1 = BaseFeature("feature1")
    feature2 = BaseFeature("feature2")
    result = get_qualitative_features([feature1, feature2])
    assert result == []
    result = get_categorical_features([feature1, feature2])
    assert result == []
    result = get_ordinal_features([feature1, feature2])
    assert result == []

    # mixed quantitative
    feature1 = CategoricalFeature("feature1")
    feature2 = BaseFeature("feature2")
    result = get_qualitative_features([feature1, feature2])
    assert result == [feature1]
    result = get_categorical_features([feature1, feature2])
    assert result == [feature1]
    result = get_ordinal_features([feature1, feature2])
    assert result == []

    # mixed quantitative
    feature1 = OrdinalFeature("feature1", values=["test"])
    feature2 = BaseFeature("feature2")
    result = get_qualitative_features([feature1, feature2])
    assert result == [feature1]
    result = get_categorical_features([feature1, feature2])
    assert result == []
    result = get_ordinal_features([feature1, feature2])
    assert result == [feature1]

    # only quantitative
    feature1 = OrdinalFeature("feature1", values=["test"])
    feature2 = CategoricalFeature("feature2")
    result = get_qualitative_features([feature1, feature2])
    assert result == [feature1, feature2]
    result = get_categorical_features([feature1, feature2])
    assert result == [feature2]
    result = get_ordinal_features([feature1, feature2])
    assert result == [feature1]


def test_nan_unique():
    # no sorting with nans
    sample_series = Series([1, 2, 2, 3, 4, 4, 4, float("nan"), float("nan"), 5])
    result = nan_unique(sample_series)
    assert result == nan_unique(Series([1, 2, 2, 3, 4, 4, 4, 5]))
    assert all(r in [1, 2, 3, 4, 5] for r in result)

    # sorting with nan
    sample_series = Series([1, 2, 2, 3, 4, 4, 4, float("nan"), float("nan"), 5])
    result = nan_unique(sample_series, sort=True)
    assert result == [1, 2, 3, 4, 5]

    # empty
    empty_series = Series([], dtype=float)
    result = nan_unique(empty_series)
    expected = []
    assert result == expected

    # only nan
    nan_series = Series([float("nan"), float("nan")])
    result = nan_unique(nan_series)
    expected = []
    assert result == expected

    # no nan no sort
    no_nan_series = Series([1, 2, 2, 3, 4])
    result = nan_unique(no_nan_series)
    assert all(r in [1, 2, 3, 4] for r in result)


@fixture
def sample_categorical_feature() -> CategoricalFeature:
    """Create a sample CategoricalFeature for testing"""

    feature = CategoricalFeature("test_feature")
    feature.update(
        GroupedList({"a": ["1", "2", "3", "4", "5", "a"], "b": ["b"], "c": ["c"], "d": ["e", "f"]}),
        replace=True,
    )
    feature.raw_order = ["1", "2", "3", "4", "5", "a", "b", "c", "d", "e", "f"]
    return feature


@fixture
def sample_ordinal_feature() -> OrdinalFeature:
    """Create a sample CategoricalFeature for testing"""

    feature = OrdinalFeature(
        "test_feature", values=["1", "2", "3", "4", "5", "a", "b", "c", "d", "e", "f"]
    )
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


def test_categorical_feature_type(sample_categorical_feature: CategoricalFeature) -> None:
    """testing type"""

    assert not sample_categorical_feature.is_quantitative
    assert sample_categorical_feature.is_qualitative
    assert not sample_categorical_feature.is_ordinal
    assert sample_categorical_feature.is_categorical


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
    result = sample_ordinal_feature._format_modalities(
        group, content + [sample_ordinal_feature.nan]
    )
    assert result == f"1 to a, {sample_ordinal_feature.nan}"

    # with smaller max_n_chars
    sample_ordinal_feature.max_n_chars = 30
    result = sample_ordinal_feature._format_modalities(
        group, content + [sample_ordinal_feature.nan]
    )
    assert result == f"1 to a, {sample_ordinal_feature.nan}"

    # empty content
    result = sample_ordinal_feature._format_modalities(group, [])
    assert result == group

    # with default
    sample_ordinal_feature.has_default = True
    sample_ordinal_feature.max_n_chars = 30
    result = sample_ordinal_feature._format_modalities(group, content)
    assert result == "1 to a"


def test_categorical_feature_format_modalities(
    sample_categorical_feature: CategoricalFeature,
) -> None:
    """testing _format_modalities"""

    # with large max_n_chars
    group = "b"
    content = sample_categorical_feature.values.get(group)
    result = sample_categorical_feature._format_modalities(group, content)
    assert result == "b"

    # with large max_n_chars
    group = "d"
    content = sample_categorical_feature.values.get(group)
    result = sample_categorical_feature._format_modalities(group, content)
    assert result == "d, e, f"

    # with large max_n_chars
    group = "a"
    content = sample_categorical_feature.values.get(group)
    result = sample_categorical_feature._format_modalities(group, content)
    assert result == "1, 2, 3, 4, 5, a"

    # with smaller max_n_chars
    sample_categorical_feature.max_n_chars = 4
    result = sample_categorical_feature._format_modalities(group, content)
    assert result == "1, 2..."

    # adding nans
    # with smaller max_n_chars
    sample_categorical_feature.max_n_chars = 4
    result = sample_categorical_feature._format_modalities(
        group, content + [sample_categorical_feature.nan]
    )
    assert result == f"1, 2..., {sample_categorical_feature.nan}"

    # with smaller max_n_chars
    sample_categorical_feature.max_n_chars = 30
    result = sample_categorical_feature._format_modalities(
        group, content + [sample_categorical_feature.nan]
    )
    assert result == f"1, 2, 3, 4, 5, a, {sample_categorical_feature.nan}"

    # empty content
    result = sample_categorical_feature._format_modalities(group, [])
    assert result == group

    # with default
    sample_categorical_feature.has_default = True
    sample_categorical_feature.max_n_chars = 30
    result = sample_categorical_feature._format_modalities(group, content)
    assert result == "1, 2, 3, 4, 5, a"


def test_categorical_feature_make_labels(sample_categorical_feature: CategoricalFeature) -> None:
    """test make_labels mehtod"""

    # without ordinal encoding
    result = sample_categorical_feature.make_labels()
    assert result == ["1, 2, 3, 4, 5, a", "b", "c", "d, e, f"]

    # with ordinal encoding
    sample_categorical_feature.ordinal_encoding = True
    result = sample_categorical_feature.make_labels()
    assert result == ["1, 2, 3, 4, 5, a", "b", "c", "d, e, f"]
    assert sample_categorical_feature.value_per_label == {0: "a", 1: "b", 2: "c", 3: "d"}
    assert sample_categorical_feature.label_per_value == {
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "a": 0,
        "b": 1,
        "c": 2,
        "e": 3,
        "f": 3,
        "d": 3,
    }


def test_ordinal_feature_make_labels(sample_ordinal_feature: OrdinalFeature) -> None:
    """test make_labels method"""

    # without ordinal encoding
    result = sample_ordinal_feature.make_labels()
    assert result == ["1 to a", "b", "c", "d to f"]

    # with ordinal encoding
    sample_ordinal_feature.ordinal_encoding = True
    result = sample_ordinal_feature.make_labels()
    assert result == ["1 to a", "b", "c", "d to f"]
    assert sample_ordinal_feature.value_per_label == {0: "a", 1: "b", 2: "c", 3: "d"}
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


def test_check_values_no_unexpected(sample_categorical_feature: CategoricalFeature) -> None:
    """test_check_values"""

    # no unexpected value in data
    sample_categorical_feature.value_per_label = {"label1": "value1", "label2": "value2"}
    sample_categorical_feature.values = GroupedList({"value1": ["value1"], "value2": ["value2"]})
    data = DataFrame({"test_feature": ["label1", "label2"]})
    sample_categorical_feature.check_values(data)

    # unexpected value in data with has_default = false
    sample_categorical_feature.value_per_label = {"label1": "value1"}
    sample_categorical_feature.values = GroupedList({"value1": ["value1"]})
    data = DataFrame({"test_feature": ["label1", "label3"]})
    with raises(ValueError):
        sample_categorical_feature.check_values(data)

    # testing with nans
    sample_categorical_feature.value_per_label = {"label1": "value1"}
    sample_categorical_feature.values = GroupedList({"value1": ["value1"]})
    sample_categorical_feature.nan = "N/A"
    sample_categorical_feature.has_nan = True
    sample_categorical_feature.dropna = True
    data = DataFrame({"test_feature": ["value1", "N/A"]})
    sample_categorical_feature.check_values(data)
    sample_categorical_feature.dropna = False

    # unexpected value in data with has_default = true
    sample_categorical_feature.value_per_label = {"label1": "value1"}
    sample_categorical_feature.values = GroupedList({"value1": ["value1"]})
    sample_categorical_feature.default = "default_value"
    sample_categorical_feature.has_default = True
    data = DataFrame({"test_feature": ["label1", "label3"]})
    sample_categorical_feature.check_values(data)

    # Ensure the unexpected value was added
    assert "label3" in sample_categorical_feature.values.content["default_value"]

    # can not come back from has default
    with raises(RuntimeError):
        sample_categorical_feature.has_default = False


def test_categorical_feature_update_ordinal_encoding(
    sample_categorical_feature: CategoricalFeature,
) -> None:
    """testing update"""

    with raises(AttributeError):  # only already known labels
        sample_categorical_feature.update(
            GroupedList({0: [0], 1: [1], "test": ["tests", "test2"]}), convert_labels=True
        )

    # fitting some nans
    sample_categorical_feature.fit(
        DataFrame({sample_categorical_feature.version: ["a", "c", "f", "1", nan]})
    )
    sample_categorical_feature.ordinal_encoding = True
    sample_categorical_feature.update(
        GroupedList({2: [2, 3]}),
        convert_labels=True,
    )
    print(sample_categorical_feature.content)
    assert sample_categorical_feature.values == ["a", "b", "c"]
    assert sample_categorical_feature.content == {
        "a": ["1", "2", "3", "4", "5", "a"],
        "b": ["b"],
        "c": ["e", "f", "d", "c"],
    }
    assert sample_categorical_feature.labels == [0, 1, 2]
    assert sample_categorical_feature.value_per_label == {
        0: "a",
        1: "b",
        2: "c",
    }
    print(sample_categorical_feature.label_per_value)
    assert sample_categorical_feature.label_per_value == {
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
    sample_categorical_feature.dropna = True
    print(sample_categorical_feature.content)
    sample_categorical_feature.update(
        GroupedList({3: [3, 1]}),
        convert_labels=True,
    )
    assert sample_categorical_feature.values == ["a", "c", NAN]
    assert sample_categorical_feature.content == {
        "a": ["1", "2", "3", "4", "5", "a"],
        "c": ["e", "f", "d", "c"],
        NAN: ["b", NAN],
    }
    assert sample_categorical_feature.labels == [0, 1, 2]
    assert sample_categorical_feature.value_per_label == {
        0: "a",
        1: "c",
        2: NAN,
    }
    print(sample_categorical_feature.label_per_value)
    assert sample_categorical_feature.label_per_value == {
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
        NAN: 2,
        "b": 2,
    }


def test_categorical_feature_update_no_ordinal_encoding(
    sample_categorical_feature: CategoricalFeature,
) -> None:
    """testing update"""

    with raises(AttributeError):  # only already known labels
        sample_categorical_feature.update(
            GroupedList({0: [0], 1: [1], "test": ["tests", "test2"]}), convert_labels=True
        )

    # fitting some nans
    sample_categorical_feature.fit(
        DataFrame({sample_categorical_feature.version: ["a", "c", "f", "1", nan]})
    )
    sample_categorical_feature.ordinal_encoding = False
    sample_categorical_feature.update(
        GroupedList({"d, e, f": ["c", "d, e, f"]}), convert_labels=True
    )
    print(sample_categorical_feature.content)
    assert sample_categorical_feature.values == ["a", "b", "d"]
    assert sample_categorical_feature.content == {
        "a": ["1", "2", "3", "4", "5", "a"],
        "b": ["b"],
        "d": ["c", "e", "f", "d"],
    }
    assert sample_categorical_feature.labels == ["1, 2, 3, 4, 5, a", "b", "c, d, e, f"]
    print(sample_categorical_feature.value_per_label)
    assert sample_categorical_feature.value_per_label == {
        "1, 2, 3, 4, 5, a": "a",
        "b": "b",
        "c, d, e, f": "d",
    }
    print(sample_categorical_feature.label_per_value)
    assert sample_categorical_feature.label_per_value == {
        "1": "1, 2, 3, 4, 5, a",
        "2": "1, 2, 3, 4, 5, a",
        "3": "1, 2, 3, 4, 5, a",
        "4": "1, 2, 3, 4, 5, a",
        "5": "1, 2, 3, 4, 5, a",
        "a": "1, 2, 3, 4, 5, a",
        "b": "b",
        "c": "c, d, e, f",
        "e": "c, d, e, f",
        "f": "c, d, e, f",
        "d": "c, d, e, f",
    }

    # adding nans
    sample_categorical_feature.dropna = True
    print(sample_categorical_feature.content)
    sample_categorical_feature.update(GroupedList({NAN: [NAN, "b"]}), convert_labels=True)
    assert sample_categorical_feature.values == ["a", "d", NAN]
    print(sample_categorical_feature.content)
    assert sample_categorical_feature.content == {
        "a": ["1", "2", "3", "4", "5", "a"],
        "d": ["c", "e", "f", "d"],
        "__NAN__": ["b", "__NAN__"],
    }
    assert sample_categorical_feature.labels == ["1, 2, 3, 4, 5, a", "c, d, e, f", f"b, {NAN}"]
    assert sample_categorical_feature.value_per_label == {
        "1, 2, 3, 4, 5, a": "a",
        "c, d, e, f": "d",
        f"b, {NAN}": NAN,
    }
    print(sample_categorical_feature.label_per_value)
    assert sample_categorical_feature.label_per_value == {
        "1": "1, 2, 3, 4, 5, a",
        "2": "1, 2, 3, 4, 5, a",
        "3": "1, 2, 3, 4, 5, a",
        "4": "1, 2, 3, 4, 5, a",
        "5": "1, 2, 3, 4, 5, a",
        "a": "1, 2, 3, 4, 5, a",
        "c": "c, d, e, f",
        "e": "c, d, e, f",
        "f": "c, d, e, f",
        "d": "c, d, e, f",
        NAN: f"b, {NAN}",
        "b": f"b, {NAN}",
    }


def test_ordinal_feature_update_ordinal_encoding(
    sample_ordinal_feature: OrdinalFeature,
) -> None:
    """testing update"""

    with raises(AttributeError):  # only already known labels
        sample_ordinal_feature.update(
            GroupedList({0: [0], 1: [1], "test": ["tests", "test2"]}), convert_labels=True
        )

    # fitting some nans
    sample_ordinal_feature.fit(
        DataFrame({sample_ordinal_feature.version: ["a", "c", "f", "1", nan]})
    )
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
        0: "a",
        1: "b",
        2: "c",
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
    assert sample_ordinal_feature.values == ["a", "c", NAN]
    assert sample_ordinal_feature.content == {
        "a": ["1", "2", "3", "4", "5", "a"],
        "c": ["e", "f", "d", "c"],
        NAN: ["b", NAN],
    }
    assert sample_ordinal_feature.labels == [0, 1, 2]
    assert sample_ordinal_feature.value_per_label == {
        0: "a",
        1: "c",
        2: NAN,
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
        NAN: 2,
        "b": 2,
    }


def test_ordinal_feature_update_no_ordinal_encoding(
    sample_ordinal_feature: OrdinalFeature,
) -> None:
    """testing update"""

    with raises(AttributeError):  # only already known labels
        sample_ordinal_feature.update(
            GroupedList({0: [0], 1: [1], "test": ["tests", "test2"]}), convert_labels=True
        )

    # fitting some nans
    sample_ordinal_feature.fit(
        DataFrame({sample_ordinal_feature.version: ["a", "c", "f", "1", nan]})
    )
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
    sample_ordinal_feature.update(GroupedList({NAN: [NAN, "b"]}), convert_labels=True)
    assert sample_ordinal_feature.values == ["a", "d", NAN]
    print(sample_ordinal_feature.content)
    assert sample_ordinal_feature.content == {
        "a": ["1", "2", "3", "4", "5", "a"],
        "d": ["c", "e", "f", "d"],
        "__NAN__": ["b", "__NAN__"],
    }
    assert sample_ordinal_feature.labels == ["1 to a", "c to f", f"b, {NAN}"]
    assert sample_ordinal_feature.value_per_label == {
        "1 to a": "a",
        "c to f": "d",
        f"b, {NAN}": NAN,
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
        NAN: f"b, {NAN}",
        "b": f"b, {NAN}",
    }


def test_categorical_feature_get_summary(sample_categorical_feature: CategoricalFeature) -> None:
    """test function get_summary"""

    summary = sample_categorical_feature.get_summary()

    expected_summary = [
        {
            "feature": "Categorical('test_feature')",
            "label": "a",
            "content": ["1", "2", "3", "4", "5", "a"],
        },
        {"feature": "Categorical('test_feature')", "label": "b", "content": "b"},
        {"feature": "Categorical('test_feature')", "label": "c", "content": "c"},
        {"feature": "Categorical('test_feature')", "label": "d", "content": ["e", "f", "d"]},
    ]
    assert summary == expected_summary

    sample_categorical_feature.ordinal_encoding = True
    summary = sample_categorical_feature.get_summary()

    expected_summary = [
        {
            "feature": "Categorical('test_feature')",
            "label": 0,
            "content": ["1", "2", "3", "4", "5", "a"],
        },
        {"feature": "Categorical('test_feature')", "label": 1, "content": "b"},
        {"feature": "Categorical('test_feature')", "label": 2, "content": "c"},
        {"feature": "Categorical('test_feature')", "label": 3, "content": ["e", "f", "d"]},
    ]
    assert summary == expected_summary

    # fitting some nans
    sample_categorical_feature.fit(DataFrame({sample_categorical_feature.version: ["a", "b", nan]}))
    sample_categorical_feature.dropna = True
    sample_categorical_feature.ordinal_encoding = True
    summary = sample_categorical_feature.get_summary()

    expected_summary = [
        {
            "feature": "Categorical('test_feature')",
            "label": 0,
            "content": ["1", "2", "3", "4", "5", "a"],
        },
        {"feature": "Categorical('test_feature')", "label": 1, "content": "b"},
        {"feature": "Categorical('test_feature')", "label": 2, "content": "c"},
        {"feature": "Categorical('test_feature')", "label": 3, "content": ["e", "f", "d"]},
        {"feature": "Categorical('test_feature')", "label": 4, "content": "__NAN__"},
    ]
    assert summary == expected_summary

    # adding some default
    sample_categorical_feature.has_default = True
    sample_categorical_feature.ordinal_encoding = True
    summary = sample_categorical_feature.get_summary()

    expected_summary = [
        {
            "feature": "Categorical('test_feature')",
            "label": 0,
            "content": ["1", "2", "3", "4", "5", "a"],
        },
        {"feature": "Categorical('test_feature')", "label": 1, "content": "b"},
        {"feature": "Categorical('test_feature')", "label": 2, "content": "c"},
        {"feature": "Categorical('test_feature')", "label": 3, "content": ["e", "f", "d"]},
        {"feature": "Categorical('test_feature')", "label": 4, "content": "__NAN__"},
        {"feature": "Categorical('test_feature')", "label": 5, "content": "__OTHER__"},
    ]
    assert summary == expected_summary


def test_ordinal_feature_get_summary(sample_ordinal_feature: OrdinalFeature) -> None:
    """test function get_summary"""

    summary = sample_ordinal_feature.get_summary()

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
    summary = sample_ordinal_feature.get_summary()

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
    summary = sample_ordinal_feature.get_summary()

    expected_summary = [
        {
            "feature": "Ordinal('test_feature')",
            "label": 0,
            "content": ["1", "2", "3", "4", "5", "a"],
        },
        {"feature": "Ordinal('test_feature')", "label": 1, "content": "b"},
        {"feature": "Ordinal('test_feature')", "label": 2, "content": "c"},
        {"feature": "Ordinal('test_feature')", "label": 3, "content": ["e", "f", "d"]},
        {"feature": "Ordinal('test_feature')", "label": 4, "content": "__NAN__"},
    ]
    print(summary)
    assert summary == expected_summary

    # adding some default
    sample_ordinal_feature.has_default = True
    sample_ordinal_feature.ordinal_encoding = True
    summary = sample_ordinal_feature.get_summary()

    expected_summary = [
        {
            "feature": "Ordinal('test_feature')",
            "label": 0,
            "content": ["1", "2", "3", "4", "5", "a"],
        },
        {"feature": "Ordinal('test_feature')", "label": 1, "content": "b"},
        {"feature": "Ordinal('test_feature')", "label": 2, "content": "c"},
        {"feature": "Ordinal('test_feature')", "label": 3, "content": ["e", "f", "d"]},
        {"feature": "Ordinal('test_feature')", "label": 4, "content": "__NAN__"},
        {"feature": "Ordinal('test_feature')", "label": 5, "content": "__OTHER__"},
    ]
    assert summary == expected_summary


def test_categorical_feature_fit_initial_values(
    sample_categorical_feature: CategoricalFeature,
) -> None:
    """test fit method"""

    sample_categorical_feature.values = None
    sample_categorical_feature.raw_order = []

    data = DataFrame({"test_feature": ["value1", "value2", "value3", "value1"]})

    sample_categorical_feature.fit(data)

    # Check that values have been set
    assert sample_categorical_feature.values is not None
    assert "value1" in sample_categorical_feature.values.content
    assert "value2" in sample_categorical_feature.values.content
    assert "value3" in sample_categorical_feature.values.content

    # Check that raw_order has been set
    assert sample_categorical_feature.raw_order == ["value1", "value2", "value3"]


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


def test_ordinal_feature_fit_with_nan(sample_ordinal_feature: CategoricalFeature) -> None:
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


def test_categorical_feature_fit_with_nan(sample_categorical_feature: CategoricalFeature) -> None:
    """test fit method"""

    sample_categorical_feature.values = None
    sample_categorical_feature.raw_order = []

    data = DataFrame({"test_feature": ["value1", "value2", "value3", "value1", nan]})

    sample_categorical_feature.fit(data)

    # Check that values have been set
    assert sample_categorical_feature.values is not None
    assert "value1" in sample_categorical_feature.values.content
    assert "value2" in sample_categorical_feature.values.content
    assert "value3" in sample_categorical_feature.values.content
    assert sample_categorical_feature.has_nan

    # Check that raw_order has been set
    assert sample_categorical_feature.raw_order == ["value1", "value2", "value3"]


def test_fit_existing_values(sample_categorical_feature: CategoricalFeature) -> None:
    """test fit funtion"""

    # unexpected data for feature "already fitted" -> values already set
    sample_categorical_feature.values = GroupedList({"value1": ["value1"], "value2": ["value2"]})
    sample_categorical_feature.raw_order = []
    data = DataFrame({"test_feature": ["value1", "value2", "value3", "value1"]})
    with raises(ValueError):
        sample_categorical_feature.fit(data)


def test_fit_with_existing_raw_order(sample_categorical_feature: CategoricalFeature) -> None:
    """test fit funtion"""

    sample_categorical_feature.values = None
    sample_categorical_feature.raw_order = ["value1", "value2"]
    data = DataFrame({"test_feature": ["value1", "value2", "value3", "value1"]})
    sample_categorical_feature.fit(data)

    # Check that raw_order remains unchanged
    assert sample_categorical_feature.raw_order == ["value1", "value2"]


def test_fit_check_values_called(
    sample_categorical_feature: CategoricalFeature, monkeypatch
) -> None:
    """test fit funtion"""

    sample_categorical_feature.values = None
    sample_categorical_feature.raw_order = []

    # Patch the check_values method
    check_values_called = False

    def mock_check_values(X):
        nonlocal check_values_called
        check_values_called = True

    monkeypatch.setattr(sample_categorical_feature, "check_values", mock_check_values)

    data = DataFrame({"test_feature": ["value1", "value2", "value3", "value1"]})

    sample_categorical_feature.fit(data)

    # Ensure check_values was called
    assert check_values_called
