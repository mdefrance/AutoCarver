""" set of tests for quantitative features"""

from numpy import inf, nan
from pandas import DataFrame
from pytest import fixture, raises

from AutoCarver.config import Constants
from AutoCarver.features.quantitative_features import (
    QuantitativeFeature,
    format_quantiles,
    get_quantitative_features,
    min_decimals_to_differentiate,
)
from AutoCarver.features.utils.base_feature import BaseFeature
from AutoCarver.features.utils.grouped_list import GroupedList

BaseFeature.__abstractmethods__ = set()


def test_min_decimals_to_differentiate() -> None:
    """test function min_decimals_to_differentiate"""
    # without value
    result = min_decimals_to_differentiate([])
    assert result == 0
    result = min_decimals_to_differentiate([0])
    assert result == 0

    # with identical values (smallest_diff == 0)
    result = min_decimals_to_differentiate([1.0, 1.0, 1.0])
    assert result == 0

    # with different numbers
    result = min_decimals_to_differentiate([1.0, 1.01, 1.1])
    assert result == 2

    # with large gap
    result = min_decimals_to_differentiate([1.0, 100.0])
    assert result == 0


def test_format_quantiles_empty_list() -> None:
    """test function min_decimals_to_differentiate"""
    # no value
    result = format_quantiles([])
    assert result == ["-inf < x < inf"]

    # signle value
    result = format_quantiles([0.5])
    assert result == ["x <= 5.0e-01", "5.0e-01 < x"]

    # multiple values
    result = format_quantiles([0.1, 0.5, 0.9])
    assert result == [
        "x <= 1.0e-01",
        "1.0e-01 < x <= 5.0e-01",
        "5.0e-01 < x <= 9.0e-01",
        "9.0e-01 < x",
    ]

    # multiple close values
    result = format_quantiles([1.0, 1.01, 1.1])
    print(result)
    assert result == [
        "x <= 1.00e+00",
        "1.00e+00 < x <= 1.01e+00",
        "1.01e+00 < x <= 1.10e+00",
        "1.10e+00 < x",
    ]


def test_get_quantitative_features() -> None:
    """test function get_quantitative_features"""
    # no value
    result = get_quantitative_features([])
    assert result == []

    # no quantitative
    feature1 = BaseFeature("feature1")
    feature2 = BaseFeature("feature2")
    result = get_quantitative_features([feature1, feature2])
    assert result == []

    # mixed quantitative
    feature1 = QuantitativeFeature("feature1")
    feature2 = BaseFeature("feature2")
    result = get_quantitative_features([feature1, feature2])
    assert result == [feature1]

    # only quantitative
    feature1 = QuantitativeFeature("feature1")
    feature2 = QuantitativeFeature("feature2")
    result = get_quantitative_features([feature1, feature2])
    assert result == [feature1, feature2]


@fixture
def sample_quantitative_feature() -> QuantitativeFeature:
    """Create a sample QuantitativeFeature for testing"""

    feature = QuantitativeFeature("test_feature")
    feature.update(GroupedList([0.1, 0.5, 0.9, inf]))  # needs to have inf in there
    return feature


def test_quantitative_feature_type(sample_quantitative_feature: QuantitativeFeature) -> None:
    """testing type"""

    assert sample_quantitative_feature.is_quantitative
    assert not sample_quantitative_feature.is_qualitative
    assert not sample_quantitative_feature.is_ordinal
    assert not sample_quantitative_feature.is_categorical


def test_quantitative_feature_make_labels(sample_quantitative_feature: QuantitativeFeature) -> None:
    """testing make_labels"""

    # fitting some nans
    sample_quantitative_feature.fit(
        DataFrame({sample_quantitative_feature.version: [1, 2, 3, 4, nan]})
    )
    # labels without setting dropna to true
    assert sample_quantitative_feature.values == [0.1, 0.5, 0.9, inf]
    labels = sample_quantitative_feature.make_labels()
    assert labels == [
        "x <= 1.0e-01",
        "1.0e-01 < x <= 5.0e-01",
        "5.0e-01 < x <= 9.0e-01",
        "9.0e-01 < x",
    ]
    # setting dropna true
    sample_quantitative_feature.dropna = True
    labels = sample_quantitative_feature.make_labels()
    assert sample_quantitative_feature.values == [0.1, 0.5, 0.9, inf, Constants.NAN]
    assert labels == [
        "x <= 1.0e-01",
        "1.0e-01 < x <= 5.0e-01",
        "5.0e-01 < x <= 9.0e-01",
        "9.0e-01 < x",
        Constants.NAN,
    ]
    # resetting dropna
    sample_quantitative_feature.dropna = False
    labels = sample_quantitative_feature.make_labels()
    assert sample_quantitative_feature.values == [0.1, 0.5, 0.9, inf]
    assert labels == [
        "x <= 1.0e-01",
        "1.0e-01 < x <= 5.0e-01",
        "5.0e-01 < x <= 9.0e-01",
        "9.0e-01 < x",
    ]

    # setting has_default true
    sample_quantitative_feature.has_default = True
    labels = sample_quantitative_feature.make_labels()
    assert sample_quantitative_feature.values == [0.1, 0.5, 0.9, inf]
    assert labels == [
        "x <= 1.0e-01",
        "1.0e-01 < x <= 5.0e-01",
        "5.0e-01 < x <= 9.0e-01",
        "9.0e-01 < x",
    ]
    # resetting has_default false
    sample_quantitative_feature.has_default = False
    labels = sample_quantitative_feature.make_labels()
    assert sample_quantitative_feature.values == [0.1, 0.5, 0.9, inf]
    assert labels == [
        "x <= 1.0e-01",
        "1.0e-01 < x <= 5.0e-01",
        "5.0e-01 < x <= 9.0e-01",
        "9.0e-01 < x",
    ]


def test_quantitative_feature_update_no_ordinal_encoding(
    sample_quantitative_feature: QuantitativeFeature,
) -> None:
    """testing update"""

    # fitting some nans
    sample_quantitative_feature.fit(
        DataFrame({sample_quantitative_feature.version: [1, 2, 3, 4, nan]})
    )
    sample_quantitative_feature.update(
        GroupedList({"x <= 1.0e-01": ["x <= 1.0e-01", "1.0e-01 < x <= 5.0e-01"]}),
        convert_labels=True,
    )
    assert sample_quantitative_feature.values == [0.5, 0.9, inf]
    assert sample_quantitative_feature.content == {0.5: [0.1, 0.5], 0.9: [0.9], inf: [inf]}
    assert sample_quantitative_feature.labels == [
        "x <= 5.0e-01",
        "5.0e-01 < x <= 9.0e-01",
        "9.0e-01 < x",
    ]
    assert sample_quantitative_feature.value_per_label == {
        "x <= 5.0e-01": 0.5,
        "5.0e-01 < x <= 9.0e-01": 0.9,
        "9.0e-01 < x": inf,
    }
    assert sample_quantitative_feature.label_per_value == {
        0.1: "x <= 5.0e-01",
        0.5: "x <= 5.0e-01",
        0.9: "5.0e-01 < x <= 9.0e-01",
        inf: "9.0e-01 < x",
    }

    # adding nans
    sample_quantitative_feature.dropna = True
    print(sample_quantitative_feature.label_per_value)
    sample_quantitative_feature.update(
        GroupedList({Constants.NAN: [Constants.NAN, "x <= 5.0e-01"]}), convert_labels=True
    )
    assert sample_quantitative_feature.values == [0.5, 0.9, inf]
    assert sample_quantitative_feature.content == {
        0.5: [Constants.NAN, 0.1, 0.5],
        0.9: [0.9],
        inf: [inf],
    }
    assert sample_quantitative_feature.labels == [
        "x <= 5.0e-01",
        "5.0e-01 < x <= 9.0e-01",
        "9.0e-01 < x",
    ]
    assert sample_quantitative_feature.value_per_label == {
        "x <= 5.0e-01": 0.5,
        "5.0e-01 < x <= 9.0e-01": 0.9,
        "9.0e-01 < x": inf,
    }
    assert sample_quantitative_feature.label_per_value == {
        Constants.NAN: "x <= 5.0e-01",
        0.1: "x <= 5.0e-01",
        0.5: "x <= 5.0e-01",
        0.9: "5.0e-01 < x <= 9.0e-01",
        inf: "9.0e-01 < x",
    }


def test_quantitative_feature_update_ordinal_encoding(
    sample_quantitative_feature: QuantitativeFeature,
) -> None:
    """testing update"""

    with raises(AttributeError):  # only already known labels
        sample_quantitative_feature.update(
            GroupedList({0: [0], 1: [1], "test": ["tests", "test2"]}), convert_labels=True
        )

    # fitting some nans
    sample_quantitative_feature.fit(
        DataFrame({sample_quantitative_feature.version: [1, 2, 3, 4, nan]})
    )
    sample_quantitative_feature.ordinal_encoding = True
    print("values", sample_quantitative_feature.values)
    print("labels", sample_quantitative_feature.labels)
    print("value_per_label", sample_quantitative_feature.value_per_label)
    print("label_per_value", sample_quantitative_feature.label_per_value)
    sample_quantitative_feature.update(GroupedList({2: [2, 3]}), convert_labels=True)
    assert sample_quantitative_feature.values == [0.1, 0.5, inf]
    assert sample_quantitative_feature.content == {0.1: [0.1], 0.5: [0.5], inf: [0.9, inf]}
    assert sample_quantitative_feature.labels == [0, 1, 2]
    assert sample_quantitative_feature.value_per_label == {
        0: "x <= 1.0e-01",
        1: "1.0e-01 < x <= 5.0e-01",
        2: "5.0e-01 < x",
    }
    assert sample_quantitative_feature.label_per_value == {
        0.1: 0,
        0.5: 1,
        0.9: 2,
        inf: 2,
    }

    # adding nans
    sample_quantitative_feature.dropna = True
    print(sample_quantitative_feature.content)
    sample_quantitative_feature.update(GroupedList({3: [3, 1]}), convert_labels=True)
    assert sample_quantitative_feature.values == [0.1, 0.5, inf]
    assert sample_quantitative_feature.content == {
        0.1: [0.1],
        0.5: [Constants.NAN, 0.5],
        inf: [0.9, inf],
    }
    assert sample_quantitative_feature.labels == [0, 1, 2]
    assert sample_quantitative_feature.value_per_label == {
        0: "x <= 1.0e-01",
        1: "1.0e-01 < x <= 5.0e-01",
        2: "5.0e-01 < x",
    }
    assert sample_quantitative_feature.label_per_value == {
        0.1: 0,
        0.5: 1,
        Constants.NAN: 1,
        0.9: 2,
        inf: 2,
    }


def test_get_summary(sample_quantitative_feature: QuantitativeFeature) -> None:
    """test function get_summary"""

    summary = sample_quantitative_feature.summary
    expected_summary = [
        {
            "feature": "Quantitative('test_feature')",
            "label": "x <= 1.0e-01",
            "content": "x <= 1.0e-01",
        },
        {
            "feature": "Quantitative('test_feature')",
            "label": "1.0e-01 < x <= 5.0e-01",
            "content": "1.0e-01 < x <= 5.0e-01",
        },
        {
            "feature": "Quantitative('test_feature')",
            "label": "5.0e-01 < x <= 9.0e-01",
            "content": "5.0e-01 < x <= 9.0e-01",
        },
        {
            "feature": "Quantitative('test_feature')",
            "label": "9.0e-01 < x",
            "content": "9.0e-01 < x",
        },
    ]
    assert summary == expected_summary

    sample_quantitative_feature.ordinal_encoding = True
    summary = sample_quantitative_feature.summary
    expected_summary = [
        {"feature": "Quantitative('test_feature')", "label": 0, "content": "x <= 1.0e-01"},
        {
            "feature": "Quantitative('test_feature')",
            "label": 1,
            "content": "1.0e-01 < x <= 5.0e-01",
        },
        {
            "feature": "Quantitative('test_feature')",
            "label": 2,
            "content": "5.0e-01 < x <= 9.0e-01",
        },
        {"feature": "Quantitative('test_feature')", "label": 3, "content": "9.0e-01 < x"},
    ]
    assert summary == expected_summary

    # fitting some nans
    sample_quantitative_feature.fit(
        DataFrame({sample_quantitative_feature.version: [0.1, 0.2, 0.3, 0.4, nan]})
    )
    sample_quantitative_feature.ordinal_encoding = True
    sample_quantitative_feature.dropna = True
    summary = sample_quantitative_feature.summary
    print(summary)
    expected_summary = [
        {"feature": "Quantitative('test_feature')", "label": 0, "content": "x <= 1.0e-01"},
        {
            "feature": "Quantitative('test_feature')",
            "label": 1,
            "content": "1.0e-01 < x <= 5.0e-01",
        },
        {
            "feature": "Quantitative('test_feature')",
            "label": 2,
            "content": "5.0e-01 < x <= 9.0e-01",
        },
        {"feature": "Quantitative('test_feature')", "label": 3, "content": "9.0e-01 < x"},
        {"feature": "Quantitative('test_feature')", "label": 4, "content": "__NAN__"},
    ]
    assert summary == expected_summary
