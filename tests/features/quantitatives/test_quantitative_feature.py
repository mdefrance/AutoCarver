"""set of tests for quantitative features"""

import numpy as np
import pandas as pd
from pytest import fixture, raises

from AutoCarver.config import Constants
from AutoCarver.features.quantitatives.quantitative_feature import (
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
    assert result == 3

    # with large gap
    result = min_decimals_to_differentiate([1.0, 100.0])
    assert result == 1


def test_min_decimals_to_differentiate_bankers_rounding_collision() -> None:
    """Regression test: scientific-notation formatting collisions due to
    banker's rounding (e.g. -118.04 and -118.05 → "-1.180e+02" at 3 decimals)
    used to make ``GroupedList(format_quantiles(...))`` silently drop labels,
    which left the trailing ``np.inf`` leader without an entry in
    ``label_per_value`` and surfaced as ``KeyError: inf`` in
    ``transform_quantitative_feature``. The function must return enough
    decimals for every formatted string to be distinct.
    """
    # Two longitudes that collide at 3 decimals via banker's rounding.
    values = sorted([-118.05, -118.04])
    decimals = min_decimals_to_differentiate(values, min_decimals=1)
    formatted = [f"{n:.{decimals}e}" for n in values]
    assert len(set(formatted)) == len(values), f"format collision at {decimals} decimals: {formatted}"

    # A dense cluster of close longitudes from the California Housing dataset.
    dense = sorted([-118.04, -118.05, -118.06, -118.15, -118.16, -118.24, -118.25])
    decimals = min_decimals_to_differentiate(dense, min_decimals=1)
    formatted = [f"{n:.{decimals}e}" for n in dense]
    assert len(set(formatted)) == len(dense)


def test_format_quantiles_produces_unique_strings_for_collision_prone_inputs() -> None:
    """``format_quantiles`` must always return distinct strings — otherwise
    ``GroupedList(format_quantiles(...))`` deduplicates labels and downstream
    ``transform_quantitative_feature`` raises ``KeyError: inf``.

    The collision-prone case: closely-spaced longitudes whose scientific-notation
    formatting collides at the naively-computed precision due to banker's rounding.
    """
    # Pre-fix bug: at 3 decimals, both round to "-1.180e+02"
    longitudes = [-118.05, -118.04, -118.0, -117.96, -117.5]
    labels = format_quantiles(longitudes)
    assert len(set(labels)) == len(labels), f"duplicate labels: {labels}"
    # one boundary per quantile + open upper bound
    assert len(labels) == len(longitudes) + 1


def test_format_quantiles_empty_list() -> None:
    """test function min_decimals_to_differentiate"""
    # no value
    result = format_quantiles([])
    assert result == ["(-inf, inf)"]

    # signle value
    result = format_quantiles([0.5])
    assert result == ["(-inf, 5.0e-01]", "(5.0e-01, inf)"]

    # multiple values
    result = format_quantiles([0.1, 0.5, 0.9])
    assert result == [
        "(-inf, 1.00e-01]",
        "(1.00e-01, 5.00e-01]",
        "(5.00e-01, 9.00e-01]",
        "(9.00e-01, inf)",
    ]

    # multiple close values
    result = format_quantiles([1.0, 1.01, 1.1])
    print(result)
    assert result == [
        "(-inf, 1.000e+00]",
        "(1.000e+00, 1.010e+00]",
        "(1.010e+00, 1.100e+00]",
        "(1.100e+00, inf)",
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
    feature.update(GroupedList([0.1, 0.5, 0.9, np.inf]))  # needs to have np.inf in there
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
    sample_quantitative_feature.fit(pd.DataFrame({sample_quantitative_feature.version: [1, 2, 3, 4, np.nan]}))
    # labels without setting dropna to true
    assert sample_quantitative_feature.values == [0.1, 0.5, 0.9, np.inf]
    labels = sample_quantitative_feature.make_labels()
    assert labels == [
        "(-inf, 1.00e-01]",
        "(1.00e-01, 5.00e-01]",
        "(5.00e-01, 9.00e-01]",
        "(9.00e-01, inf)",
    ]
    # setting dropna true
    sample_quantitative_feature.dropna = True
    labels = sample_quantitative_feature.make_labels()
    assert sample_quantitative_feature.values == [0.1, 0.5, 0.9, np.inf, Constants.NAN]
    assert labels == [
        "(-inf, 1.00e-01]",
        "(1.00e-01, 5.00e-01]",
        "(5.00e-01, 9.00e-01]",
        "(9.00e-01, inf)",
        Constants.NAN,
    ]
    # resetting dropna
    sample_quantitative_feature.dropna = False
    labels = sample_quantitative_feature.make_labels()
    assert sample_quantitative_feature.values == [0.1, 0.5, 0.9, np.inf]
    assert labels == [
        "(-inf, 1.00e-01]",
        "(1.00e-01, 5.00e-01]",
        "(5.00e-01, 9.00e-01]",
        "(9.00e-01, inf)",
    ]

    # setting has_default true
    sample_quantitative_feature.has_default = True
    labels = sample_quantitative_feature.make_labels()
    assert sample_quantitative_feature.values == [0.1, 0.5, 0.9, np.inf]
    assert labels == [
        "(-inf, 1.00e-01]",
        "(1.00e-01, 5.00e-01]",
        "(5.00e-01, 9.00e-01]",
        "(9.00e-01, inf)",
    ]
    # resetting has_default false
    sample_quantitative_feature.has_default = False
    labels = sample_quantitative_feature.make_labels()
    assert sample_quantitative_feature.values == [0.1, 0.5, 0.9, np.inf]
    assert labels == [
        "(-inf, 1.00e-01]",
        "(1.00e-01, 5.00e-01]",
        "(5.00e-01, 9.00e-01]",
        "(9.00e-01, inf)",
    ]


def test_quantitative_feature_update_no_ordinal_encoding(
    sample_quantitative_feature: QuantitativeFeature,
) -> None:
    """testing update"""

    # fitting some nans
    sample_quantitative_feature.fit(pd.DataFrame({sample_quantitative_feature.version: [1, 2, 3, 4, np.nan]}))
    sample_quantitative_feature.update(
        GroupedList({"(-inf, 1.00e-01]": ["(-inf, 1.00e-01]", "(1.00e-01, 5.00e-01]"]}),
        convert_labels=True,
    )
    assert sample_quantitative_feature.values == [0.5, 0.9, np.inf]
    assert sample_quantitative_feature.content == {0.5: [0.1, 0.5], 0.9: [0.9], np.inf: [np.inf]}
    assert sample_quantitative_feature.labels == [
        "(-inf, 5.00e-01]",
        "(5.00e-01, 9.00e-01]",
        "(9.00e-01, inf)",
    ]
    assert sample_quantitative_feature.value_per_label == {
        "(-inf, 5.00e-01]": 0.5,
        "(5.00e-01, 9.00e-01]": 0.9,
        "(9.00e-01, inf)": np.inf,
    }
    assert sample_quantitative_feature.label_per_value == {
        0.1: "(-inf, 5.00e-01]",
        0.5: "(-inf, 5.00e-01]",
        0.9: "(5.00e-01, 9.00e-01]",
        np.inf: "(9.00e-01, inf)",
    }

    # adding nans
    sample_quantitative_feature.dropna = True
    print(sample_quantitative_feature.label_per_value)
    sample_quantitative_feature.update(
        GroupedList({Constants.NAN: [Constants.NAN, "(-inf, 5.00e-01]"]}), convert_labels=True
    )
    assert sample_quantitative_feature.values == [0.5, 0.9, np.inf]
    assert sample_quantitative_feature.content == {
        0.5: [Constants.NAN, 0.1, 0.5],
        0.9: [0.9],
        np.inf: [np.inf],
    }
    assert sample_quantitative_feature.labels == [
        f"(-inf, 5.00e-01], {Constants.NAN}",
        "(5.00e-01, 9.00e-01]",
        "(9.00e-01, inf)",
    ]
    assert sample_quantitative_feature.value_per_label == {
        f"(-inf, 5.00e-01], {Constants.NAN}": 0.5,
        "(5.00e-01, 9.00e-01]": 0.9,
        "(9.00e-01, inf)": np.inf,
    }
    assert sample_quantitative_feature.label_per_value == {
        Constants.NAN: f"(-inf, 5.00e-01], {Constants.NAN}",
        0.1: f"(-inf, 5.00e-01], {Constants.NAN}",
        0.5: f"(-inf, 5.00e-01], {Constants.NAN}",
        0.9: "(5.00e-01, 9.00e-01]",
        np.inf: "(9.00e-01, inf)",
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
    sample_quantitative_feature.fit(pd.DataFrame({sample_quantitative_feature.version: [1, 2, 3, 4, np.nan]}))
    sample_quantitative_feature.ordinal_encoding = True
    print("values", sample_quantitative_feature.values)
    print("labels", sample_quantitative_feature.labels)
    print("value_per_label", sample_quantitative_feature.value_per_label)
    print("label_per_value", sample_quantitative_feature.label_per_value)
    sample_quantitative_feature.update(GroupedList({2: [2, 3]}), convert_labels=True)
    assert sample_quantitative_feature.values == [0.1, 0.5, np.inf]
    assert sample_quantitative_feature.content == {0.1: [0.1], 0.5: [0.5], np.inf: [0.9, np.inf]}
    assert sample_quantitative_feature.labels == [0, 1, 2]
    assert sample_quantitative_feature.value_per_label == {
        0: "(-inf, 1.00e-01]",
        1: "(1.00e-01, 5.00e-01]",
        2: "(5.00e-01, inf)",
    }
    assert sample_quantitative_feature.label_per_value == {
        0.1: 0,
        0.5: 1,
        0.9: 2,
        np.inf: 2,
    }

    # adding nans
    sample_quantitative_feature.dropna = True
    print(sample_quantitative_feature.content)
    sample_quantitative_feature.update(GroupedList({3: [3, 1]}), convert_labels=True)
    assert sample_quantitative_feature.values == [0.1, 0.5, np.inf]
    assert sample_quantitative_feature.content == {
        0.1: [0.1],
        0.5: [Constants.NAN, 0.5],
        np.inf: [0.9, np.inf],
    }
    assert sample_quantitative_feature.labels == [0, 1, 2]
    assert sample_quantitative_feature.value_per_label == {
        0: "(-inf, 1.00e-01]",
        1: f"(1.00e-01, 5.00e-01], {Constants.NAN}",
        2: "(5.00e-01, inf)",
    }
    assert sample_quantitative_feature.label_per_value == {
        0.1: 0,
        0.5: 1,
        Constants.NAN: 1,
        0.9: 2,
        np.inf: 2,
    }


def test_get_summary(sample_quantitative_feature: QuantitativeFeature) -> None:
    """test function get_summary"""

    summary = sample_quantitative_feature.summary
    expected_summary = [
        {
            "feature": "Quantitative('test_feature')",
            "label": "(-inf, 1.00e-01]",
            "content": "(-inf, 1.00e-01]",
        },
        {
            "feature": "Quantitative('test_feature')",
            "label": "(1.00e-01, 5.00e-01]",
            "content": "(1.00e-01, 5.00e-01]",
        },
        {
            "feature": "Quantitative('test_feature')",
            "label": "(5.00e-01, 9.00e-01]",
            "content": "(5.00e-01, 9.00e-01]",
        },
        {
            "feature": "Quantitative('test_feature')",
            "label": "(9.00e-01, inf)",
            "content": "(9.00e-01, inf)",
        },
    ]
    assert summary == expected_summary

    sample_quantitative_feature.ordinal_encoding = True
    summary = sample_quantitative_feature.summary
    expected_summary = [
        {"feature": "Quantitative('test_feature')", "label": 0, "content": "(-inf, 1.00e-01]"},
        {
            "feature": "Quantitative('test_feature')",
            "label": 1,
            "content": "(1.00e-01, 5.00e-01]",
        },
        {
            "feature": "Quantitative('test_feature')",
            "label": 2,
            "content": "(5.00e-01, 9.00e-01]",
        },
        {"feature": "Quantitative('test_feature')", "label": 3, "content": "(9.00e-01, inf)"},
    ]
    assert summary == expected_summary

    # fitting some nans
    sample_quantitative_feature.fit(pd.DataFrame({sample_quantitative_feature.version: [0.1, 0.2, 0.3, 0.4, np.nan]}))
    sample_quantitative_feature.ordinal_encoding = True
    sample_quantitative_feature.dropna = True
    summary = sample_quantitative_feature.summary
    print(summary)
    expected_summary = [
        {"feature": "Quantitative('test_feature')", "label": 0, "content": "(-inf, 1.00e-01]"},
        {
            "feature": "Quantitative('test_feature')",
            "label": 1,
            "content": "(1.00e-01, 5.00e-01]",
        },
        {
            "feature": "Quantitative('test_feature')",
            "label": 2,
            "content": "(5.00e-01, 9.00e-01]",
        },
        {"feature": "Quantitative('test_feature')", "label": 3, "content": "(9.00e-01, inf)"},
        {"feature": "Quantitative('test_feature')", "label": 4, "content": "__NAN__"},
    ]
    assert summary == expected_summary
