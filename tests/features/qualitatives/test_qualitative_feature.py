""" set of tests for quantitative features"""

from pandas import Series
from pytest import fixture

from AutoCarver.features import (
    CategoricalFeature,
    OrdinalFeature,
    get_categorical_features,
    get_ordinal_features,
    get_qualitative_features,
)
from AutoCarver.features.qualitatives.qualitative_feature import nan_unique
from AutoCarver.features.utils.base_feature import BaseFeature
from AutoCarver.features.utils.grouped_list import GroupedList

BaseFeature.__abstractmethods__ = set()


# TODO rewrite for QualitativeFeature
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
