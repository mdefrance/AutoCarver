"""Set of tests for base correlation filters."""

from pytest import fixture

from AutoCarver.features import BaseFeature
from AutoCarver.selectors.filters import BaseFilter, ValidFilter

# setting BaseFilter as non abstract classes for the duration of the test
BaseFilter.__abstractmethods__ = set()


@fixture
def valid_filter() -> ValidFilter:
    return ValidFilter()


@fixture
def base_filter() -> BaseFilter:
    return BaseFilter()


def test_update_feature(base_filter: BaseFilter) -> None:
    """tests base filter"""
    feature = BaseFeature("feature")
    base_filter._update_feature(feature, 0.5, True, {"some_info": "value"})
    expected_statistics = {
        "BaseFilter": {
            "value": 0.5,
            "threshold": 1.0,
            "valid": True,
            "info": {"higher_is_better": False, "some_info": "value"},
        }
    }
    assert feature.filters == expected_statistics


def test_valid_filter(valid_filter: ValidFilter) -> None:
    """tests base validation filter"""
    feature1 = BaseFeature("feature1")
    feature1.measures = {"nan": {"valid": True}, "mode": {"valid": True}}
    feature2 = BaseFeature("feature2")
    feature2.measures = {"nan": {"valid": False}, "mode": {"valid": True}}
    assert valid_filter.filter(None, [feature1, feature2]) == [feature1], "should remove non valid measures"
    assert valid_filter.is_default, "should be default"
