"""Set of tests for base correlation filters."""

from pytest import fixture
from AutoCarver.features import BaseFeature

from AutoCarver.selectors.filters import ValidFilter


@fixture
def valid_filter() -> ValidFilter:
    return ValidFilter()


def test_valid_filter(valid_filter: ValidFilter) -> None:
    """tests base"""
    feature1 = BaseFeature("feature1")
    feature1.statistics = {"measures": {"nan": {"valid": True}, "mode": {"valid": True}}}
    feature2 = BaseFeature("feature2")
    feature2.statistics = {"measures": {"nan": {"valid": False}, "mode": {"valid": True}}}
    assert valid_filter.filter(None, [feature1, feature2]) == [
        feature1
    ], "should remove non valid measures"
