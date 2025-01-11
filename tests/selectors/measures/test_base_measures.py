"""Set of tests for base correlation measures module."""

from numpy import nan
from pandas import Series
from pytest import fixture, raises

from AutoCarver.features import BaseFeature
from AutoCarver.selectors import (
    AbsoluteMeasure,
    BaseMeasure,
    ModeMeasure,
    NanMeasure,
    OutlierMeasure,
)
from AutoCarver.selectors.measures import ReversibleMeasure

# setting BaseMeasure, OutlierMeasure, AbsoluteMeasure, ReversibleMeasure as non abstract classes
# for the duration of the test
OutlierMeasure.__abstractmethods__ = set()
AbsoluteMeasure.__abstractmethods__ = set()
BaseMeasure.__abstractmethods__ = set()
ReversibleMeasure.__abstractmethods__ = set()

THRESHOLD = 1


@fixture
def base_measure() -> BaseMeasure:
    return BaseMeasure(threshold=THRESHOLD)


@fixture
def reversible_measure() -> ReversibleMeasure:
    return ReversibleMeasure()


@fixture
def nan_measure() -> NanMeasure:
    return NanMeasure()


@fixture
def mode_measure() -> ModeMeasure:
    return ModeMeasure()


@fixture
def outlier_measure() -> OutlierMeasure:
    return OutlierMeasure(threshold=THRESHOLD)


@fixture
def absolute_measure() -> AbsoluteMeasure:
    return AbsoluteMeasure(threshold=THRESHOLD)


def test_to_dict(base_measure: BaseMeasure) -> None:
    """test dict conversion"""

    value = THRESHOLD - 1
    base_measure.value = value
    expected_dict = {
        "BaseMeasure": {
            "value": value,
            "threshold": THRESHOLD,
            "valid": False,
            "info": {"higher_is_better": True, "correlation_with": "target", "is_default": False},
        }
    }
    print(base_measure.to_dict())
    assert base_measure.to_dict() == expected_dict


def test_update_feature(base_measure: BaseMeasure) -> None:
    """test dict conversion"""

    value = THRESHOLD - 1
    base_measure.value = value
    feature = BaseFeature("test")
    base_measure._update_feature(feature)
    expected_statistics = {
        "BaseMeasure": {
            "value": value,
            "threshold": THRESHOLD,
            "valid": False,
            "info": {
                "higher_is_better": True,
                "correlation_with": "target",
                "is_default": False,
            },
        }
    }
    assert feature.measures == expected_statistics

    # not set value
    base_measure.value = None
    with raises(ValueError):
        base_measure._update_feature(feature)


def test_base_validate(base_measure: BaseMeasure) -> None:
    """checks that validates works as expected for base measure (remove low correlation)"""

    # value below threshold
    base_measure.value = base_measure.threshold - 1
    assert not base_measure.validate(), "not validating correcly (remove low correlation)"

    # value above threshold
    base_measure.value = base_measure.threshold + 1
    assert base_measure.validate(), "not validating correcly (keep high correlation)"

    # value at threshold
    base_measure.value = base_measure.threshold
    assert base_measure.validate(), "not validating correcly (keep high correlation)"

    # null value
    base_measure.value = None
    assert (
        not base_measure.validate()
    ), "when value is missing, should default to false (remove low correlation)"


def test_outlier_validate(outlier_measure: OutlierMeasure) -> None:
    """checks that validates works as expected for outlier measure (remove low correlation)"""

    # value below threshold
    outlier_measure.value = outlier_measure.threshold - 1
    assert outlier_measure.validate(), "not validating correcly (keep low outlier rates)"

    # value above threshold
    outlier_measure.value = outlier_measure.threshold + 1
    assert not outlier_measure.validate(), "not validating correcly (drop high outlier rates)"

    # value at threshold
    outlier_measure.value = outlier_measure.threshold
    assert not outlier_measure.validate(), "not validating correcly (drop high outlier rates)"

    # null value
    outlier_measure.value = None
    assert outlier_measure.validate(), "keep undefined outlier rates"


def test_outlier_measure_type(outlier_measure: OutlierMeasure) -> None:
    """checks types of x and y"""
    assert not outlier_measure.is_x_qualitative, "x should be quantitative"
    assert outlier_measure.is_x_quantitative, "x should be quantitative"


def test_absolute_validate(absolute_measure: AbsoluteMeasure) -> None:
    """checks that validates works as expected for absolute measure (remove low correlation)"""

    # positive value below threshold
    absolute_measure.value = absolute_measure.threshold - 1
    assert not absolute_measure.validate(), "not validating correcly (remove low correlation)"

    # negative value below threshold
    absolute_measure.value = -(absolute_measure.threshold - 1)
    assert not absolute_measure.validate(), "not validating correcly (remove low correlation)"

    # positive value above threshold
    absolute_measure.value = absolute_measure.threshold + 1
    assert absolute_measure.validate(), "not validating correcly (keep high correlation)"

    # negative value above threshold
    absolute_measure.value = -(absolute_measure.threshold + 1)
    assert absolute_measure.validate(), "not validating correcly (keep high correlation)"

    # positive value at threshold
    absolute_measure.value = absolute_measure.threshold
    assert absolute_measure.validate(), "not validating correcly (keep high correlation)"

    # negative value at threshold
    absolute_measure.value = -absolute_measure.threshold
    assert absolute_measure.validate(), "not validating correcly (keep high correlation)"

    # null value
    absolute_measure.value = None
    assert (
        not absolute_measure.validate()
    ), "when value is missing, should default to false (remove low correlation)"


def test_nan_measure(nan_measure: NanMeasure) -> None:
    """tests nan measure"""
    series = Series([1, 2, None, 4])
    assert nan_measure.compute_association(series, None) == 0.25, "should measure None as missing"
    series = Series([1, 2, nan, 4])
    assert nan_measure.compute_association(series, None) == 0.25, "should measure nan as missing"
    assert nan_measure.is_default, "should be default"

    nan_measure.threshold = 1.0
    nan_measure.value = 0.5
    assert nan_measure.validate(), "should be valid"

    nan_measure.threshold = 0.5
    nan_measure.value = 1.0
    assert not nan_measure.validate(), "should not be valid"


def test_mode_measure(mode_measure: ModeMeasure) -> None:
    """tests mode measure"""
    series = Series([1, 2, 2, 4])
    assert mode_measure.compute_association(series, None) == 0.5
    assert mode_measure.is_default, "should be default"

    mode_measure.threshold = 1.0
    mode_measure.value = 0.5
    assert mode_measure.validate(), "should be valid"

    mode_measure.threshold = 0.5
    mode_measure.value = 1.0
    assert not mode_measure.validate(), "should not be valid"


def test_reverse_xy(reversible_measure: ReversibleMeasure) -> None:
    """tests reversible measure"""
    assert reversible_measure.reversed is False
    assert reversible_measure.reverse_xy() is True
    assert reversible_measure.reversed is True
