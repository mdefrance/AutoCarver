"""Set of tests for Qualitative correlation measures module."""

from numpy import nan
from pandas import Series
from pytest import FixtureRequest, fixture
from AutoCarver.selectors import BaseMeasure, Chi2Measure, CramerVMeasure, TschuprowTMeasure


# setting BaseMeasure as non abstract class for the duration of the test
BaseMeasure.__abstractmethods__ = set()


@fixture
def base_measure() -> BaseMeasure:
    return BaseMeasure(threshold=0.5)


@fixture(params=[Chi2Measure, CramerVMeasure, TschuprowTMeasure])
def measure(request: type[FixtureRequest]) -> BaseMeasure:
    return request.param(threshold=0.5)


@fixture
def series_data() -> Series:
    x = Series([1, 2, 3, 4, 5])
    return x


@fixture
def nan_series_data() -> Series:
    x = Series([nan, nan, 3, 4, 5])
    return x


def test_validate_with_value_below_threshold(base_measure: BaseMeasure) -> None:
    """checks that validates works as expected for base measure (remove low correlation)"""
    base_measure.value = 0
    print(base_measure.value, base_measure.threshold, base_measure.validate())
    assert not base_measure.validate(), "not validating correcly (remove low correlation)"


def test_validate_with_value_above_threshold(base_measure: BaseMeasure) -> None:
    """checks that validates works as expected for base measure (keep high correlation)"""
    base_measure.value = 1
    print(base_measure.value, base_measure.threshold, base_measure.validate())
    assert base_measure.validate(), "not validating correcly (keep high correlation)"


def test_validate_with_null_value(base_measure: BaseMeasure) -> None:
    """checks that validates works as expected for base measure"""
    base_measure.value = None
    print(base_measure.value, base_measure.threshold, base_measure.validate())
    assert (
        not base_measure.validate()
    ), "when value is missing, should default to false (remove low correlation)"


def test_measure_type(measure: BaseMeasure) -> None:
    """checks types of x and y"""
    assert measure.is_x_qualitative, "x should be qualitative"
    assert not measure.is_x_quantitative, "x should be qualitative"
    assert measure.is_y_qualitative, "y should be qualitative"
    assert not measure.is_y_quantitative, "y should be qualitative"


def test_compute_association(measure: BaseMeasure, series_data: Series) -> None:
    """checks that correlation measurement works"""
    association = measure.compute_association(series_data, series_data)
    assert association is not None, "not correctly computed association"
    assert measure.value == association, "not stored measurement"


def test_validate_with_computed_association_below_threshold(
    measure: BaseMeasure, series_data: Series
) -> None:
    """checks that correlated features are not removed"""
    measure.compute_association(series_data, series_data * 0)
    print(measure.value, measure.threshold, measure.validate())
    assert not measure.validate(), "kept feature with lower than threshold correlation"


def test_validate_with_computed_association_above_threshold(
    measure: BaseMeasure, series_data: Series
) -> None:
    """checks that non-correlated features are removed"""
    measure.compute_association(series_data, series_data)
    print(measure.__name__, measure.value, measure.threshold, measure.validate())
    assert measure.validate(), "removed feature with lower than threshold correlation"


def test_compute_association_with_nans(measure: BaseMeasure, nan_series_data: Series) -> None:
    """checks that correlation measurement works"""
    association = measure.compute_association(nan_series_data, nan_series_data)
    assert association is not None, "not correctly computed association"
    assert measure.value == association, "not stored measurement"


def test_validate_with_computed_association_below_threshold_with_nans(
    measure: BaseMeasure, nan_series_data: Series, series_data: Series
) -> None:
    """checks that correlated features are not removed"""
    measure.compute_association(nan_series_data, series_data * 0)
    print(measure.__name__, measure.value, measure.threshold, measure.validate())
    assert not measure.validate(), "kept feature with lower than threshold correlation"


def test_validate_with_computed_association_above_threshold_with_nans(
    measure: BaseMeasure, nan_series_data: Series, series_data: Series
) -> None:
    """checks that non-correlated features are removed"""
    measure.compute_association(nan_series_data, series_data)
    print(measure.__name__, measure, measure.value, measure.threshold, measure.validate())
    assert measure.validate(), "removed feature with lower than threshold correlation"
