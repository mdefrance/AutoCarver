"""Set of tests for Qualitative correlation measures module."""

from numpy import nan
from pandas import Series, isna
from pytest import FixtureRequest, fixture
from AutoCarver.selectors import (
    BaseMeasure,
    OutlierMeasure,
    DistanceMeasure,
    PearsonMeasure,
    SpearmanMeasure,
    RMeasure,
    KruskalMeasure,
    IqrOutlierMeasure,
    ZscoreOutlierMeasure,
)


# setting OutlierMeasure as non abstract class for the duration of the test
OutlierMeasure.__abstractmethods__ = set()


threshold = 0.1


@fixture
def outlier_measure() -> OutlierMeasure:
    return OutlierMeasure(threshold=threshold)


@fixture(params=[DistanceMeasure, PearsonMeasure, SpearmanMeasure])
def quanti_quanti_measure(request: type[FixtureRequest]) -> BaseMeasure:
    return request.param(threshold=threshold)


@fixture(params=[KruskalMeasure])
def quanti_quali_measure(request: type[FixtureRequest]) -> BaseMeasure:
    return request.param(threshold=threshold)


@fixture(params=[RMeasure])
def quanti_binary_measure(request: type[FixtureRequest]) -> BaseMeasure:
    return request.param(threshold=threshold)


@fixture
def series_data() -> Series:
    x = Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    return x


# @fixture
# def nan_series_data() -> Series:
#     x = Series([nan, nan, 3, 4, 5])
#     return x


def test_validate_with_value_below_threshold(outlier_measure: OutlierMeasure) -> None:
    """checks that validates works as expected for base measure (keep low outlier rates)"""
    outlier_measure.value = 0
    print(outlier_measure.value, outlier_measure.threshold, outlier_measure.validate())
    assert outlier_measure.validate(), "not validating correcly (keep low outlier rates)"


def test_validate_with_value_above_threshold(outlier_measure: OutlierMeasure) -> None:
    """checks that validates works as expected for base measure (drop high outlier rates)"""
    outlier_measure.value = 1
    print(outlier_measure.value, outlier_measure.threshold, outlier_measure.validate())
    assert not outlier_measure.validate(), "not validating correcly (drop high outlier rates)"


def test_validate_with_null_value(outlier_measure: OutlierMeasure) -> None:
    """checks that validates works as expected for base measure"""
    outlier_measure.value = None
    print(outlier_measure.value, outlier_measure.threshold, outlier_measure.validate())
    assert outlier_measure.validate(), "keep undefined outlier rates"


def test_outlier_measure_type(outlier_measure: BaseMeasure) -> None:
    """checks types of x and y"""
    assert not outlier_measure.is_x_qualitative, "x should be quantitative"
    assert outlier_measure.is_x_quantitative, "x should be quantitative"


def test_quanti_quanti_measure_type(quanti_quanti_measure: BaseMeasure) -> None:
    """checks types of x and y"""
    assert not quanti_quanti_measure.is_x_qualitative, "x should be quantitative"
    assert quanti_quanti_measure.is_x_quantitative, "x should be quantitative"
    assert not quanti_quanti_measure.is_y_qualitative, "y should be quantitative"
    assert quanti_quanti_measure.is_y_quantitative, "y should be quantitative"


def test_quanti_quali_measure_type(quanti_quali_measure: BaseMeasure) -> None:
    """checks types of x and y"""
    assert not quanti_quali_measure.is_x_qualitative, "x should be quantitative"
    assert quanti_quali_measure.is_x_quantitative, "x should be quantitative"
    assert quanti_quali_measure.is_y_qualitative, "y should be qualitative"
    assert not quanti_quali_measure.is_y_quantitative, "y should be qualitative"


def test_quanti_binary_measure_type(quanti_binary_measure: BaseMeasure) -> None:
    """checks types of x and y"""
    assert not quanti_binary_measure.is_x_qualitative, "x should be quantitative"
    assert quanti_binary_measure.is_x_quantitative, "x should be quantitative"
    assert quanti_binary_measure.is_y_qualitative, "y should be quantitative"
    assert not quanti_binary_measure.is_y_quantitative, "y should be quantitative"
    assert quanti_binary_measure.is_y_binary, "y should be binary"


@fixture
def binary_series_data() -> Series:
    x = Series([1, 1, 0, 1, 0, 0, 0, 0, 0, 0])
    return x


@fixture
def quali_series_data() -> Series:
    x = Series([2, 2, 0, 1, 1, 1, 1, 0, 0, 0])
    return x


def test_quanti_binary_compute_association(
    quanti_binary_measure: BaseMeasure, series_data: Series, binary_series_data: Series
) -> None:
    """checks that correlation measurement works"""

    # without nans
    association = quanti_binary_measure.compute_association(series_data, binary_series_data)
    assert association is not None, "not correctly computed association"
    assert quanti_binary_measure.value == association, "not stored measurement"

    # with nans
    association = quanti_binary_measure.compute_association(
        series_data.replace(1, nan), binary_series_data
    )
    assert association is not None, "not correctly computed association"
    assert quanti_binary_measure.value == association, "not stored measurement"


def test_quanti_quali_compute_association(
    quanti_quali_measure: BaseMeasure, series_data: Series, quali_series_data: Series
) -> None:
    """checks that correlation measurement works"""

    # without nans
    association = quanti_quali_measure.compute_association(series_data, quali_series_data)
    assert association is not None, "not correctly computed association"
    assert quanti_quali_measure.value == association, "not stored measurement"

    # with nans
    association = quanti_quali_measure.compute_association(
        series_data.replace(1, nan), quali_series_data
    )
    print(
        quanti_quali_measure.__name__,
        quanti_quali_measure.value,
        association,
        quanti_quali_measure.validate(),
        quanti_quali_measure.threshold,
    )
    assert association is not None, "not correctly computed association"
    assert quanti_quali_measure.value == association or (
        isna(quanti_quali_measure.value) and isna(association)
    ), "not stored measurement"


def test_quanti_quanti_compute_association(
    quanti_quanti_measure: BaseMeasure, series_data: Series
) -> None:
    """checks that correlation measurement works"""

    # without nans
    association = quanti_quanti_measure.compute_association(series_data, series_data)
    assert association is not None, "not correctly computed association"
    assert quanti_quanti_measure.value == association, "not stored measurement"

    # with nans
    association = quanti_quanti_measure.compute_association(
        series_data.replace(1, nan), series_data
    )
    assert association is not None, "not correctly computed association"
    assert quanti_quanti_measure.value == association, "not stored measurement"


def test_quanti_quali_validate_with_computed_association_below_threshold(
    quanti_quali_measure: BaseMeasure, quali_series_data: Series, series_data: Series
) -> None:
    """checks that correlated features are not removed"""

    # without nans
    quanti_quali_measure.compute_association(series_data, quali_series_data)
    print(
        quanti_quali_measure.value,
        quanti_quali_measure.threshold,
        quanti_quali_measure.validate(),
    )
    assert not quanti_quali_measure.validate(), "kept feature with lower than threshold correlation"

    # with nans
    quanti_quali_measure.compute_association(series_data.replace(1, nan), quali_series_data)
    print(
        quanti_quali_measure.__name__,
        quanti_quali_measure.value,
        quanti_quali_measure.threshold,
        quanti_quali_measure.validate(),
    )
    assert not quanti_quali_measure.validate(), "kept feature with lower than threshold correlation"


def test_quanti_quali_validate_with_computed_association_above_threshold(
    quanti_quali_measure: BaseMeasure, quali_series_data: Series, series_data: Series
) -> None:
    """checks that non-correlated features are removed"""

    # without nans
    quanti_quali_measure.compute_association(series_data, quali_series_data)
    print(
        quanti_quali_measure.__name__,
        quanti_quali_measure.value,
        quanti_quali_measure.threshold,
        quanti_quali_measure.validate(),
    )
    assert quanti_quali_measure.validate(), "removed feature with lower than threshold correlation"

    # with nans
    quanti_quali_measure.compute_association(series_data.replace(1, nan), quali_series_data)
    print(
        quanti_quali_measure.__name__,
        quanti_quali_measure.value,
        quanti_quali_measure.threshold,
        quanti_quali_measure.validate(),
    )
    assert quanti_quali_measure.validate(), "removed feature with lower than threshold correlation"


def test_quanti_quanti_validate_with_computed_association_below_threshold(
    quanti_quanti_measure: BaseMeasure, series_data: Series
) -> None:
    """checks that correlated features are not removed"""

    # without nans
    quanti_quanti_measure.compute_association(series_data, series_data * 0)
    print(
        quanti_quanti_measure.value,
        quanti_quanti_measure.threshold,
        quanti_quanti_measure.validate(),
    )
    assert (
        not quanti_quanti_measure.validate()
    ), "kept feature with lower than threshold correlation"

    # with nans
    quanti_quanti_measure.compute_association(series_data.replace(1, nan), series_data * 0)
    print(
        quanti_quanti_measure.__name__,
        quanti_quanti_measure.value,
        quanti_quanti_measure.threshold,
        quanti_quanti_measure.validate(),
    )
    assert (
        not quanti_quanti_measure.validate()
    ), "kept feature with lower than threshold correlation"


def test_quanti_quanti_validate_with_computed_association_above_threshold(
    quanti_quanti_measure: BaseMeasure, series_data: Series
) -> None:
    """checks that non-correlated features are removed"""

    # without nans
    quanti_quanti_measure.compute_association(series_data, series_data**20)
    print(
        quanti_quanti_measure.__name__,
        quanti_quanti_measure.value,
        quanti_quanti_measure.threshold,
        quanti_quanti_measure.validate(),
    )
    assert quanti_quanti_measure.validate(), "removed feature with lower than threshold correlation"

    # with nans
    quanti_quanti_measure.compute_association(series_data.replace(1, nan), series_data**20)
    print(
        quanti_quanti_measure.__name__,
        quanti_quanti_measure.value,
        quanti_quanti_measure.threshold,
        quanti_quanti_measure.validate(),
    )
    assert quanti_quanti_measure.validate(), "removed feature with lower than threshold correlation"
