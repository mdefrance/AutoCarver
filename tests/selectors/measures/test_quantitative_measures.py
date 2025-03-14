"""Set of tests for Qualitative correlation measures module."""

from numpy import nan
from pandas import Series, isna
from pytest import FixtureRequest, fixture

from AutoCarver.selectors import (
    BaseMeasure,
    DistanceMeasure,
    IqrOutlierMeasure,
    KruskalMeasure,
    PearsonMeasure,
    RMeasure,
    SpearmanMeasure,
    ZscoreOutlierMeasure,
)

threshold = 0.1


@fixture(params=[IqrOutlierMeasure, ZscoreOutlierMeasure])
def outlier_measure(request: FixtureRequest) -> BaseMeasure:
    return request.param(threshold=threshold)


@fixture(params=[DistanceMeasure, PearsonMeasure, SpearmanMeasure])
def quanti_quanti_measure(request: FixtureRequest) -> BaseMeasure:
    return request.param(threshold=threshold)


@fixture(params=[KruskalMeasure])
def quanti_quali_measure(request: FixtureRequest) -> BaseMeasure:
    return request.param(threshold=threshold)


@fixture(params=[RMeasure])
def quanti_binary_measure(request: FixtureRequest) -> BaseMeasure:
    return request.param(threshold=threshold)


@fixture
def series_data() -> Series:
    x = Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    return x


def test_quanti_quanti_measure_type(quanti_quanti_measure: BaseMeasure) -> None:
    """checks types of x and y"""
    assert not quanti_quanti_measure.is_x_qualitative, "x should be quantitative"
    assert quanti_quanti_measure.is_x_quantitative, "x should be quantitative"
    assert not quanti_quanti_measure.is_y_qualitative, "y should be quantitative"
    assert quanti_quanti_measure.is_y_quantitative, "y should be quantitative"
    assert not quanti_quanti_measure.is_default, "should not be default"


def test_quanti_quali_measure_type(quanti_quali_measure: BaseMeasure) -> None:
    """checks types of x and y"""
    assert not quanti_quali_measure.is_x_qualitative, "x should be quantitative"
    assert quanti_quali_measure.is_x_quantitative, "x should be quantitative"
    assert quanti_quali_measure.is_y_qualitative, "y should be qualitative"
    assert not quanti_quali_measure.is_y_quantitative, "y should be qualitative"
    assert not quanti_quali_measure.is_default, "should not be default"

    # testing reversing measure
    quanti_quali_measure.reverse_xy()
    assert quanti_quali_measure.is_x_qualitative, "(reversed) x should be qualitative"
    assert not quanti_quali_measure.is_x_quantitative, "(reversed) x should be qualitative"
    assert not quanti_quali_measure.is_y_qualitative, "(reversed) y should be quantitative"
    assert quanti_quali_measure.is_y_quantitative, "(reversed) y should be quantitative"


def test_quanti_binary_measure_type(quanti_binary_measure: BaseMeasure) -> None:
    """checks types of x and y"""
    assert not quanti_binary_measure.is_x_qualitative, "x should be quantitative"
    assert quanti_binary_measure.is_x_quantitative, "x should be quantitative"
    assert quanti_binary_measure.is_y_qualitative, "y should be quantitative"
    assert not quanti_binary_measure.is_y_quantitative, "y should be quantitative"
    assert quanti_binary_measure.is_y_binary, "y should be binary"
    assert not quanti_binary_measure.is_default, "should not be default"


@fixture
def binary_series_data() -> Series:
    """creates a series with binary data"""
    x = Series([1, 1, 0, 1, 0, 0, 0, 0, 0, 0])
    return x


@fixture
def quali_series_data() -> Series:
    """creates a series with qualitative data"""
    x = Series([2, 2, 0, 1, 1, 1, 1, 0, 0, 0])
    return x


@fixture
def outlier_series_data() -> Series:
    """creates a series with outliers"""
    x = Series(
        [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            50,
            51,
        ]
    )
    return x


def test_outlier_compute_association(outlier_measure: BaseMeasure, series_data: Series) -> None:
    """checks that correlation measurement works"""

    # without nans
    association = outlier_measure.compute_association(series_data)
    assert association is not None, "not correctly computed association"
    assert outlier_measure.value == association, "not stored measurement"

    # with nans
    association = outlier_measure.compute_association(series_data.replace(1, nan))
    assert association is not None, "not correctly computed association"
    assert outlier_measure.value == association, "not stored measurement"


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


def test_quanti_quanli_reverse_xy(
    quanti_quali_measure: BaseMeasure, series_data: Series, quali_series_data: Series
) -> None:
    """checks that reverse measurement works"""

    association = quanti_quali_measure.compute_association(series_data, quali_series_data)
    quanti_quali_measure.reverse_xy()
    association_reversed = quanti_quali_measure.compute_association(quali_series_data, series_data)
    assert association == association_reversed, "not same correlation when reversed"


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


def test_outlier_validate_with_computed_association_below_threshold(
    outlier_measure: BaseMeasure, series_data: Series
) -> None:
    """checks that correlated features are not removed"""

    # without nans
    outlier_measure.threshold = 1
    outlier_measure.compute_association(series_data)
    print(
        outlier_measure.__name__,
        outlier_measure.value,
        outlier_measure.threshold,
        outlier_measure.validate(),
    )
    assert outlier_measure.validate(), "kept feature with lower than threshold correlation"

    # with nans
    outlier_measure.compute_association(series_data.replace(1, nan))
    print(
        outlier_measure.__name__,
        outlier_measure.value,
        outlier_measure.threshold,
        outlier_measure.validate(),
    )
    assert outlier_measure.validate(), "kept feature with lower than threshold correlation"


def test_quanti_quali_validate_with_computed_association_below_threshold(
    quanti_quali_measure: BaseMeasure, quali_series_data: Series, series_data: Series
) -> None:
    """checks that correlated features are not removed"""

    # without nans
    quanti_quali_measure.threshold = 10
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

    quanti_quanti_measure.threshold = 5
    # without nans
    quanti_quanti_measure.compute_association(series_data, (series_data <= 2).astype(int))
    print(
        quanti_quanti_measure.__name__,
        quanti_quanti_measure.value,
        quanti_quanti_measure.threshold,
        quanti_quanti_measure.validate(),
    )
    assert (
        not quanti_quanti_measure.validate()
    ), "kept feature with lower than threshold correlation"

    # with nans
    quanti_quanti_measure.compute_association(
        series_data.replace(1, nan), (series_data <= 2).astype(int)
    )
    print(
        quanti_quanti_measure.__name__,
        quanti_quanti_measure.value,
        quanti_quanti_measure.threshold,
        quanti_quanti_measure.validate(),
    )
    assert (
        not quanti_quanti_measure.validate()
    ), "kept feature with lower than threshold correlation"

    # test with negative value
    quanti_quanti_measure.value = -0.5
    quanti_quanti_measure.thresold = 0.6
    assert not quanti_quanti_measure.validate()


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

    # test with negative value
    quanti_quanti_measure.value = -0.5
    quanti_quanti_measure.thresold = 0.3
    assert quanti_quanti_measure.validate()


def test_quanti_binary_validate_with_computed_association_below_threshold(
    quanti_binary_measure: BaseMeasure, binary_series_data: Series, series_data: Series
) -> None:
    """checks that correlated features are not removed"""

    quanti_binary_measure.threshold = 1.0
    # without nans
    quanti_binary_measure.compute_association(series_data, binary_series_data)
    print(
        quanti_binary_measure.value,
        quanti_binary_measure.threshold,
        quanti_binary_measure.validate(),
    )
    assert (
        not quanti_binary_measure.validate()
    ), "kept feature with lower than threshold correlation"

    # with nans
    quanti_binary_measure.compute_association(series_data.replace(1, nan), binary_series_data)
    print(
        quanti_binary_measure.__name__,
        quanti_binary_measure.value,
        quanti_binary_measure.threshold,
        quanti_binary_measure.validate(),
    )
    assert (
        not quanti_binary_measure.validate()
    ), "kept feature with lower than threshold correlation"


def test_quanti_binary_validate_with_computed_association_above_threshold(
    quanti_binary_measure: BaseMeasure, binary_series_data: Series, series_data: Series
) -> None:
    """checks that non-correlated features are removed"""

    # without nans
    quanti_binary_measure.compute_association(series_data, binary_series_data)
    print(
        quanti_binary_measure.__name__,
        quanti_binary_measure.value,
        quanti_binary_measure.threshold,
        quanti_binary_measure.validate(),
    )
    assert quanti_binary_measure.validate(), "removed feature with lower than threshold correlation"

    # with nans
    quanti_binary_measure.compute_association(series_data.replace(1, nan), binary_series_data)
    print(
        quanti_binary_measure.__name__,
        quanti_binary_measure.value,
        quanti_binary_measure.threshold,
        quanti_binary_measure.validate(),
    )
    assert quanti_binary_measure.validate(), "removed feature with lower than threshold correlation"


def test_outlier_validate_with_computed_association_above_threshold(
    outlier_measure: BaseMeasure, outlier_series_data: Series
) -> None:
    """checks that correlated features are not removed"""

    # without nans
    outlier_measure.threshold = 0.02
    outlier_measure.compute_association(outlier_series_data.replace(51, 5e15))
    print(
        outlier_measure.__name__,
        outlier_measure.value,
        outlier_measure.threshold,
        outlier_measure.validate(),
    )
    assert not outlier_measure.validate(), "kept feature with lower than threshold correlation"

    # with nans
    print(outlier_series_data.dtype, outlier_series_data.apply(type).unique())
    outlier_measure.compute_association(outlier_series_data.replace(50, nan).replace(51, 5e50))
    print(
        outlier_measure.__name__,
        outlier_measure.value,
        outlier_measure.threshold,
        outlier_measure.validate(),
    )
    assert not outlier_measure.validate(), "kept feature with lower than threshold correlation"
