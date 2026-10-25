"""Set of tests for Qualitative correlation measures module."""

from importlib.util import find_spec

import numpy as np
import pandas as pd
from pytest import FixtureRequest, approx, fixture, mark, param, raises
from scipy.stats import kruskal

from AutoCarver.selectors import (
    BaseMeasure,
    DistanceMeasure,
    IqrOutlierMeasure,
    KruskalEpsilonSquaredMeasure,
    KruskalEtaSquaredMeasure,
    KruskalMeasure,
    PearsonMeasure,
    RMeasure,
    SpearmanMeasure,
    ZscoreOutlierMeasure,
)

threshold = 0.1

# DistanceMeasure needs the optional dcor package (pip install autocarver[dcor])
needs_dcor = mark.skipif(find_spec("dcor") is None, reason="dcor extra not installed")


@fixture(params=[IqrOutlierMeasure, ZscoreOutlierMeasure])
def outlier_measure(request: FixtureRequest) -> BaseMeasure:
    return request.param(threshold=threshold)


@fixture(params=[param(DistanceMeasure, marks=needs_dcor), PearsonMeasure, SpearmanMeasure])
def quanti_quanti_measure(request: FixtureRequest) -> BaseMeasure:
    return request.param(threshold=threshold)


@fixture(params=[KruskalMeasure, KruskalEpsilonSquaredMeasure, KruskalEtaSquaredMeasure])
def quanti_quali_measure(request: FixtureRequest) -> BaseMeasure:
    return request.param(threshold=threshold)


@fixture(params=[RMeasure])
def quanti_binary_measure(request: FixtureRequest) -> BaseMeasure:
    return request.param(threshold=threshold)


@fixture
def series_data() -> pd.Series:
    x = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
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
def binary_series_data() -> pd.Series:
    """creates a series with binary data"""
    x = pd.Series([1, 1, 0, 1, 0, 0, 0, 0, 0, 0])
    return x


@fixture
def quali_series_data() -> pd.Series:
    """creates a series with qualitative data"""
    x = pd.Series([2, 2, 0, 1, 1, 1, 1, 0, 0, 0])
    return x


@fixture
def outlier_series_data() -> pd.Series:
    """creates a series with outliers"""
    x = pd.Series(
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


def test_outlier_compute_association(outlier_measure: BaseMeasure, series_data: pd.Series) -> None:
    """checks that correlation measurement works"""

    # without nans
    association = outlier_measure.compute_association(series_data)
    assert association is not None, "not correctly computed association"
    assert outlier_measure.value == association, "not stored measurement"

    # with nans
    association = outlier_measure.compute_association(series_data.replace(1, np.nan))
    assert association is not None, "not correctly computed association"
    assert outlier_measure.value == association, "not stored measurement"


def test_quanti_binary_compute_association(
    quanti_binary_measure: BaseMeasure, series_data: pd.Series, binary_series_data: pd.Series
) -> None:
    """checks that correlation measurement works"""

    # without nans
    association = quanti_binary_measure.compute_association(series_data, binary_series_data)
    assert association is not None, "not correctly computed association"
    assert quanti_binary_measure.value == association, "not stored measurement"

    # with nans
    association = quanti_binary_measure.compute_association(series_data.replace(1, np.nan), binary_series_data)
    assert association is not None, "not correctly computed association"
    assert quanti_binary_measure.value == association, "not stored measurement"


def test_quanti_quali_compute_association(
    quanti_quali_measure: BaseMeasure, series_data: pd.Series, quali_series_data: pd.Series
) -> None:
    """checks that correlation measurement works"""

    # without nans
    association = quanti_quali_measure.compute_association(series_data, quali_series_data)
    assert association is not None, "not correctly computed association"
    assert quanti_quali_measure.value == association, "not stored measurement"

    # with nans
    association = quanti_quali_measure.compute_association(series_data.replace(1, np.nan), quali_series_data)
    print(
        quanti_quali_measure.__name__,
        quanti_quali_measure.value,
        association,
        quanti_quali_measure.validate(),
        quanti_quali_measure.threshold,
    )
    assert association is not None, "not correctly computed association"
    assert quanti_quali_measure.value == association or (
        pd.isna(quanti_quali_measure.value) and pd.isna(association)
    ), "not stored measurement"


def test_quanti_quanli_reverse_xy(
    quanti_quali_measure: BaseMeasure, series_data: pd.Series, quali_series_data: pd.Series
) -> None:
    """checks that reverse measurement works"""

    association = quanti_quali_measure.compute_association(series_data, quali_series_data)
    quanti_quali_measure.reverse_xy()
    association_reversed = quanti_quali_measure.compute_association(quali_series_data, series_data)
    assert association == association_reversed, "not same correlation when reversed"


def test_kruskal_effect_sizes_formula(series_data: pd.Series, quali_series_data: pd.Series) -> None:
    """checks epsilon-squared and eta-squared match their closed forms and are bounded"""

    # raw Kruskal-Wallis H and sample/group sizes (no nans here)
    h = kruskal(*(series_data[quali_series_data == v] for v in quali_series_data.unique()))[0]
    n_obs = len(series_data)
    n_groups = quali_series_data.nunique()

    # epsilon-squared = H / (N - 1)
    eps2 = KruskalEpsilonSquaredMeasure().compute_association(series_data, quali_series_data)
    assert eps2 == h / (n_obs - 1), "epsilon-squared does not match H / (N - 1)"
    assert 0.0 <= eps2 <= 1.0, "epsilon-squared out of [0, 1]"

    # eta-squared = (H - k + 1) / (N - k)
    eta2 = KruskalEtaSquaredMeasure().compute_association(series_data, quali_series_data)
    assert eta2 == max(0.0, (h - n_groups + 1) / (n_obs - n_groups)), "eta-squared does not match closed form"
    assert 0.0 <= eta2 <= 1.0, "eta-squared out of [0, 1]"


def test_quanti_quanti_compute_association(quanti_quanti_measure: BaseMeasure, series_data: pd.Series) -> None:
    """checks that correlation measurement works"""

    # without nans
    association = quanti_quanti_measure.compute_association(series_data, series_data)
    assert association is not None, "not correctly computed association"
    assert quanti_quanti_measure.value == association, "not stored measurement"

    # with nans
    association = quanti_quanti_measure.compute_association(series_data.replace(1, np.nan), series_data)
    assert association is not None, "not correctly computed association"
    assert quanti_quanti_measure.value == association, "not stored measurement"


def test_outlier_validate_with_computed_association_below_threshold(
    outlier_measure: BaseMeasure, series_data: pd.Series
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
    outlier_measure.compute_association(series_data.replace(1, np.nan))
    print(
        outlier_measure.__name__,
        outlier_measure.value,
        outlier_measure.threshold,
        outlier_measure.validate(),
    )
    assert outlier_measure.validate(), "kept feature with lower than threshold correlation"


def test_quanti_quali_validate_with_computed_association_below_threshold(
    quanti_quali_measure: BaseMeasure, quali_series_data: pd.Series, series_data: pd.Series
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
    quanti_quali_measure.compute_association(series_data.replace(1, np.nan), quali_series_data)
    print(
        quanti_quali_measure.__name__,
        quanti_quali_measure.value,
        quanti_quali_measure.threshold,
        quanti_quali_measure.validate(),
    )
    assert not quanti_quali_measure.validate(), "kept feature with lower than threshold correlation"


def test_quanti_quali_validate_with_computed_association_above_threshold(
    quanti_quali_measure: BaseMeasure, quali_series_data: pd.Series, series_data: pd.Series
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
    quanti_quali_measure.compute_association(series_data.replace(1, np.nan), quali_series_data)
    print(
        quanti_quali_measure.__name__,
        quanti_quali_measure.value,
        quanti_quali_measure.threshold,
        quanti_quali_measure.validate(),
    )
    assert quanti_quali_measure.validate(), "removed feature with lower than threshold correlation"


def test_quanti_quanti_validate_with_computed_association_below_threshold(
    quanti_quanti_measure: BaseMeasure, series_data: pd.Series
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
    assert not quanti_quanti_measure.validate(), "kept feature with lower than threshold correlation"

    # with nans
    quanti_quanti_measure.compute_association(series_data.replace(1, np.nan), (series_data <= 2).astype(int))
    print(
        quanti_quanti_measure.__name__,
        quanti_quanti_measure.value,
        quanti_quanti_measure.threshold,
        quanti_quanti_measure.validate(),
    )
    assert not quanti_quanti_measure.validate(), "kept feature with lower than threshold correlation"

    # test with negative value
    quanti_quanti_measure.value = -0.5
    quanti_quanti_measure.thresold = 0.6
    assert not quanti_quanti_measure.validate()


def test_quanti_quanti_validate_with_computed_association_above_threshold(
    quanti_quanti_measure: BaseMeasure, series_data: pd.Series
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
    quanti_quanti_measure.compute_association(series_data.replace(1, np.nan), series_data**20)
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
    quanti_binary_measure: BaseMeasure, binary_series_data: pd.Series, series_data: pd.Series
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
    assert not quanti_binary_measure.validate(), "kept feature with lower than threshold correlation"

    # with nans
    quanti_binary_measure.compute_association(series_data.replace(1, np.nan), binary_series_data)
    print(
        quanti_binary_measure.__name__,
        quanti_binary_measure.value,
        quanti_binary_measure.threshold,
        quanti_binary_measure.validate(),
    )
    assert not quanti_binary_measure.validate(), "kept feature with lower than threshold correlation"


def test_quanti_binary_validate_with_computed_association_above_threshold(
    quanti_binary_measure: BaseMeasure, binary_series_data: pd.Series, series_data: pd.Series
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
    quanti_binary_measure.compute_association(series_data.replace(1, np.nan), binary_series_data)
    print(
        quanti_binary_measure.__name__,
        quanti_binary_measure.value,
        quanti_binary_measure.threshold,
        quanti_binary_measure.validate(),
    )
    assert quanti_binary_measure.validate(), "removed feature with lower than threshold correlation"


def test_outlier_validate_with_computed_association_above_threshold(
    outlier_measure: BaseMeasure, outlier_series_data: pd.Series
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
    outlier_measure.compute_association(outlier_series_data.replace(50, np.nan).replace(51, 5e50))
    print(
        outlier_measure.__name__,
        outlier_measure.value,
        outlier_measure.threshold,
        outlier_measure.validate(),
    )
    assert not outlier_measure.validate(), "kept feature with lower than threshold correlation"


def _naive_distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """O(n²) reference: double-centred distance matrices (Székely et al., 2007)."""
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    a = a - a.mean(axis=0) - a.mean(axis=1)[:, None] + a.mean()
    b = b - b.mean(axis=0) - b.mean(axis=1)[:, None] + b.mean()
    return float(np.sqrt((a * b).mean() / np.sqrt((a * a).mean() * (b * b).mean())))


@needs_dcor
def test_distance_measure_detects_non_linear_dependence() -> None:
    """y = x² is invisible to Pearson but not to distance correlation"""
    z = np.random.default_rng(0).normal(size=500)
    x = pd.Series(np.concatenate([z, -z]))  # exactly symmetric, so Pearson's r is 0
    y = x**2
    assert abs(PearsonMeasure().compute_association(x, y)) < 0.1
    assert DistanceMeasure().compute_association(x, y) > 0.4


@needs_dcor
def test_distance_measure_linear_dependence() -> None:
    """an affine relation gives a distance correlation of 1"""
    x = pd.Series(np.random.default_rng(0).normal(size=1000))
    assert DistanceMeasure().compute_association(x, 2 * x + 1) > 0.99


@needs_dcor
def test_distance_measure_independent() -> None:
    """independent draws give a distance correlation close to 0"""
    rng = np.random.default_rng(0)
    x, y = pd.Series(rng.normal(size=1000)), pd.Series(rng.normal(size=1000))
    assert DistanceMeasure().compute_association(x, y) < 0.1


@needs_dcor
def test_distance_measure_constant_target() -> None:
    """a constant y has no distance variance: distance correlation is 0"""
    x = pd.Series(np.random.default_rng(0).normal(size=100))
    assert DistanceMeasure().compute_association(x, pd.Series(np.ones(100))) == 0.0


@needs_dcor
def test_distance_measure_uses_every_row() -> None:
    """no subsampling on large inputs: the value equals dcor's other exact algorithm (AVL) on all rows"""
    import dcor

    rng = np.random.default_rng(0)
    x = rng.normal(size=12_000)
    y = x**2 + rng.normal(size=12_000)
    value = DistanceMeasure().compute_association(pd.Series(x), pd.Series(y))
    assert value == approx(dcor.distance_correlation(x, y, method="avl"), abs=1e-10)


@needs_dcor
def test_distance_measure_matches_naive_reference() -> None:
    """the O(n log n) dcor algorithm equals the naive double-centring (ties included)"""
    rng = np.random.default_rng(0)
    x = rng.normal(size=300)
    y = np.round(x**2 + rng.normal(size=300), 1)
    measure = DistanceMeasure()
    assert measure.compute_association(pd.Series(x), pd.Series(y)) == approx(
        _naive_distance_correlation(x, y), abs=1e-10
    )


def test_distance_measure_without_dcor_names_the_extra(monkeypatch) -> None:
    """without the dcor package, the error says which extra to install"""
    import sys

    monkeypatch.setitem(sys.modules, "dcor", None)  # makes `import dcor` raise ImportError
    x = pd.Series([1.0, 2.0, 3.0])
    with raises(ImportError, match=r"autocarver\[dcor\]"):
        DistanceMeasure().compute_association(x, x)
