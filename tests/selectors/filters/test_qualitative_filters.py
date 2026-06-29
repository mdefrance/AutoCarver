import numpy as np
import pandas as pd
from pytest import FixtureRequest, approx, fixture

from AutoCarver.features import BaseFeature
from AutoCarver.selectors import (
    BaseMeasure,
    CramervFilter,
    CramervMeasure,
    QualitativeFilter,
    TschuprowtFilter,
    TschuprowtMeasure,
)

THRESHOLD = 1.0


@fixture
def sample_data() -> pd.DataFrame:
    data = {
        "feature1": [0, 1, 0, 1, 0, 1, 0],
        "feature2": [2, 0, 2, 0, 2, 0, 2],
        "feature3": [-3, 0, -3, 0, -3, -3, -3],
        "feature4": [0, 1, 2, 0, 1, 2, 0],
    }
    return pd.DataFrame(data)


@fixture
def sample_ranks() -> list[BaseFeature]:
    return [
        BaseFeature("feature2"),
        BaseFeature("feature3"),
        BaseFeature("feature1"),
        BaseFeature("feature4"),
    ]


@fixture(params=[CramervFilter, TschuprowtFilter])
def filter(request: FixtureRequest) -> QualitativeFilter:
    return request.param(THRESHOLD)


def test_qualitative_filter(
    filter: QualitativeFilter, sample_data: pd.DataFrame, sample_ranks: list[BaseFeature]
) -> None:
    # test with negatively and 100% correlated features
    filter.threshold = 0.3
    filtered_features = filter.filter(sample_data, sample_ranks)
    print([feature.statistics for feature in filtered_features])
    assert len(filtered_features) == 2, "not filtered too correlated features"
    assert filtered_features == [sample_ranks[0], sample_ranks[-1]], "Should conserve the ranking"

    # testing turning out filter
    filter.threshold = 1.0
    filtered_features = filter.filter(sample_data, sample_ranks)
    print([feature.statistics for feature in filtered_features])
    assert len(filtered_features) == len(sample_ranks), "filtered some features"
    assert filtered_features == sample_ranks, "Should conserve the ranking"

    # testing filtering out all features
    filter.threshold = 0.0
    filtered_features = filter.filter(sample_data, sample_ranks)
    print([feature.statistics for feature in filtered_features])
    assert len(filtered_features) == 1, "sould keep the best feature"


@fixture(params=[(CramervFilter, CramervMeasure), (TschuprowtFilter, TschuprowtMeasure)])
def filter_measure(request: FixtureRequest) -> tuple:
    return request.param


def test_pairwise_association_parity(filter_measure: tuple) -> None:
    """Vectorized ``_pairwise_association`` must equal scalar ``compute_association``.

    Covers NaN columns, heavy ties, a single-modality column and 2x2 pairs (Yates).
    """
    filter_cls, measure_cls = filter_measure
    rng = np.random.default_rng(0)
    X = pd.DataFrame({f"c{j}": rng.integers(0, 4, 600) for j in range(5)}).astype(object)
    X["c_bin"] = rng.integers(0, 2, 600).astype(object)  # 2x2 pairs -> Yates path
    X.loc[rng.random(600) < 0.2, "c2"] = np.nan
    X["c_const"] = "z"

    filt = filter_cls(THRESHOLD)
    filt._codes_cache = {}
    scalar = measure_cls(THRESHOLD)

    for a in X.columns:
        for b in X.columns:
            if a == b:
                continue
            vec = filt._pairwise_association(X, a, b)
            ref = scalar.compute_association(X[a], X[b])
            if ref is None or (isinstance(ref, float) and np.isnan(ref)):
                assert vec is None or np.isnan(vec), (a, b)
            else:
                assert vec == approx(ref, rel=1e-9, abs=1e-9), (a, b)


def test_n_best_early_stop_matches_prefix(filter_measure: tuple) -> None:
    """``filter(..., n_best=k)`` returns the first ``k`` of the full filtered list."""
    filter_cls, _ = filter_measure
    rng = np.random.default_rng(1)
    X = pd.DataFrame({f"c{j}": rng.integers(0, 3, 400) for j in range(8)}).astype(object)
    ranks = [BaseFeature(c) for c in X.columns]

    full = filter_cls(0.9).filter(X, ranks)
    for k in (1, 2, 3):
        stopped = filter_cls(0.9).filter(X, ranks, n_best=k)
        assert stopped == full[:k], k


def test_filter(filter: QualitativeFilter, sample_data: pd.DataFrame, sample_ranks: list[BaseFeature]) -> None:
    # testing type
    assert filter.is_x_qualitative, "x should be qulitative"
    assert not filter.is_x_quantitative, "x should be qualitative"
    assert not filter.is_default, "should not be default"

    assert isinstance(filter.measure, BaseMeasure), "measure should be a BaseMeasure"

    # testing _compute_worst_correlation
    correlation_with, worst_correlation = filter._compute_worst_correlation(sample_data, sample_ranks[1], sample_ranks)
    assert isinstance(worst_correlation, float)
    assert isinstance(correlation_with, str)

    # testing _validate
    filter.threshold = 0.5
    valid = filter._validate(0.6)
    assert valid is False
