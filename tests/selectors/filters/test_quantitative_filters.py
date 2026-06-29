import numpy as np
import pandas as pd
from pytest import FixtureRequest, fixture

from AutoCarver.features import BaseFeature
from AutoCarver.selectors import PearsonFilter, QuantitativeFilter, SpearmanFilter

THRESHOLD = 1.0


def test_spearman_correlation_matches_pandas_no_nan() -> None:
    """rank-once + Pearson must equal pandas corr('spearman') exactly when no NaN."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({f"q{j}": rng.normal(size=300) for j in range(6)})
    X["q_dup"] = X["q0"]  # a perfectly correlated column
    ranks = [BaseFeature(c) for c in X.columns]

    got = SpearmanFilter(THRESHOLD)._compute_correlation(X, ranks)
    versions = [feature.version for feature in ranks]
    ref = X[versions].corr("spearman").where(np.triu(np.ones((len(versions),) * 2), k=1).astype(bool))
    pd.testing.assert_frame_equal(got, ref, atol=1e-9)


@fixture
def sample_data() -> pd.DataFrame:
    data = {
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [2, 3, 4, 5, 6],
        "feature3": [-2, -3, -4, -5, -6],
        "feature4": [-2, 10, -1, 30, 0],
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


@fixture(params=[SpearmanFilter, PearsonFilter])
def filter(request: FixtureRequest) -> QuantitativeFilter:
    return request.param(THRESHOLD)


def test_n_best_early_stop_matches_prefix() -> None:
    """``filter(..., n_best=k)`` returns the first ``k`` of the full filtered list."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame({f"q{j}": rng.normal(size=300) for j in range(8)})
    X["q_dup0"], X["q_dup1"] = X["q0"], X["q1"]  # correlated -> forces some drops
    ranks = [BaseFeature(c) for c in X.columns]

    for filter_cls in (SpearmanFilter, PearsonFilter):
        full = filter_cls(0.9).filter(X, ranks)
        for k in (1, 2, 3):
            stopped = filter_cls(0.9).filter(X, ranks, n_best=k)
            assert stopped == full[:k], (filter_cls.__name__, k)


def test_quantitative_filter(
    filter: QuantitativeFilter, sample_data: pd.DataFrame, sample_ranks: list[BaseFeature]
) -> None:
    # testing type
    assert filter.is_x_quantitative, "x should be quantitative"
    assert not filter.is_x_qualitative, "x should be quantitative"
    assert not filter.is_default, "should not be default"

    # test with negatively and 100% correlated features
    filter.threshold = 0.9
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


def test_filter(filter: QuantitativeFilter, sample_data: pd.DataFrame, sample_ranks: list[BaseFeature]) -> None:
    assert isinstance(filter.measure, str), "measure should be a name for DataFrame.corr"

    # testing _compute_correlation
    correlation = filter._compute_correlation(sample_data, sample_ranks)
    assert isinstance(correlation, pd.DataFrame), "Correlation should be a dataframe"
    assert correlation.shape == (
        len(sample_ranks),
        len(sample_ranks),
    ), "Compute correlation for each feature"
    assert pd.isna(correlation.iloc[0, 0]) and pd.isna(correlation.iloc[-1, -1]), "no autocorrelation"
    assert all(correlation >= 0), "should be positive"

    # testing _filter_correlated_features
    filtered_features = filter._filter_correlated_features(correlation, sample_ranks)
    assert isinstance(filtered_features, list), "should be a list"
    assert all(isinstance(feature, BaseFeature) for feature in filtered_features), "should be BaseFeature"

    # testing _compute_worst_correlation
    feature = sample_ranks[1]
    correlation_with, worst_correlation = filter._compute_worst_correlation(correlation, feature)
    assert isinstance(worst_correlation, float)
    assert isinstance(correlation_with, str)
    print(correlation)
    assert correlation.loc[correlation_with, feature.version] == worst_correlation

    # testing _validate
    feature = sample_ranks[0]
    filter.threshold = 0.5
    valid = filter._validate(feature, 0.6, "feature3")
    assert valid is False
    assert feature.filters[filter.__name__]["valid"] is False
