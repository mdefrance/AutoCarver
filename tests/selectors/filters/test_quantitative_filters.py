from pytest import fixture, FixtureRequest
from pandas import DataFrame, isna
from AutoCarver.selectors import QuantitativeFilter, SpearmanFilter, PearsonFilter
from AutoCarver.features import BaseFeature

THRESHOLD = 1.0


@fixture
def sample_data() -> DataFrame:
    data = {
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [2, 3, 4, 5, 6],
        "feature3": [-2, -3, -4, -5, -6],
        "feature4": [-2, 10, -1, 30, 0],
    }
    return DataFrame(data)


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


def test_quantitative_filter(
    filter: QuantitativeFilter, sample_data: DataFrame, sample_ranks: list[BaseFeature]
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


def test_filter(
    filter: QuantitativeFilter, sample_data: DataFrame, sample_ranks: list[BaseFeature]
) -> None:

    assert isinstance(filter.measure, str), "measure should be a name for DataFrame.corr"

    # testing _compute_correlation
    correlation = filter._compute_correlation(sample_data, sample_ranks)
    assert isinstance(correlation, DataFrame), "Correlation should be a dataframe"
    assert correlation.shape == (
        len(sample_ranks),
        len(sample_ranks),
    ), "Compute correlation for each feature"
    assert isna(correlation.iloc[0, 0]) and isna(correlation.iloc[-1, -1]), "no autocorrelation"
    assert all(correlation >= 0), "should be positive"

    # testing _filter_correlated_features
    filtered_features = filter._filter_correlated_features(correlation, sample_ranks)
    assert isinstance(filtered_features, list), "should be a list"
    assert all(
        isinstance(feature, BaseFeature) for feature in filtered_features
    ), "should be BaseFeature"

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
    assert valid == False
    assert feature.statistics["filters"][filter.__name__]["valid"] == False
