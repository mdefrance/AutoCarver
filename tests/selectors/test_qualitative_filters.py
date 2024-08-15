from pytest import fixture, FixtureRequest
from pandas import DataFrame, isna
from AutoCarver.selectors import QualitativeFilter, CramervFilter, TschuprowtFilter
from AutoCarver.features import BaseFeature

THRESHOLD = 1.0


@fixture
def sample_data() -> DataFrame:
    data = {
        "feature1": [0, 1, 0, 1, 0, 1, 0],
        "feature2": [2, 0, 2, 0, 2, 0, 2],
        "feature3": [-3, 0, -3, 0, -3, -3, -3],
        "feature4": [0, 1, 2, 0, 1, 2, 0],
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


@fixture(params=[CramervFilter, TschuprowtFilter])
def filter(request: FixtureRequest) -> QualitativeFilter:
    return request.param(THRESHOLD)


def test_qualitative_filter(
    filter: QualitativeFilter, sample_data: DataFrame, sample_ranks: list[BaseFeature]
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


def test_filter(
    filter: QualitativeFilter, sample_data: DataFrame, sample_ranks: list[BaseFeature]
) -> None:

    # testing _compute_worst_correlation
    correlation_with, worst_correlation = filter._compute_worst_correlation(
        sample_data, sample_ranks[1], sample_ranks
    )
    assert isinstance(worst_correlation, float)
    assert isinstance(correlation_with, str)

    # testing _validate
    filter.threshold = 0.5
    valid = filter._validate(0.6)
    assert valid == False
