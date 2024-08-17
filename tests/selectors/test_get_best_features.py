import pytest
from pandas import DataFrame, Series
from unittest.mock import Mock
from AutoCarver.selectors import BaseFilter, BaseMeasure
from AutoCarver.features import BaseFeature
from AutoCarver.selectors.base_selector import get_best_features


def test_get_best_features_all_measures_sortable():
    features = [
        Mock(spec=BaseFeature, version="feature1", statistics={}),
        Mock(spec=BaseFeature, version="feature2", statistics={}),
    ]
    X = DataFrame([[1, 2], [3, 4]])
    y = Series([0, 1])
    measures = [Mock(spec=BaseMeasure, is_sortable=True), Mock(spec=BaseMeasure, is_sortable=True)]
    filters = [Mock(spec=BaseFilter)]
    n_best = 1

    result = get_best_features(features, X, y, measures, filters, n_best)

    assert isinstance(result, list)
    assert len(result) <= len(features)


# def test_get_best_features_measure_not_sortable():
#     mock_feature = Mock(spec=BaseFeature)
#     features = [mock_feature]
#     X = DataFrame([[1, 2]])
#     y = Series([0])
#     measures = [Mock(spec=BaseMeasure, is_sortable=False)]
#     filters = []

#     with pytest.raises(ValueError, match="All provided measures should be sortable"):
#         get_best_features(features, X, y, measures, filters, n_best=1)


# def test_get_best_features_no_measures():
#     mock_feature = Mock(spec=BaseFeature)
#     features = [mock_feature]
#     X = DataFrame([[1, 2]])
#     y = Series([0])
#     measures = []
#     filters = []
#     n_best = 1

#     result = get_best_features(features, X, y, measures, filters, n_best)

#     assert result == []


# def test_get_best_features_apply_filters():
#     mock_feature = Mock(spec=BaseFeature)
#     features = [mock_feature, mock_feature]
#     X = DataFrame([[1, 2], [3, 4]])
#     y = Series([0, 1])
#     measures = [Mock(spec=BaseMeasure, is_sortable=True)]
#     filters = [Mock(spec=BaseFilter)]
#     n_best = 1

#     filtered_mock = Mock(spec=BaseFeature)
#     apply_filters = Mock(return_value=[filtered_mock])
#     result = get_best_features(features, X, y, measures, filters, n_best)

#     assert filtered_mock in result


# def test_get_best_features_deduplication():
#     mock_feature = Mock(spec=BaseFeature)
#     features = [mock_feature, mock_feature]
#     X = DataFrame([[1, 2], [3, 4]])
#     y = Series([0, 1])
#     measures = [Mock(spec=BaseMeasure, is_sortable=True)]
#     filters = [Mock(spec=BaseFilter)]
#     n_best = 2

#     remove_duplicates = Mock(return_value=features)
#     result = get_best_features(features, X, y, measures, filters, n_best)

#     assert len(result) == len(set(result))
