"""Set of tests for RegressionSelector module."""

from pytest import raises

from AutoCarver.features import Features
from AutoCarver.selectors import RegressionSelector
from AutoCarver.selectors.filters import BaseFilter, NonDefaultValidFilter, ValidFilter
from AutoCarver.selectors.measures import BaseMeasure, ModeMeasure, NanMeasure
from AutoCarver.selectors.utils.base_selector import get_default_metrics, remove_default_metrics


def test_regression_selector_initiate_default(features_object: Features) -> None:
    """tests initiation of default measures and filters"""
    # checking for default measures
    n_best, max_num_features_per_chunk = 2, 100
    selector = RegressionSelector(
        n_best_per_type=n_best,
        features=features_object,
        max_num_features_per_chunk=max_num_features_per_chunk,
    )

    mode_measure = ModeMeasure()
    assert any(measure.__name__ == mode_measure.__name__ for measure in selector.measures)
    nan_measure = NanMeasure()
    assert any(measure.__name__ == nan_measure.__name__ for measure in selector.measures)
    valid_filter = NonDefaultValidFilter()
    assert any(measure.__name__ == valid_filter.__name__ for measure in selector.filters)
    valid_filter = ValidFilter()
    assert any(measure.__name__ == valid_filter.__name__ for measure in selector.filters)
    assert len(remove_default_metrics(selector.measures)) >= 1
    assert len(remove_default_metrics(selector.filters)) >= 1


def test_regression_selector_initiate_measures(features_object: Features, measures: list[BaseMeasure]) -> None:
    """tests initiation of measures"""

    n_best, max_num_features_per_chunk = 2, 100

    # adding default measure
    default_measures = get_default_metrics(measures)
    if len(default_measures) > 0:
        selector = RegressionSelector(
            n_best_per_type=n_best,
            features=features_object,
            max_num_features_per_chunk=max_num_features_per_chunk,
            measures=default_measures,
        )
        mode_measure = ModeMeasure()
        assert any(measure.__name__ == mode_measure.__name__ for measure in selector.measures)
        nan_measure = NanMeasure()
        assert any(measure.__name__ == nan_measure.__name__ for measure in selector.measures)
        assert len(selector.measures) == 2 + sum(
            not isinstance(measure, (NanMeasure, ModeMeasure)) for measure in default_measures
        )

    # adding qualitative target measures
    regression_measures = [
        measure
        for measure in remove_default_metrics(measures)
        if measure.is_y_quantitative or (measure.reverse_xy() and measure.is_y_quantitative)
    ]
    if len(regression_measures) > 0:
        selector = RegressionSelector(
            n_best_per_type=n_best,
            features=features_object,
            max_num_features_per_chunk=max_num_features_per_chunk,
            measures=regression_measures,
        )
        mode_measure = ModeMeasure()
        assert any(measure.__name__ == mode_measure.__name__ for measure in selector.measures)
        nan_measure = NanMeasure()
        assert any(measure.__name__ == nan_measure.__name__ for measure in selector.measures)
        assert len(selector.measures) == len(regression_measures) + 2

    # checking error for quantitative target measures
    classification_measures = [
        measure
        for measure in remove_default_metrics(measures)
        if not (measure.is_y_quantitative or (measure.reverse_xy() and measure.is_y_quantitative))
    ]
    if len(classification_measures) > 0:
        with raises(ValueError):
            selector = RegressionSelector(
                n_best_per_type=n_best,
                features=features_object,
                max_num_features_per_chunk=max_num_features_per_chunk,
                measures=classification_measures,
            )


def test_regression_selector_initiate_filters(features_object: Features, filters: list[BaseFilter]) -> None:
    """tests initiation of filters"""

    # checking for default filters
    n_best, max_num_features_per_chunk = 2, 100

    # adding default filter
    default_filters = get_default_metrics(filters)
    if len(default_filters) > 0:
        selector = RegressionSelector(
            n_best_per_type=n_best,
            features=features_object,
            max_num_features_per_chunk=max_num_features_per_chunk,
            filters=default_filters,
        )
        valid_filter = NonDefaultValidFilter()
        assert any(measure.__name__ == valid_filter.__name__ for measure in selector.filters)
        valid_filter = ValidFilter()
        assert any(measure.__name__ == valid_filter.__name__ for measure in selector.filters)
        assert (
            len(selector.filters)
            == len([filter_ for filter_ in default_filters if filter_.__name__ not in [ValidFilter.__name__]]) + 1
        )

    # adding filters
    filters = remove_default_metrics(filters)
    if len(filters) > 0:
        selector = RegressionSelector(
            n_best_per_type=n_best,
            features=features_object,
            max_num_features_per_chunk=max_num_features_per_chunk,
            filters=filters,
        )
        valid_filter = NonDefaultValidFilter()
        assert any(measure.__name__ == valid_filter.__name__ for measure in selector.filters)
        valid_filter = ValidFilter()
        assert any(measure.__name__ == valid_filter.__name__ for measure in selector.filters)
        assert len(selector.filters) == len(filters) + 2
