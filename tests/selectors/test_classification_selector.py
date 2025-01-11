"""Set of tests for ClassificationSelector module."""

from pytest import raises

from AutoCarver.features import Features
from AutoCarver.selectors import ClassificationSelector
from AutoCarver.selectors.base_selector import get_default_metrics, remove_default_metrics
from AutoCarver.selectors.filters import BaseFilter
from AutoCarver.selectors.measures import BaseMeasure


def test_classification_selector_initiate_default(features_object: Features) -> None:
    """tests initiation of default measures and filters"""
    # checking for default measures
    n_best, max_num_features_per_chunk = 2, 100
    selector = ClassificationSelector(
        n_best=n_best,
        features=features_object,
        max_num_features_per_chunk=max_num_features_per_chunk,
    )

    assert any(measure.__name__ == "Mode" for measure in selector.measures)
    assert any(measure.__name__ == "NaN" for measure in selector.measures)
    assert any(filter_.__name__ == "Valid" for filter_ in selector.filters)
    assert len(remove_default_metrics(selector.measures)) >= 1
    assert len(remove_default_metrics(selector.filters)) >= 1


def test_classification_selector_initiate_measures(
    features_object: Features, measures: list[BaseMeasure]
) -> None:
    """tests initiation of measures"""

    n_best, max_num_features_per_chunk = 2, 100

    # adding default measure
    default_measures = get_default_metrics(measures)
    if len(default_measures) > 0:
        selector = ClassificationSelector(
            n_best=n_best,
            features=features_object,
            max_num_features_per_chunk=max_num_features_per_chunk,
            measures=default_measures,
        )
        assert any(measure.__name__ == "Mode" for measure in selector.measures)
        assert any(measure.__name__ == "NaN" for measure in selector.measures)
        assert (
            len(selector.measures)
            == len(
                [measure for measure in default_measures if measure.__name__ not in ["Mode", "NaN"]]
            )
            + 2
        )

    # adding qualitative target measures
    classification_measures = [
        measure
        for measure in remove_default_metrics(measures)
        if measure.is_y_qualitative or (measure.reverse_xy() and measure.is_y_qualitative)
    ]
    if len(classification_measures) > 0:
        selector = ClassificationSelector(
            n_best=n_best,
            features=features_object,
            max_num_features_per_chunk=max_num_features_per_chunk,
            measures=classification_measures,
        )
        assert any(measure.__name__ == "Mode" for measure in selector.measures)
        assert any(measure.__name__ == "NaN" for measure in selector.measures)
        assert len(selector.measures) == len(classification_measures) + 2

    # checking error for quantitative target measures
    regression_measures = [
        measure
        for measure in remove_default_metrics(measures)
        if not (measure.is_y_qualitative or (measure.reverse_xy() and measure.is_y_qualitative))
    ]
    if len(regression_measures) > 0:
        with raises(ValueError):
            selector = ClassificationSelector(
                n_best=n_best,
                features=features_object,
                max_num_features_per_chunk=max_num_features_per_chunk,
                measures=regression_measures,
            )


def test_classification_selector_initiate_filters(
    features_object: Features, filters: list[BaseFilter]
) -> None:
    """tests initiation of filters"""

    # checking for default filters
    n_best, max_num_features_per_chunk = 2, 100

    # adding default filter
    default_filters = get_default_metrics(filters)
    if len(default_filters) > 0:
        selector = ClassificationSelector(
            n_best=n_best,
            features=features_object,
            max_num_features_per_chunk=max_num_features_per_chunk,
            filters=default_filters,
        )
        assert any(filter_.__name__ == "Valid" for filter_ in selector.filters)
        assert (
            len(selector.filters)
            == len([filter_ for filter_ in default_filters if filter_.__name__ not in ["Valid"]])
            + 1
        )

    # adding filters
    filters = remove_default_metrics(filters)
    if len(filters) > 0:
        selector = ClassificationSelector(
            n_best=n_best,
            features=features_object,
            max_num_features_per_chunk=max_num_features_per_chunk,
            filters=filters,
        )
        assert any(filter_.__name__ == "Valid" for filter_ in selector.filters)
        assert len(selector.filters) == len(filters) + 1
