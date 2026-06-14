"""set of tests for base selector"""

import pandas as pd
from pytest import approx, raises

from AutoCarver.features import (
    BaseFeature,
    Features,
    get_qualitative_features,
    get_quantitative_features,
)
from AutoCarver.selectors import BaseFilter, BaseMeasure
from AutoCarver.selectors.utils.base_selector import (
    BaseSelector,
    apply_filters,
    apply_measures,
    get_best_features,
    get_default_metrics,
    get_qualitative_metrics,
    get_quantitative_metrics,
    remove_default_metrics,
    remove_duplicates,
    sort_features_per_measure,
)


def test_get_quantitative_metrics(
    quanti_measure: BaseMeasure,
    quali_measure: BaseMeasure,
    default_measure: BaseMeasure,
    quanti_filter: BaseFilter,
    quali_filter: BaseFilter,
    default_filter: BaseFilter,
) -> None:
    """function test of get_quantitative_metrics"""

    # test with measures
    measures = [quanti_measure, quali_measure, default_measure]
    quantitative_measures = get_quantitative_metrics(measures)
    assert len(quantitative_measures) == 2
    assert quantitative_measures[0] == quanti_measure
    assert quantitative_measures[1] == default_measure

    # test with filters
    filters = [quanti_filter, quali_filter, default_filter]
    quantitative_filters = get_quantitative_metrics(filters)
    assert len(quantitative_filters) == 2
    assert quantitative_filters[0] == quanti_filter
    assert quantitative_filters[1] == default_filter


def test_get_qualitative_metrics(
    quanti_measure: BaseMeasure,
    quali_measure: BaseMeasure,
    default_measure: BaseMeasure,
    quanti_filter: BaseFilter,
    quali_filter: BaseFilter,
    default_filter: BaseFilter,
) -> None:
    """function test of get_qualitative_metrics"""

    # test with measures
    measures = [quanti_measure, quali_measure, default_measure]
    qualitative_measures = get_qualitative_metrics(measures)
    assert len(qualitative_measures) >= 1
    assert qualitative_measures[0] == quali_measure
    # checking for outlier measures that are default but only for quantitatives
    if len(qualitative_measures) == 2:
        assert qualitative_measures[1] == default_measure

    # test with filters
    filters = [quanti_filter, quali_filter, default_filter]
    qualitative_filters = get_qualitative_metrics(filters)
    assert len(qualitative_filters) == 2
    assert qualitative_filters[0] == quali_filter
    assert qualitative_filters[1] == default_filter


def test_get_default_metrics(
    quanti_measure: BaseMeasure,
    quali_measure: BaseMeasure,
    default_measure: BaseMeasure,
    quanti_filter: BaseFilter,
    quali_filter: BaseFilter,
    default_filter: BaseFilter,
) -> None:
    """function test of get_default_metrics"""

    # test with measures
    measures = [quanti_measure, quali_measure, default_measure]
    default_measures = get_default_metrics(measures)
    assert len(default_measures) <= 1
    # checking for outlier measures that are default but only for quantitatives
    if len(default_measures) == 1:
        assert default_measures[0] == default_measure

    # test with filters
    filters = [quanti_filter, quali_filter, default_filter]
    default_filters = get_default_metrics(filters)
    assert len(default_filters) == 1
    assert default_filters[0] == default_filter


def test_remove_default_metrics(
    quanti_measure: BaseMeasure,
    quali_measure: BaseMeasure,
    default_measure: BaseMeasure,
    quanti_filter: BaseFilter,
    quali_filter: BaseFilter,
    default_filter: BaseFilter,
) -> None:
    """function test of remove_default_metrics"""

    # test with measures
    measures = [quanti_measure, quali_measure, default_measure]
    nondefault_measures = remove_default_metrics(measures)
    assert len(nondefault_measures) == 2
    assert nondefault_measures[0] == quanti_measure
    assert nondefault_measures[1] == quali_measure

    # test with filters
    filters = [quanti_filter, quali_filter, default_filter]
    nondefault_filters = remove_default_metrics(filters)
    assert len(nondefault_filters) == 2
    assert nondefault_filters[0] == quanti_filter
    assert nondefault_filters[1] == quali_filter


def test_remove_duplicates() -> None:
    """function test of remove_duplicates"""
    feature1 = BaseFeature("feature1")
    feature2 = BaseFeature("feature2")
    feature3 = feature1
    features = [feature1, feature2, feature3]
    unique_features = remove_duplicates(features)
    assert len(unique_features) == 2
    assert unique_features[0] == feature1
    assert unique_features[1] == feature2


def test_sort_features_per_measure(measure: BaseMeasure) -> None:
    """function test of sort_features_per_measure"""
    feature1 = BaseFeature("feature1")
    feature1.measures = {measure.__name__: {"value": 0.5}}
    feature2 = BaseFeature("feature2")
    feature2.measures = {measure.__name__: {"value": 0.2}}
    feature3 = BaseFeature("feature3")
    feature3.measures = {measure.__name__: {"value": 0.8}}
    features = [feature1, feature2, feature3]
    sorted_features = sort_features_per_measure(features, measure)
    assert sorted_features[0] == feature2
    assert sorted_features[1] == feature1
    assert sorted_features[2] == feature3


def test_apply_measures(
    features: list[BaseFeature], X: pd.DataFrame, y: pd.Series, measures: list[BaseMeasure]
) -> None:
    """testing function apply_measures

    The batched ``compute_all`` path may differ from the scalar
    ``compute_association`` in the last floating-point digits, so values are
    compared with ``approx`` (parity itself is asserted exactly elsewhere, in
    ``test_vectorized_parity.py``).
    """

    # sorting out measures
    quantitative_measures = get_quantitative_metrics(measures)
    qualitative_measures = get_qualitative_metrics(measures)

    # sorting out features
    qualitative_features = get_qualitative_features(features)
    quantitative_features = get_quantitative_features(features)

    # applying qualitative measures
    apply_measures(qualitative_features, X, y, qualitative_measures, default_measures=True)
    apply_measures(qualitative_features, X, y, qualitative_measures, default_measures=False)
    for feature in qualitative_features:
        for measure in qualitative_measures:
            assert measure.compute_association(X[feature.version], y) == approx(
                feature.measures.get(measure.__name__).get("value"), rel=1e-9, abs=1e-9, nan_ok=True
            )

    # applying quantitative measures
    apply_measures(quantitative_features, X, y, quantitative_measures, default_measures=True)
    apply_measures(quantitative_features, X, y, quantitative_measures, default_measures=False)
    for feature in quantitative_features:
        for measure in quantitative_measures:
            assert measure.compute_association(X[feature.version], y) == approx(
                feature.measures.get(measure.__name__).get("value"), rel=1e-9, abs=1e-9, nan_ok=True
            )

    # type mismatch
    with raises(TypeError):
        apply_measures(quantitative_features, X, y, qualitative_measures, default_measures=False)
    with raises(TypeError):
        apply_measures(qualitative_features, X, y, quantitative_measures, default_measures=False)


def test_apply_filters(features: list[BaseFeature], X: pd.DataFrame, filters: list[BaseFilter]) -> None:
    """testing function apply_filters"""

    # sorting out filters
    quantitative_filters = get_quantitative_metrics(filters)
    qualitative_filters = get_qualitative_metrics(filters)

    # sorting out features
    qualitative_features = get_qualitative_features(features)
    quantitative_features = get_quantitative_features(features)

    # applying default filters
    filtered = apply_filters(features, X, filters, default_filters=True)
    assert len(filtered) == len(features)

    features[-1].measures = {"measure_name": {"valid": False}}
    filtered = apply_filters(features, X, filters, default_filters=True)
    assert len(filtered) == (len(features) - 1)

    # applying qualitative filters
    filtered = apply_filters(qualitative_features, X, qualitative_filters, default_filters=False)
    assert len(filtered) == len(qualitative_features)
    qualitative_filters[-1].threshold = 0.0
    filtered = apply_filters(qualitative_features, X, qualitative_filters, default_filters=False)
    assert len(filtered) == (len(qualitative_features) - 1)

    # applying quantitative filters
    filtered = apply_filters(quantitative_features, X, quantitative_filters, default_filters=False)
    assert len(filtered) == len(quantitative_features)
    quantitative_filters[-1].threshold = 0.0
    filtered = apply_filters(quantitative_features, X, quantitative_filters, default_filters=False)
    assert len(filtered) == (len(quantitative_features) - 1)


def test_get_best_features(
    features: list[BaseFeature],
    X: pd.DataFrame,
    y: pd.Series,
    measures: list[BaseMeasure],
    filters: list[BaseFilter],
) -> None:
    """testing function get_best_features"""
    # sorting out features
    qualitative_features = get_qualitative_features(features)
    quantitative_features = get_quantitative_features(features)

    # sorting out measures
    quantitative_measures = get_quantitative_metrics(measures)
    qualitative_measures = get_qualitative_metrics(measures)

    # sorting out filters
    quantitative_filters = get_quantitative_metrics(filters)
    qualitative_filters = get_qualitative_metrics(filters)

    # non sortable measures
    with raises(ValueError):
        get_best_features(quantitative_features, X, y, quantitative_measures, quantitative_filters, 1)
    # when default_measure is OutlierMeasure there are no default_measure for qualtitatives
    if any(not measure.is_sortable for measure in qualitative_measures):
        with raises(ValueError):
            get_best_features(qualitative_features, X, y, qualitative_measures, qualitative_filters, 1)

    # sorting out measures
    quantitative_measures = remove_default_metrics(quantitative_measures)
    qualitative_measures = remove_default_metrics(qualitative_measures)

    # getting all quantitative features
    n_best = len(quantitative_features)
    best_features = get_best_features(quantitative_features, X, y, quantitative_measures, quantitative_filters, n_best)
    assert len(best_features) == len(quantitative_features)
    for feature in quantitative_features:
        assert feature in best_features

    # getting all qualitative features
    n_best = len(qualitative_features)
    best_features = get_best_features(qualitative_features, X, y, qualitative_measures, qualitative_filters, n_best)
    assert len(best_features) == len(qualitative_features)
    for feature in qualitative_features:
        assert feature in best_features

    # testing out quantitative measures
    n_best = 1
    best_features = get_best_features(quantitative_features, X, y, quantitative_measures, quantitative_filters, n_best)
    assert len(best_features) == n_best

    # testing out qualitative measures
    n_best = 1
    best_features = get_best_features(qualitative_features, X, y, qualitative_measures, qualitative_filters, n_best)
    assert len(best_features) == n_best

    # testing out quantitative filters
    n_best = len(quantitative_features)
    quantitative_filters[-1].threshold = 0
    best_features = get_best_features(quantitative_features, X, y, quantitative_measures, quantitative_filters, n_best)
    assert len(best_features) == 1

    # testing out qualitative filters
    n_best = len(quantitative_features)
    qualitative_filters[-1].threshold = 0
    best_features = get_best_features(qualitative_features, X, y, qualitative_measures, qualitative_filters, n_best)
    assert len(best_features) == 1

    # mismatched qualitatitve features and measures
    with raises(TypeError):
        get_best_features(qualitative_features, X, y, quantitative_measures, qualitative_filters, 1)
    # mismatched qualitative features and filters
    with raises(TypeError):
        get_best_features(qualitative_features, X, y, qualitative_measures, quantitative_filters, 1)
    # mismatched quantitatitve features and measures
    with raises(TypeError):
        get_best_features(quantitative_features, X, y, qualitative_measures, quantitative_filters, 1)
    # mismatched quantitative features and filters
    with raises(TypeError):
        get_best_features(quantitative_features, X, y, quantitative_measures, qualitative_filters, 1)


def test_base_selector_init_valid_parameters(features_object: Features) -> None:
    """test init of base selector"""

    # n_best < len(features)
    selector = BaseSelector(n_best_per_type=2, features=features_object)
    assert selector.n_best_per_type == 2
    assert selector.features == features_object

    # n_best > len(features)
    with raises(ValueError):
        BaseSelector(n_best_per_type=100, features=features_object)


def test_base_selector_select(
    features_object: Features,
    X: pd.DataFrame,
    y: pd.Series,
    measures: list[BaseMeasure],
    filters: list[BaseFilter],
) -> None:
    """tests BaseSelector select function"""

    # keeping all features
    selector = BaseSelector(n_best_per_type=2, features=features_object, measures=measures, filters=filters)
    best_features = selector.fit(X, y).selected_features
    assert isinstance(best_features, Features)
    for feature in features_object.quantitatives:
        assert feature in best_features
    for feature in features_object.qualitatives:
        assert feature in best_features

    # transform restricts to the selected columns
    assert set(selector.transform(X).columns) == {feature.version for feature in best_features}

    # keeping best feature per type
    selector = BaseSelector(n_best_per_type=1, features=features_object, measures=measures, filters=filters)
    best_features = selector.fit(X, y).selected_features

    # checking that one feature has been selected per type
    assert len(best_features) == 2

    # checking that one feature has been selected in qualitatives
    qualitative_sorted_features = sort_features_per_measure(
        get_qualitative_features(features_object),
        remove_default_metrics(get_qualitative_metrics(measures))[0],
    )
    assert len(get_qualitative_features(best_features)) == 1
    assert get_qualitative_features(best_features)[0] == qualitative_sorted_features[0]

    # checking that one feature has been selected in quantitatives
    quantitative_sorted_features = sort_features_per_measure(
        get_quantitative_features(features_object),
        remove_default_metrics(get_quantitative_metrics(measures))[0],
    )
    assert len(get_quantitative_features(best_features)) == 1
    assert get_quantitative_features(best_features)[0] == quantitative_sorted_features[0]


def test_base_selector_scores_all_features_exhaustively(
    features_object: Features,
    X: pd.DataFrame,
    y: pd.Series,
    measures: list[BaseMeasure],
    filters: list[BaseFilter],
) -> None:
    """selection is exhaustive: the global best per type is always selected,
    even across many (correlated) features — there is no chunk sampling."""
    # generating several correlated copies of each feature
    new_features = []
    new_X = {}
    from AutoCarver.features.qualitatives import OrdinalFeature

    for i in range(10):
        for feature in features_object:
            new_name = feature.name + f"_{i}"
            new_X.update({new_name: X[feature.name]})
            if isinstance(feature, OrdinalFeature):
                new_feature = OrdinalFeature(new_name, values=feature.raw_order)
            else:
                new_feature = type(feature)(new_name)
            new_features += [new_feature]

    X = pd.DataFrame(new_X)
    features_object = Features.from_list(new_features)

    selector = BaseSelector(n_best_per_type=1, features=features_object, measures=measures, filters=filters)
    best_features = selector.fit(X, y).selected_features
    assert len(best_features) == 2
    assert len(get_quantitative_features(best_features)) == 1
    assert len(get_qualitative_features(best_features)) == 1

    # the selected feature is the global best across all features of its type
    quantitative_sorted_features = sort_features_per_measure(
        get_quantitative_features(features_object),
        remove_default_metrics(get_quantitative_metrics(measures))[0],
    )
    assert get_quantitative_features(best_features)[0] == quantitative_sorted_features[0]

    qualitative_sorted_features = sort_features_per_measure(
        get_qualitative_features(features_object),
        remove_default_metrics(get_qualitative_metrics(measures))[0],
    )
    assert get_qualitative_features(best_features)[0] == qualitative_sorted_features[0]
