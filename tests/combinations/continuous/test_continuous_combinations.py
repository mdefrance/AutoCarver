""" set of tests for the continuous_combinations module """

from pandas import DataFrame, Series, isna
from pytest import FixtureRequest, fixture, raises
from scipy.stats import kruskal

from AutoCarver.carvers.continuous_carver import get_target_values_by_modality
from AutoCarver.combinations.continuous.continuous_combination_evaluators import (
    ContinuousCombinationEvaluator,
    KruksalCombinations,
)
from AutoCarver.features import OrdinalFeature


@fixture(params=[KruksalCombinations])
def evaluator(request: FixtureRequest) -> ContinuousCombinationEvaluator:
    return request.param(max_n_mod=5, min_freq=0.2)


def test_init(evaluator: ContinuousCombinationEvaluator):
    assert evaluator.is_y_binary is False
    assert evaluator.is_y_continuous is True
    assert evaluator.sort_by in ["kruskal"]


def test_association_measure_basic(evaluator: ContinuousCombinationEvaluator):
    X = DataFrame({"feature": ["A", "B", "A", "B", "C"]})
    y = Series([1, 2, 3, 4, 5])
    feature = OrdinalFeature("feature", ["A", "B", "C"])
    xagg = get_target_values_by_modality(X, y, feature)

    result = evaluator._association_measure(xagg)
    expected = {"kruskal": kruskal(*tuple(xagg.values))[0]}
    assert result == expected


def test_association_measure_with_nan(evaluator: ContinuousCombinationEvaluator):
    X = DataFrame({"feature": ["A", "B", "A", "B", None]})
    y = Series([1, 2, 3, 4, 5])
    feature = OrdinalFeature("feature", ["A", "B", "C"])
    xagg = get_target_values_by_modality(X, y, feature)

    result = evaluator._association_measure(xagg)
    assert isna(result.get("kruskal"))


def test_association_measure_single_value(evaluator: ContinuousCombinationEvaluator):
    X = DataFrame({"feature": ["A"]})
    y = Series([1])
    feature = OrdinalFeature("feature", ["A"])
    xagg = get_target_values_by_modality(X, y, feature)

    with raises(ValueError):
        evaluator._association_measure(xagg)


def test_association_measure_identical_values(evaluator: ContinuousCombinationEvaluator):
    X = DataFrame({"feature": ["A", "A", "A", "A", "A"]})
    y = Series([1, 1, 1, 1, 1])
    feature = OrdinalFeature("feature", ["A"])
    xagg = get_target_values_by_modality(X, y, feature)

    with raises(ValueError):
        evaluator._association_measure(xagg)


def test_grouper_basic(evaluator: ContinuousCombinationEvaluator):
    X = DataFrame({"feature": ["A", "B", "A", "B", "C"]})
    y = Series([1, 2, 3, 4, 5])
    feature = OrdinalFeature("feature", ["A", "B", "C"])
    xagg = get_target_values_by_modality(X, y, feature)

    groupby = {"A": "group1", "B": "group1", "C": "group2"}
    result = evaluator._grouper(xagg, groupby)

    expected = Series({"group1": [1, 3, 2, 4], "group2": [5]})
    assert result.equals(expected)


def test_grouper_with_nan(evaluator: ContinuousCombinationEvaluator):
    X = DataFrame({"feature": ["A", "B", "A", "B", None]})
    y = Series([1, 2, 3, 4, 5])
    feature = OrdinalFeature("feature", ["A", "B", "C"])
    xagg = get_target_values_by_modality(X, y, feature)

    groupby = {"A": "group1", "B": "group1", "C": "group2"}
    result = evaluator._grouper(xagg, groupby)

    expected = Series({"group1": [1, 3, 2, 4], "group2": []})
    assert result.equals(expected)


def test_grouper_unordered_labels(evaluator: ContinuousCombinationEvaluator):
    X = DataFrame({"feature": ["A", "B", "A", "B", "C"]})
    y = Series([1, 2, 3, 4, 5])
    feature = OrdinalFeature("feature", ["C", "A", "B"])
    xagg = get_target_values_by_modality(X, y, feature)

    groupby = {"A": "group1", "B": "group1", "C": "group2"}
    result = evaluator._grouper(xagg, groupby)

    expected = Series({"group1": [1, 3, 2, 4], "group2": [5]})
    assert result.equals(expected)


def test_grouper_missing_labels(evaluator: ContinuousCombinationEvaluator):
    X = DataFrame({"feature": ["A", "B", "A", "B", "C"]})
    y = Series([1, 2, 3, 4, 5])
    feature = OrdinalFeature("feature", ["A", "B"])
    xagg = get_target_values_by_modality(X, y, feature)
    groupby = {"A": "group1", "B": "group1", "C": "group2"}

    result = evaluator._grouper(xagg, groupby)
    expected = Series({"group1": [1, 3, 2, 4]})
    assert result.equals(expected)


def test_grouper_extra_labels(evaluator: ContinuousCombinationEvaluator):
    X = DataFrame({"feature": ["A", "B", "A", "B", "C"]})
    y = Series([1, 2, 3, 4, 5])
    feature = OrdinalFeature("feature", ["A", "B", "C", "D"])
    xagg = get_target_values_by_modality(X, y, feature)
    groupby = {"A": "group1", "B": "group1", "C": "group2", "D": "group3"}

    result = evaluator._grouper(xagg, groupby)
    expected = Series({"group1": [1, 3, 2, 4], "group2": [5], "group3": []})
    assert result.equals(expected)


def test_compute_target_rates_basic(evaluator: ContinuousCombinationEvaluator):
    X = DataFrame({"feature": ["A", "B", "A", "B", "C"]})
    y = Series([1, 2, 3, 4, 5])
    feature = OrdinalFeature("feature", ["A", "B", "C"])
    xagg = get_target_values_by_modality(X, y, feature)

    result = evaluator._compute_target_rates(xagg)
    expected = DataFrame(
        {"target_rate": [2.0, 3.0, 5.0], "frequency": [2 / 5, 2 / 5, 1 / 5]}, index=["A", "B", "C"]
    )
    assert result.equals(expected)


def test_compute_target_rates_with_nan(evaluator: ContinuousCombinationEvaluator):
    X = DataFrame({"feature": ["A", "B", "A", "B", None]})
    y = Series([1, 2, 3, 4, 5])
    feature = OrdinalFeature("feature", ["A", "B", "C"])
    xagg = get_target_values_by_modality(X, y, feature)

    result = evaluator._compute_target_rates(xagg)
    expected = DataFrame(
        {"target_rate": [2.0, 3.0, None], "frequency": [2 / 4, 2 / 4, 0]}, index=["A", "B", "C"]
    )
    assert result.equals(expected)


def test_compute_target_rates_unordered_labels(evaluator: ContinuousCombinationEvaluator):
    X = DataFrame({"feature": ["A", "B", "A", "B", "C"]})
    y = Series([1, 2, 3, 4, 5])
    feature = OrdinalFeature("feature", ["C", "A", "B"])
    xagg = get_target_values_by_modality(X, y, feature)

    result = evaluator._compute_target_rates(xagg)
    expected = DataFrame(
        {"target_rate": [5.0, 2.0, 3.0], "frequency": [1 / 5, 2 / 5, 2 / 5]}, index=["C", "A", "B"]
    )
    assert result.equals(expected)


def test_compute_target_rates_missing_labels(evaluator: ContinuousCombinationEvaluator):
    X = DataFrame({"feature": ["A", "B", "A", "B", "C"]})
    y = Series([1, 2, 3, 4, 5])
    feature = OrdinalFeature("feature", ["A", "B"])
    xagg = get_target_values_by_modality(X, y, feature)

    result = evaluator._compute_target_rates(xagg)
    expected = DataFrame({"target_rate": [2.0, 3.0], "frequency": [2 / 4, 2 / 4]}, index=["A", "B"])
    assert result.equals(expected)


def test_compute_target_rates_extra_labels(evaluator: ContinuousCombinationEvaluator):
    X = DataFrame({"feature": ["A", "B", "A", "B", "C"]})
    y = Series([1, 2, 3, 4, 5])
    feature = OrdinalFeature("feature", ["A", "B", "C", "D"])
    xagg = get_target_values_by_modality(X, y, feature)

    result = evaluator._compute_target_rates(xagg)
    expected = DataFrame(
        {"target_rate": [2.0, 3.0, 5.0, None], "frequency": [2 / 5, 2 / 5, 1 / 5, 0]},
        index=["A", "B", "C", "D"],
    )
    assert result.equals(expected)
