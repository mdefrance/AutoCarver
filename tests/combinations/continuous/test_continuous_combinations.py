"""set of tests for the continuous_combinations module"""

import json

import numpy as np
import pandas as pd
from pytest import FixtureRequest, fixture, raises
from scipy.stats import kruskal

from AutoCarver.carvers.continuous_carver import get_target_values_by_modality
from AutoCarver.combinations.binary.binary_combination_evaluators import (
    CramervCombinations,
    TschuprowtCombinations,
)
from AutoCarver.combinations.continuous.continuous_combination_evaluators import (
    ContinuousCombinationEvaluator,
    KruskalCombinations,
)
from AutoCarver.combinations.utils.combination_evaluator import AggregatedSample
from AutoCarver.combinations.utils.combinations import consecutive_combinations, nan_combinations
from AutoCarver.combinations.utils.testing import Keys, Messages
from AutoCarver.features import OrdinalFeature

MAX_N_MOD = 5
MIN_FREQ = 0.2


@fixture(params=[KruskalCombinations])
def evaluator(request: FixtureRequest) -> ContinuousCombinationEvaluator:
    return request.param(max_n_mod=MAX_N_MOD, min_freq=MIN_FREQ)


def test_init(evaluator: ContinuousCombinationEvaluator):
    assert evaluator.is_y_binary is False
    assert evaluator.is_y_continuous is True
    assert evaluator.sort_by == "kruskal"


def test_to_json(evaluator: ContinuousCombinationEvaluator):
    """Test the to_json method"""
    expected_json = {
        "sort_by": evaluator.sort_by,
        "max_n_mod": MAX_N_MOD,
        "dropna": evaluator.dropna,
        "min_freq": MIN_FREQ,
        "verbose": evaluator.verbose,
    }
    assert evaluator.to_json() == expected_json


def test_save(evaluator: ContinuousCombinationEvaluator, tmp_path):
    """Test the save method"""
    file_name = tmp_path / "test.json"
    evaluator.save(str(file_name))

    with open(file_name, encoding="utf-8") as json_file:
        data = json.load(json_file)

    expected_json = {
        "sort_by": evaluator.sort_by,
        "max_n_mod": MAX_N_MOD,
        "dropna": evaluator.dropna,
        "min_freq": MIN_FREQ,
        "verbose": evaluator.verbose,
    }
    assert data == expected_json


def test_save_invalid_filename(evaluator: ContinuousCombinationEvaluator):
    """Test the save method with an invalid filename"""
    with raises(ValueError):
        evaluator.save("invalid_file.txt")


def test_load(tmp_path):
    """Test the load method"""
    file_name = tmp_path / "test.json"
    data = {
        "sort_by": "kruskal",
        "max_n_mod": MAX_N_MOD,
        "dropna": True,
        "min_freq": MIN_FREQ,
        "verbose": True,
    }

    with open(file_name, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file)

    loaded_evaluator = KruskalCombinations.load(str(file_name))

    assert loaded_evaluator.max_n_mod == MAX_N_MOD
    assert loaded_evaluator.dropna is True
    assert loaded_evaluator.min_freq == MIN_FREQ
    assert loaded_evaluator.verbose is True
    assert loaded_evaluator.sort_by == "kruskal"
    assert loaded_evaluator.is_y_binary is False
    assert loaded_evaluator.is_y_continuous is True

    with raises(ValueError):
        TschuprowtCombinations.load(str(file_name))

    with raises(ValueError):
        CramervCombinations.load(str(file_name))


def test_association_measure_basic(evaluator: ContinuousCombinationEvaluator):
    X = pd.DataFrame({"feature": ["A", "B", "A", "B", "C"]})
    y = pd.Series([1, 2, 3, 4, 5])
    feature = OrdinalFeature("feature", ["A", "B", "C"])
    xagg = get_target_values_by_modality(X, y, feature)

    result = evaluator._association_measure(xagg)
    expected = {"kruskal": kruskal(*tuple(xagg.values))[0]}
    assert result == expected


def test_association_measure_with_nan(evaluator: ContinuousCombinationEvaluator):
    X = pd.DataFrame({"feature": ["A", "B", "A", "B", None]})
    y = pd.Series([1, 2, 3, 4, 5])
    feature = OrdinalFeature("feature", ["A", "B", "C"])
    xagg = get_target_values_by_modality(X, y, feature)

    result = evaluator._association_measure(xagg)
    assert pd.isna(result.get("kruskal"))


def test_association_measure_single_value(evaluator: ContinuousCombinationEvaluator):
    X = pd.DataFrame({"feature": ["A"]})
    y = pd.Series([1])
    feature = OrdinalFeature("feature", ["A"])
    xagg = get_target_values_by_modality(X, y, feature)

    result = evaluator._association_measure(xagg)
    assert result.get("kruskal") is None


def test_association_measure_identical_values(evaluator: ContinuousCombinationEvaluator):
    X = pd.DataFrame({"feature": ["A", "A", "A", "A", "A"]})
    y = pd.Series([1, 1, 1, 1, 1])
    feature = OrdinalFeature("feature", ["A"])
    xagg = get_target_values_by_modality(X, y, feature)

    result = evaluator._association_measure(xagg)
    assert result.get("kruskal") is None


def test_grouper_basic(evaluator: ContinuousCombinationEvaluator):
    X = pd.DataFrame({"feature": ["A", "B", "A", "B", "C"]})
    y = pd.Series([1, 2, 3, 4, 5])
    feature = OrdinalFeature("feature", ["A", "B", "C"])
    xagg = get_target_values_by_modality(X, y, feature)

    groupby = {"A": "group1", "B": "group1", "C": "group2"}
    result = evaluator._grouper(xagg, groupby)

    expected = pd.Series({"group1": [1, 3, 2, 4], "group2": [5]})
    assert result.equals(expected)


def test_grouper_with_nan(evaluator: ContinuousCombinationEvaluator):
    X = pd.DataFrame({"feature": ["A", "B", "A", "B", None]})
    y = pd.Series([1, 2, 3, 4, 5])
    feature = OrdinalFeature("feature", ["A", "B", "C"])
    xagg = get_target_values_by_modality(X, y, feature)

    groupby = {"A": "group1", "B": "group1", "C": "group2"}
    result = evaluator._grouper(xagg, groupby)

    expected = pd.Series({"group1": [1, 3, 2, 4], "group2": []})
    assert result.equals(expected)


def test_grouper_unordered_labels(evaluator: ContinuousCombinationEvaluator):
    X = pd.DataFrame({"feature": ["A", "B", "A", "B", "C"]})
    y = pd.Series([1, 2, 3, 4, 5])
    feature = OrdinalFeature("feature", ["C", "A", "B"])
    xagg = get_target_values_by_modality(X, y, feature)

    groupby = {"A": "group1", "B": "group1", "C": "group2"}
    result = evaluator._grouper(xagg, groupby)

    expected = pd.Series({"group1": [1, 3, 2, 4], "group2": [5]})
    assert result.equals(expected)


def test_grouper_missing_labels(evaluator: ContinuousCombinationEvaluator):
    X = pd.DataFrame({"feature": ["A", "B", "A", "B", "C"]})
    y = pd.Series([1, 2, 3, 4, 5])
    feature = OrdinalFeature("feature", ["A", "B"])
    xagg = get_target_values_by_modality(X, y, feature)
    groupby = {"A": "group1", "B": "group1", "C": "group2"}

    result = evaluator._grouper(xagg, groupby)
    expected = pd.Series({"group1": [1, 3, 2, 4]})
    assert result.equals(expected)


def test_grouper_extra_labels(evaluator: ContinuousCombinationEvaluator):
    X = pd.DataFrame({"feature": ["A", "B", "A", "B", "C"]})
    y = pd.Series([1, 2, 3, 4, 5])
    feature = OrdinalFeature("feature", ["A", "B", "C", "D"])
    xagg = get_target_values_by_modality(X, y, feature)
    groupby = {"A": "group1", "B": "group1", "C": "group2", "D": "group3"}

    result = evaluator._grouper(xagg, groupby)
    expected = pd.Series({"group1": [1, 3, 2, 4], "group2": [5], "group3": []})
    assert result.equals(expected)


def test_compute_target_rates_basic(evaluator: ContinuousCombinationEvaluator):
    X = pd.DataFrame({"feature": ["A", "B", "A", "B", "C"]})
    y = pd.Series([1, 2, 3, 4, 5])
    feature = OrdinalFeature("feature", ["A", "B", "C"])
    xagg = get_target_values_by_modality(X, y, feature)

    result = evaluator.target_rate.compute(xagg)
    expected = pd.DataFrame({"target_mean": [2.0, 3.0, 5.0], "frequency": [2 / 5, 2 / 5, 1 / 5]}, index=["A", "B", "C"])
    assert result.equals(expected)


def test_compute_target_rates_with_nan(evaluator: ContinuousCombinationEvaluator):
    X = pd.DataFrame({"feature": ["A", "B", "A", "B", None]})
    y = pd.Series([1, 2, 3, 4, 5])
    feature = OrdinalFeature("feature", ["A", "B", "C"])
    xagg = get_target_values_by_modality(X, y, feature)

    result = evaluator.target_rate.compute(xagg)
    expected = pd.DataFrame({"target_mean": [2.0, 3.0, None], "frequency": [2 / 4, 2 / 4, 0]}, index=["A", "B", "C"])
    assert result.equals(expected)


def test_compute_target_rates_unordered_labels(evaluator: ContinuousCombinationEvaluator):
    X = pd.DataFrame({"feature": ["A", "B", "A", "B", "C"]})
    y = pd.Series([1, 2, 3, 4, 5])
    feature = OrdinalFeature("feature", ["C", "A", "B"])
    xagg = get_target_values_by_modality(X, y, feature)

    result = evaluator.target_rate.compute(xagg)
    expected = pd.DataFrame({"target_mean": [5.0, 2.0, 3.0], "frequency": [1 / 5, 2 / 5, 2 / 5]}, index=["C", "A", "B"])
    assert result.equals(expected)


def test_compute_target_rates_missing_labels(evaluator: ContinuousCombinationEvaluator):
    X = pd.DataFrame({"feature": ["A", "B", "A", "B", "C"]})
    y = pd.Series([1, 2, 3, 4, 5])
    feature = OrdinalFeature("feature", ["A", "B"])
    xagg = get_target_values_by_modality(X, y, feature)

    result = evaluator.target_rate.compute(xagg)
    expected = pd.DataFrame({"target_mean": [2.0, 3.0], "frequency": [2 / 4, 2 / 4]}, index=["A", "B"])
    assert result.equals(expected)


def test_compute_target_rates_extra_labels(evaluator: ContinuousCombinationEvaluator):
    X = pd.DataFrame({"feature": ["A", "B", "A", "B", "C"]})
    y = pd.Series([1, 2, 3, 4, 5])
    feature = OrdinalFeature("feature", ["A", "B", "C", "D"])
    xagg = get_target_values_by_modality(X, y, feature)

    result = evaluator.target_rate.compute(xagg)
    expected = pd.DataFrame(
        {"target_mean": [2.0, 3.0, 5.0, None], "frequency": [2 / 5, 2 / 5, 1 / 5, 0]},
        index=["A", "B", "C", "D"],
    )
    assert result.equals(expected)


def test_group_xagg_by_combinations(evaluator: ContinuousCombinationEvaluator):
    """Test the _group_xagg_by_combinations method"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])
    xagg = pd.Series({"a": [0, 2, 0], "b": [2, 1], "c": [2, 0]})

    combinations = consecutive_combinations(feature.labels, 2)

    evaluator.samples.train = AggregatedSample(xagg)

    result = evaluator._group_xagg_by_combinations(combinations)
    print(result)

    expected = [
        {
            "xagg": pd.Series({"a": [0, 2, 0], "b": [2, 1, 2, 0]}),
            "combination": [["a"], ["b", "c"]],
            "index_to_groupby": {"a": "a", "b": "b", "c": "b"},
        },
        {
            "xagg": pd.Series({"a": [0, 2, 0, 2, 1], "c": [2, 0]}),
            "combination": [["a", "b"], ["c"]],
            "index_to_groupby": {"a": "a", "b": "a", "c": "c"},
        },
    ]
    for res, exp in zip(result, expected):
        assert list(res["xagg"]) == list(exp["xagg"])
        assert res["combination"] == exp["combination"]
        assert res["index_to_groupby"] == exp["index_to_groupby"]


def test_group_xagg_by_combinations_with_nan(evaluator: ContinuousCombinationEvaluator):
    """Test with a xagg with np.nan values"""

    feature = OrdinalFeature("feature", ["A", "B", "C"])
    xagg = pd.Series({"A": [0, 2, 0], "B": [2, 1], "C": [2, 0]})

    combinations = consecutive_combinations(feature.labels, 3)

    evaluator.samples.train = AggregatedSample(xagg)

    result = evaluator._group_xagg_by_combinations(combinations)
    print(result)

    expected = [
        {
            "xagg": pd.Series({"A": [0, 2, 0], "B": [2, 1], "C": [2, 0]}),
            "combination": [["A"], ["B"], ["C"]],
            "index_to_groupby": {"A": "A", "B": "B", "C": "C"},
        },
        {
            "xagg": pd.Series({"A": [0, 2, 0], "B": [2, 1, 2, 0]}),
            "combination": [["A"], ["B", "C"]],
            "index_to_groupby": {"A": "A", "B": "B", "C": "B"},
        },
        {
            "xagg": pd.Series({"A": [0, 2, 0, 2, 1], "C": [2, 0]}),
            "combination": [["A", "B"], ["C"]],
            "index_to_groupby": {"A": "A", "B": "A", "C": "C"},
        },
    ]
    for res, exp in zip(result, expected):
        assert list(res["xagg"]) == list(exp["xagg"])
        assert res["combination"] == exp["combination"]
        assert res["index_to_groupby"] == exp["index_to_groupby"]


def test_compute_associations(evaluator: ContinuousCombinationEvaluator):
    """Test the compute_associations method"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])
    xagg = pd.Series({"a": [0, 2, 0], "b": [2, 1], "c": [2, 0]})

    combinations = consecutive_combinations(feature.labels, 2)

    evaluator.samples.train = AggregatedSample(xagg)

    grouped_xaggs = evaluator._group_xagg_by_combinations(combinations)
    result = evaluator._compute_associations(grouped_xaggs)
    print(result)

    expected = [
        {
            "xagg": pd.Series({"a": [0, 2, 0], "b": [2, 1, 2, 0]}),
            "combination": [["a"], ["b", "c"]],
            "index_to_groupby": {"a": "a", "b": "b", "c": "b"},
            "kruskal": 0.5833333333333333,
        },
        {
            "xagg": pd.Series({"a": [0, 2, 0, 2, 1], "c": [2, 0]}),
            "combination": [["a", "b"], ["c"]],
            "index_to_groupby": {"a": "a", "b": "a", "c": "c"},
            "kruskal": 0.0,
        },
    ]
    for res, exp in zip(result, expected):
        assert list(res["xagg"]) == list(exp["xagg"])
        assert res["combination"] == exp["combination"]
        assert res["index_to_groupby"] == exp["index_to_groupby"]
        assert res["kruskal"] == exp["kruskal"]


def test_compute_associations_with_unobserved(evaluator: ContinuousCombinationEvaluator):
    """Test with a small xagg"""

    feature = OrdinalFeature("feature", ["A", "B", "C"])
    xagg = pd.Series({"A": [0, 2, 0], "B": [2, 1], "C": []})
    evaluator.samples.train = AggregatedSample(xagg)

    combinations = consecutive_combinations(feature.labels, 3)

    grouped_xaggs = evaluator._group_xagg_by_combinations(combinations)

    # cannot compute associations with a modality that has no observations
    result = evaluator._compute_associations(grouped_xaggs)
    # print(result)

    expected = [
        {
            "xagg": pd.Series({"A": [0, 2, 0], "B": [2, 1]}),
            "combination": [["A"], ["B", "C"]],
            "index_to_groupby": {"A": "A", "B": "B", "C": "B"},
            "kruskal": 0.8333333333333333,
        },
        {
            "xagg": pd.Series({"A": [0, 2, 0], "B": [2, 1], "C": []}),
            "combination": [["A"], ["B"], ["C"]],
            "index_to_groupby": {"A": "A", "B": "B", "C": "C"},
            "kruskal": np.nan,
        },
        {
            "xagg": pd.Series({"A": [0, 2, 0, 2, 1], "C": []}),
            "combination": [["A", "B"], ["C"]],
            "index_to_groupby": {"A": "A", "B": "A", "C": "C"},
            "kruskal": np.nan,
        },
    ]
    for res, exp in zip(result, expected):
        print(res, exp)
        assert list(res["xagg"]) == list(exp["xagg"])
        assert res["combination"] == exp["combination"]
        assert res["index_to_groupby"] == exp["index_to_groupby"]
        assert res["kruskal"] == exp["kruskal"] or (pd.isna(res["kruskal"]) and pd.isna(exp["kruskal"]))


def test_compute_associations_with_three_rows(evaluator: ContinuousCombinationEvaluator):
    """Test with a small xagg"""

    feature = OrdinalFeature("feature", ["A", "B", "C"])
    xagg = pd.Series({"A": [0, 2, 0], "B": [2, 1], "C": [2, 0]})
    evaluator.samples.train = AggregatedSample(xagg)

    combinations = consecutive_combinations(feature.labels, 3)

    grouped_xaggs = evaluator._group_xagg_by_combinations(combinations)
    result = evaluator._compute_associations(grouped_xaggs)
    print(result)

    expected = [
        {
            "xagg": pd.Series({"A": [0, 2, 0], "B": [2, 1], "C": [2, 0]}),
            "combination": [["A"], ["B"], ["C"]],
            "index_to_groupby": {"A": "A", "B": "B", "C": "C"},
            "kruskal": 0.8333333333333345,
        },
        {
            "xagg": pd.Series({"A": [0, 2, 0], "B": [2, 1, 2, 0]}),
            "combination": [["A"], ["B", "C"]],
            "index_to_groupby": {"A": "A", "B": "B", "C": "B"},
            "kruskal": 0.5833333333333333,
        },
        {
            "xagg": pd.Series({"A": [0, 2, 0, 2, 1], "C": [2, 0]}),
            "combination": [["A", "B"], ["C"]],
            "index_to_groupby": {"A": "A", "B": "A", "C": "C"},
            "kruskal": 0.0,
        },
    ]
    for res, exp in zip(result, expected):
        assert list(res["xagg"]) == list(exp["xagg"])
        assert res["combination"] == exp["combination"]
        assert res["index_to_groupby"] == exp["index_to_groupby"]
        assert res["kruskal"] == exp["kruskal"]


def test_compute_associations_with_ten_labels(evaluator: ContinuousCombinationEvaluator):
    """Test with a larger xagg (10 labels, 84 combinations)."""
    feature = OrdinalFeature("feature", [chr(i) for i in range(65, 75)])  # A to J
    xagg = pd.Series(
        {
            "A": [0, 2, 0],
            "B": [2, 1],
            "C": [2, 0, 5, 6],
            "D": [1, 3, 4],
            "E": [0, 1, 2, 3],
            "F": [4, 5],
            "G": [6, 7, 8],
            "H": [9, 10],
            "I": [11, 12, 13],
            "J": [7, 8],
        }
    )
    evaluator.samples.train = AggregatedSample(xagg)

    combinations = consecutive_combinations(feature.labels, 4)

    grouped_xaggs = evaluator._group_xagg_by_combinations(combinations)
    result = evaluator._compute_associations(grouped_xaggs)

    expected = {
        "xagg": pd.Series(
            {
                "A": [0, 2, 0],
                "B": [2, 1, 2, 0, 5, 6, 1, 3, 4, 0, 1, 2, 3],
                "F": [4, 5, 6, 7, 8],
                "H": [9, 10, 11, 12, 13, 7, 8],
            }
        ),
        "combination": [["A"], ["B", "C", "D", "E"], ["F", "G"], ["H", "I", "J"]],
        "index_to_groupby": {
            "A": "A",
            "B": "B",
            "C": "B",
            "D": "B",
            "E": "B",
            "F": "F",
            "G": "F",
            "H": "H",
            "I": "H",
            "J": "H",
        },
        "kruskal": 20.728840695728103,
    }
    res = result[0]
    assert list(res["xagg"]) == list(expected["xagg"])
    assert res["combination"] == expected["combination"]
    assert res["index_to_groupby"] == expected["index_to_groupby"]
    assert res["kruskal"] == expected["kruskal"]


def test_viability_train(evaluator: ContinuousCombinationEvaluator):
    """Test the viability of the combination on xagg_train"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])
    xagg = pd.Series({"a": [0, 2, 0], "b": [2, 1], "c": [2, 0]})

    combinations = consecutive_combinations(feature.labels, 2)

    evaluator.samples.train = AggregatedSample(xagg)

    grouped_xaggs = evaluator._group_xagg_by_combinations(combinations)
    associations = evaluator._compute_associations(grouped_xaggs)
    result = []
    for combination in associations:
        result += [evaluator._test_viability_train(combination)]
    print(result)

    expected = [
        {
            "train": {Keys.VIABLE.value: True},
            "train_rates": pd.DataFrame(
                {"target_mean": [0.666667, 1.250000], "frequency": [0.428571, 0.571429]},
                index=["a", "b"],
            ),
        },
        {
            "train": {Keys.VIABLE.value: False},
            "train_rates": pd.DataFrame(
                {"target_mean": [1.0, 1.0], "frequency": [0.714286, 0.285714]}, index=["a", "c"]
            ),
        },
    ]
    for res, exp in zip(result, expected):
        assert all(res["train_rates"].index == exp["train_rates"].index)
        assert np.allclose(res["train_rates"], exp["train_rates"])
        assert res["train"][Keys.VIABLE.value] == exp["train"][Keys.VIABLE.value]


def test_viability_dev(evaluator: ContinuousCombinationEvaluator):
    """Test the viability of the combination on xagg_dev"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])
    xagg = pd.Series({"a": [0, 2, 0], "b": [2, 1], "c": [2, 0]})

    combinations = consecutive_combinations(feature.labels, 2)

    evaluator.samples.train = AggregatedSample(xagg)

    grouped_xaggs = evaluator._group_xagg_by_combinations(combinations)
    associations = evaluator._compute_associations(grouped_xaggs)

    # test with no xagg_dev
    for combination in associations:
        test_results = evaluator._test_viability_train(combination)
        test_results = evaluator._test_viability_dev(test_results, combination)
        assert test_results.get("dev").get(Keys.VIABLE.value) is None

    # test with xagg_dev but not viable on train
    evaluator.samples.dev = AggregatedSample(pd.Series({"a": [0, 2, 0], "b": [2, 1], "c": [2, 0]}))
    for combination in associations:
        test_results = evaluator._test_viability_dev(
            {"train": {Keys.VIABLE.value: False}, Keys.VIABLE.value: False},
            combination,
        )
        assert test_results.get("dev").get(Keys.VIABLE.value) is None

    # test with xagg_dev same as train
    result = []
    for combination in associations:
        test_results = evaluator._test_viability_train(combination)
        result += [evaluator._test_viability_dev(test_results, combination)]

    expected = [
        {
            "train": {Keys.VIABLE.value: True, Keys.INFO: Messages.PASSED_TESTS},
            "dev": {
                Keys.VIABLE.value: True,
                Keys.INFO: Messages.PASSED_TESTS,
            },
        },
        {
            "train": {Keys.VIABLE.value: False, Keys.INFO: Messages.NON_DISTINCT_RATES},
            "dev": {Keys.VIABLE.value: None},
        },
    ]
    for res, exp in zip(result, expected):
        assert res["train"] == exp["train"]
        assert res["dev"] == exp["dev"]

    # test with xagg_dev wrong
    evaluator.samples.dev = AggregatedSample(pd.Series({"a": [10000, 2, 0], "b": [2, 1], "c": [2, 0]}))
    result = []
    for combination in associations:
        test_results = evaluator._test_viability_train(combination)
        result += [evaluator._test_viability_dev(test_results, combination)]
    # print(result)

    expected = [
        {
            "train": {
                Keys.VIABLE.value: True,
                Keys.INFO.value: Messages.PASSED_TESTS.value,
            },
            "dev": {
                Keys.VIABLE.value: False,
                Keys.INFO.value: Messages.INVERSION_RATES.value,
            },
        },
        {
            "train": {
                Keys.VIABLE.value: False,
                Keys.INFO.value: Messages.NON_DISTINCT_RATES.value,
            },
            "dev": {Keys.VIABLE.value: None},
        },
    ]
    for res, exp in zip(result, expected):
        assert res["train"] == exp["train"]
        assert res["dev"] == exp["dev"]


def test_get_viable_combination_without_dev(evaluator: ContinuousCombinationEvaluator):
    """Test the get_viable_combination method without a dev xagg"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])
    xagg = pd.Series({"a": [0, 2, 0], "b": [2, 1], "c": [2, 0]})

    combinations = consecutive_combinations(feature.labels, 2)
    evaluator.feature = feature

    evaluator.samples.train = AggregatedSample(xagg)

    grouped_xaggs = evaluator._group_xagg_by_combinations(combinations)
    associations = evaluator._compute_associations(grouped_xaggs)

    # test with no xagg_dev
    result = evaluator._get_viable_combination(associations)
    print(result)
    expected = {
        "xagg": pd.Series({"a": [0, 2, 0], "b": [2, 1, 2, 0]}),
        "combination": [["a"], ["b", "c"]],
        "kruskal": 0.5833333333333333,
    }
    assert result["xagg"].equals(expected["xagg"])
    assert result["combination"] == expected["combination"]
    assert result["kruskal"] == expected["kruskal"]


def test_get_viable_combination_with_non_viable_train(evaluator: ContinuousCombinationEvaluator):
    """Test the get_viable_combination method with a non-viable train"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])
    xagg = pd.Series({"a": [0, 2, 0], "b": [2, 1], "c": [2, 0]})
    evaluator.samples.train = AggregatedSample(xagg)
    evaluator.feature = feature

    combinations = consecutive_combinations(feature.labels, 2)

    grouped_xaggs = evaluator._group_xagg_by_combinations(combinations)
    associations = evaluator._compute_associations(grouped_xaggs)

    # test with xagg_dev but not viable on train
    evaluator.min_freq = 0.6
    evaluator.samples.dev = AggregatedSample(xagg)
    result = evaluator._get_viable_combination(associations)
    print(result)

    assert result is None


def test_get_viable_combination_with_viable_train(evaluator: ContinuousCombinationEvaluator):
    """Test the get_viable_combination method with a viable train"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])
    xagg = pd.Series({"a": [0, 2, 0], "b": [2, 1], "c": [2, 0]})
    evaluator.samples.train = AggregatedSample(xagg)
    evaluator.feature = feature

    combinations = consecutive_combinations(feature.labels, 2)

    grouped_xaggs = evaluator._group_xagg_by_combinations(combinations)
    associations = evaluator._compute_associations(grouped_xaggs)

    # test with xagg_dev same as train
    evaluator.samples.dev = AggregatedSample(xagg)
    result = evaluator._get_viable_combination(associations)
    print(result)

    expected = {
        "xagg": pd.Series({"a": [0, 2, 0], "b": [2, 1, 2, 0]}),
        "combination": [["a"], ["b", "c"]],
        "kruskal": 0.5833333333333333,
    }
    assert result["xagg"].equals(expected["xagg"])
    assert result["combination"] == expected["combination"]
    assert result["kruskal"] == expected["kruskal"]


def test_get_viable_combination_with_not_viable_dev(evaluator: ContinuousCombinationEvaluator):
    """Test the get_viable_combination method with a non-viable dev"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])
    xagg = pd.Series({"a": [0, 2, 0], "b": [2, 1], "c": [2, 0]})
    evaluator.samples.train = AggregatedSample(xagg)
    evaluator.feature = feature

    combinations = consecutive_combinations(feature.labels, 2)

    grouped_xaggs = evaluator._group_xagg_by_combinations(combinations)
    associations = evaluator._compute_associations(grouped_xaggs)

    # test with xagg_dev wrong
    evaluator.samples.dev = AggregatedSample(pd.Series({"a": [0, 2, 1000], "b": [2, 1], "c": [2, 0]}))
    result = evaluator._get_viable_combination(associations)
    print(result)
    assert result is None


def test_apply_best_combination_with_viable(evaluator: ContinuousCombinationEvaluator):
    """Test the apply_best_combination method with a viable combination"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])
    evaluator.feature = feature
    xagg = pd.Series({"a": [0, 2, 0], "b": [2, 1], "c": [2, 0]})
    evaluator.samples.train = AggregatedSample(xagg)

    combinations = consecutive_combinations(feature.labels, 2)

    grouped_xaggs = evaluator._group_xagg_by_combinations(combinations)
    associations = evaluator._compute_associations(grouped_xaggs)

    # test with xagg_dev same as train
    evaluator.samples.dev = AggregatedSample(xagg)

    best_combination = evaluator._get_viable_combination(associations)

    evaluator._apply_best_combination(best_combination)

    expected = pd.Series({"a": [0, 2, 0], "b to c": [2, 1, 2, 0]})
    print(evaluator.samples.train.xagg)
    assert feature.labels == list(expected.index)
    assert evaluator.samples.train.xagg.equals(expected)
    assert evaluator.samples.dev.xagg.equals(expected)
    assert evaluator.samples.dev.raw.equals(expected)
    assert evaluator.samples.train.raw.equals(expected)


def test_apply_best_combination_with_non_viable(evaluator: ContinuousCombinationEvaluator):
    """Test the apply best combination with a non-viable combination"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])
    evaluator.feature = feature
    xagg = pd.Series({"a": [0, 2, 0], "b": [2, 1], "c": [2, 0]})
    evaluator.samples.train = AggregatedSample(xagg)

    # test with xagg_dev same as train
    evaluator.samples.dev = AggregatedSample(xagg)

    best_combination = None

    evaluator._apply_best_combination(best_combination)

    expected = xagg
    assert feature.labels == list(expected.index)
    assert evaluator.samples.train.xagg.equals(expected)
    assert evaluator.samples.dev.xagg.equals(expected)
    assert evaluator.samples.dev.raw.equals(expected)
    assert evaluator.samples.train.raw.equals(expected)


def test_best_association_with_combinations_viable(evaluator: ContinuousCombinationEvaluator):
    """Test the best association with viable combinations"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])
    evaluator.feature = feature

    xagg = pd.Series({"a": [0, 2, 0], "b": [2, 1], "c": [2, 0]})
    evaluator.samples.train = AggregatedSample(xagg)
    evaluator.samples.dev = AggregatedSample(xagg)

    combinations = consecutive_combinations(feature.labels, 2)

    result = evaluator._get_best_association(combinations)
    print(result)

    expected = {
        "xagg": pd.Series({"a": [0, 2, 0], "b": [2, 1, 2, 0]}),
        "combination": [["a"], ["b", "c"]],
        "kruskal": 0.5833333333333333,
    }
    assert result["xagg"].equals(expected["xagg"])
    assert result["combination"] == expected["combination"]
    assert result["kruskal"] == expected["kruskal"]

    expected = pd.Series({"a": [0, 2, 0], "b to c": [2, 1, 2, 0]})
    assert feature.labels == list(expected.index)
    assert evaluator.samples.train.xagg.equals(expected)
    assert evaluator.samples.dev.xagg.equals(expected)
    assert evaluator.samples.dev.raw.equals(expected)
    assert evaluator.samples.train.raw.equals(expected)


def test_best_association_with_combinations_non_viable(evaluator: ContinuousCombinationEvaluator):
    """Test the best association with no viable combinations"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])
    evaluator.feature = feature

    xagg = pd.Series({"a": [0, 2, 0], "b": [2, 1], "c": [2, 0]})
    evaluator.samples.train = AggregatedSample(xagg)

    xagg_dev = pd.Series({"a": [0, 2, 10], "b": [2, 1], "c": [2, 0]})
    evaluator.samples.dev = AggregatedSample(xagg_dev)

    combinations = consecutive_combinations(feature.labels, 2)

    result = evaluator._get_best_association(combinations)
    print(result)

    assert result is None
    assert feature.labels == list(xagg.index)
    assert evaluator.samples.train.xagg.equals(xagg)
    assert evaluator.samples.dev.xagg.equals(xagg_dev)
    assert evaluator.samples.dev.raw.equals(xagg_dev)
    assert evaluator.samples.train.raw.equals(xagg)


def test_best_association_with_nan_combinations_viable(evaluator: ContinuousCombinationEvaluator):
    """Test the best association with a feature that has np.nan values"""

    feature = OrdinalFeature("feature", ["a", "b"])
    feature.has_nan = True
    feature.dropna = True
    evaluator.feature = feature

    xagg = pd.Series({"a": [0, 2, 0], "b": [2, 1], feature.nan: [2, 0]})
    evaluator.samples.train = AggregatedSample(xagg)

    evaluator.samples.dev = AggregatedSample(xagg)

    combinations = nan_combinations(feature, 2)

    result = evaluator._get_best_association(combinations)
    print(result)

    expected = {
        "xagg": pd.Series({"a": [0, 2, 0, 2, 0], "b": [2, 1]}),
        "combination": [["a", "__NAN__"], ["b"]],
        "kruskal": 0.6999999999999975,
    }
    assert result["xagg"].equals(expected["xagg"])
    assert result["combination"] == expected["combination"]
    assert result["kruskal"] == expected["kruskal"]
    print(evaluator.samples.train.xagg)

    expected = pd.Series({f"a, {feature.nan}": [0, 2, 0, 2, 0], "b": [2, 1]})
    assert feature.labels == list(expected.index)
    assert evaluator.samples.train.xagg.equals(expected)
    assert evaluator.samples.dev.xagg.equals(expected)
    assert evaluator.samples.dev.raw.equals(expected)
    assert evaluator.samples.train.raw.equals(expected)


def test_get_best_combination_non_nan_viable(evaluator: ContinuousCombinationEvaluator):
    """Test the get_best_combination method with a feature that has no np.nan values"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])
    evaluator.feature = feature

    xagg = pd.Series({"a": [0, 2, 0], "b": [2, 1], "c": [2, 0]})
    evaluator.samples.train = AggregatedSample(xagg)
    evaluator.samples.dev = AggregatedSample(xagg)

    evaluator.max_n_mod = 2

    result = evaluator._get_best_combination_non_nan()
    print(result)

    expected = {
        "xagg": pd.Series({"a": [0, 2, 0], "b": [2, 1, 2, 0]}),
        "combination": [["a"], ["b", "c"]],
        "kruskal": 0.5833333333333333,
    }
    assert result["xagg"].equals(expected["xagg"])
    assert result["combination"] == expected["combination"]
    assert result["kruskal"] == expected["kruskal"]

    expected = pd.Series({"a": [0, 2, 0], "b to c": [2, 1, 2, 0]})
    assert feature.labels == list(expected.index)
    assert evaluator.samples.train.xagg.equals(expected)
    assert evaluator.samples.dev.xagg.equals(expected)
    assert evaluator.samples.dev.raw.equals(expected)
    assert evaluator.samples.train.raw.equals(expected)


def test_get_best_combination_non_nan_not_viable(evaluator: ContinuousCombinationEvaluator):
    """Test the get_best_combination method with a feature that has no np.nan values"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])
    evaluator.feature = feature

    xagg = pd.Series({"a": [0, 2, 0]})
    evaluator.samples.train = AggregatedSample(xagg)
    evaluator.samples.dev = AggregatedSample(xagg)

    evaluator.max_n_mod = 2

    result = evaluator._get_best_combination_non_nan()
    assert result is None


def test_get_best_combination_non_nan_viable_with_nan(evaluator: ContinuousCombinationEvaluator):
    """Test the get_best_combination method with a feature that has no np.nan values"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])
    feature.has_nan = True
    feature.dropna = True
    evaluator.feature = feature

    xagg = pd.Series({"a": [0, 2, 0], "b": [2, 1], "c": [2, 0], feature.nan: [-1, 5]})
    evaluator.samples.train = AggregatedSample(xagg)
    evaluator.samples.dev = AggregatedSample(xagg)
    evaluator.max_n_mod = 2

    result = evaluator._get_best_combination_non_nan()
    print(result)

    expected = {
        "xagg": pd.Series({"a": [0, 2, 0], "b": [2, 1, 2, 0]}),
        "combination": [["a"], ["b", "c"]],
        "kruskal": 0.5833333333333333,
    }
    assert result["xagg"].equals(expected["xagg"])
    assert result["combination"] == expected["combination"]
    assert result["kruskal"] == expected["kruskal"]

    expected = pd.Series({"a": [0, 2, 0], "b to c": [2, 1, 2, 0], feature.nan: [-1, 5]})
    print(evaluator.samples.train.xagg)
    assert feature.labels == list(expected.index)
    assert evaluator.samples.train.xagg.equals(expected)
    assert evaluator.samples.dev.xagg.equals(expected)
    assert evaluator.samples.dev.raw.equals(expected)
    assert evaluator.samples.train.raw.equals(expected)


def test_get_best_combination_with_nan_viable(evaluator: ContinuousCombinationEvaluator):
    """Test the get_best_combination method with a feature that has no np.nan values"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])
    evaluator.feature = feature

    xagg = pd.Series({"a": [0, 2, 0], "b": [2, 1], "c": [2, 0]})
    evaluator.samples.train = AggregatedSample(xagg)
    evaluator.samples.dev = AggregatedSample(xagg)
    evaluator.max_n_mod = 2

    best_combination = evaluator._get_best_combination_non_nan()

    result = evaluator._get_best_combination_with_nan(best_combination)
    print(result)

    expected = {
        "xagg": pd.Series({"a": [0, 2, 0], "b": [2, 1, 2, 0]}),
        "combination": [["a"], ["b", "c"]],
        "kruskal": 0.5833333333333333,
    }
    assert result["xagg"].equals(expected["xagg"])
    assert result["combination"] == expected["combination"]
    assert result["kruskal"] == expected["kruskal"]

    expected = pd.Series({"a": [0, 2, 0], "b to c": [2, 1, 2, 0]})
    print(evaluator.samples.train.xagg)
    assert feature.labels == list(expected.index)
    assert evaluator.samples.train.xagg.equals(expected)
    assert evaluator.samples.dev.xagg.equals(expected)
    assert evaluator.samples.dev.raw.equals(expected)
    assert evaluator.samples.train.raw.equals(expected)


def test_get_best_combination_with_nan_not_viable(evaluator: ContinuousCombinationEvaluator):
    """Test the get_best_combination method with a feature that has no np.nan values"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])
    evaluator.feature = feature

    xagg = pd.Series({"a": [0, 2, 0]})
    evaluator.samples.train = AggregatedSample(xagg)
    evaluator.samples.dev = AggregatedSample(xagg)
    evaluator.max_n_mod = 2

    best_combination = evaluator._get_best_combination_non_nan()
    result = evaluator._get_best_combination_with_nan(best_combination)
    assert result is None


def test_get_best_combination_with_nan_viable_with_nan_without_combi(
    evaluator: ContinuousCombinationEvaluator,
):
    """Test the get_best_combination method with a feature that has no np.nan values"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])
    feature.has_nan = True
    feature.dropna = True
    evaluator.feature = feature

    xagg = pd.Series({"a": [0, 2, 0], "b": [2, 1], "c": [2, 0], feature.nan: [-1, 5]})
    evaluator.samples.train = AggregatedSample(xagg)
    evaluator.samples.dev = AggregatedSample(xagg)
    evaluator.max_n_mod = 2

    # test without best_combination
    result = evaluator._get_best_combination_with_nan(None)
    assert result is None
    evaluator.dropna = True
    result = evaluator._get_best_combination_with_nan(None)
    assert result is None
    feature.has_nan = False
    result = evaluator._get_best_combination_with_nan(None)
    assert result is None

    feature = OrdinalFeature("feature", ["a", "b", "c"])
    feature.has_nan = True
    feature.dropna = False
    evaluator.feature = feature
    evaluator.dropna = True
    result = evaluator._get_best_combination_with_nan(None)
    assert result is None


def test_get_best_combination_with_nan_viable_with_nan_without_feature_nan(
    evaluator: ContinuousCombinationEvaluator,
):
    """Test the get_best_combination method with a feature that has no np.nan values"""

    feature = OrdinalFeature("feature", ["a", "b", "c", "d"])
    feature.has_nan = False
    evaluator.feature = feature

    xagg = pd.Series({"a": [0, 2, 0], "b": [2, 1], "c": [2, 0], "d": [-1, 5]})
    evaluator.samples.train = AggregatedSample(xagg)
    evaluator.samples.dev = AggregatedSample(xagg)
    evaluator.max_n_mod = 2

    best_combination = evaluator._get_best_combination_non_nan()

    result = evaluator._get_best_combination_with_nan(best_combination)
    assert result == best_combination


def test_get_best_combination_with_nan_viable_with_nan_without_dropna(
    evaluator: ContinuousCombinationEvaluator,
):
    """Test the get_best_combination method with a feature that has no np.nan values"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])
    feature.has_nan = False
    evaluator.feature = feature

    xagg = pd.Series({"a": [0, 2, 0], "b": [2, 1], "c": [2, 0], feature.nan: [-1, 5]})
    evaluator.samples.train = AggregatedSample(xagg)
    evaluator.samples.dev = AggregatedSample(xagg)
    evaluator.max_n_mod = 2

    with raises(ValueError):
        best_combination = evaluator._get_best_combination_non_nan()
        evaluator._get_best_combination_with_nan(best_combination)


def test_get_best_combination_with_nan_viable_with_nan(evaluator: ContinuousCombinationEvaluator):
    feature = OrdinalFeature("feature", ["a", "b", "c"])
    feature.has_nan = True
    feature.dropna = True  # has to be set to True
    evaluator.feature = feature
    evaluator.dropna = True

    xagg = pd.Series({"a": [0, 2, 0], "b": [2, 1], "c": [2, 0], feature.nan: [-1, 5]})
    evaluator.samples.train = AggregatedSample(xagg)
    evaluator.samples.dev = AggregatedSample(xagg)
    evaluator.max_n_mod = 2

    best_combination = evaluator._get_best_combination_non_nan()
    result = evaluator._get_best_combination_with_nan(best_combination)
    print(result)
    assert feature.dropna is True

    expected = {
        "xagg": pd.Series({"a": [0, 2, 0], "b to c": [2, 1, 2, 0, -1, 5]}),
        "combination": [["a"], ["b to c", "__NAN__"]],
        "kruskal": 0.2857142857142847,
    }
    assert result["xagg"].equals(expected["xagg"])
    assert result["combination"] == expected["combination"]
    assert result["kruskal"] == expected["kruskal"]

    expected = pd.Series({"a": [0, 2, 0], "b to c, __NAN__": [2, 1, 2, 0, -1, 5]})
    print(evaluator.samples.train.xagg)
    assert feature.labels == list(expected.index)
    assert evaluator.samples.train.xagg.equals(expected)
    assert evaluator.samples.dev.xagg.equals(expected)
    assert evaluator.samples.dev.raw.equals(expected)
    assert evaluator.samples.train.raw.equals(expected)


def test_get_best_combination_viable(evaluator: ContinuousCombinationEvaluator):
    """Test the get_best_combination method with a feature that has no np.nan values"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])

    xagg = pd.Series({"a": [0, 2, 0], "b": [2, 1], "c": [2, 0]})
    evaluator.max_n_mod = 2

    result = evaluator.get_best_combination(feature, xagg, xagg)
    print(result)
    assert evaluator.feature == feature
    assert feature.dropna is evaluator.dropna

    expected = {
        "xagg": pd.Series({"a": [0, 2, 0], "b": [2, 1, 2, 0]}),
        "combination": [["a"], ["b", "c"]],
        "kruskal": 0.5833333333333333,
    }
    assert result["xagg"].equals(expected["xagg"])
    assert result["combination"] == expected["combination"]
    assert result["kruskal"] == expected["kruskal"]

    expected = pd.Series({"a": [0, 2, 0], "b to c": [2, 1, 2, 0]})
    print(evaluator.samples.train.xagg)
    assert feature.labels == list(expected.index)
    assert evaluator.samples.train.xagg.equals(expected)
    assert evaluator.samples.dev.xagg.equals(expected)
    assert evaluator.samples.dev.raw.equals(expected)
    assert evaluator.samples.train.raw.equals(expected)


def test_get_best_combination_viable_without_dev(evaluator: ContinuousCombinationEvaluator):
    """Test the get_best_combination method with a feature that has no np.nan values"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])

    xagg = pd.Series({"a": [0, 2, 0], "b": [2, 1], "c": [2, 0]})
    evaluator.max_n_mod = 2

    result = evaluator.get_best_combination(feature, xagg)
    print(result)
    assert evaluator.feature == feature
    assert feature.dropna is evaluator.dropna

    expected = {
        "xagg": pd.Series({"a": [0, 2, 0], "b": [2, 1, 2, 0]}),
        "combination": [["a"], ["b", "c"]],
        "kruskal": 0.5833333333333333,
    }
    assert result["xagg"].equals(expected["xagg"])
    assert result["combination"] == expected["combination"]
    assert result["kruskal"] == expected["kruskal"]

    expected = pd.Series({"a": [0, 2, 0], "b to c": [2, 1, 2, 0]})
    print(evaluator.samples.train.xagg)
    assert feature.labels == list(expected.index)
    assert evaluator.samples.train.xagg.equals(expected)
    assert evaluator.samples.dev.xagg is None
    assert evaluator.samples.dev.raw is None
    assert evaluator.samples.train.raw.equals(expected)


def test_get_best_combination_not_viable(evaluator: ContinuousCombinationEvaluator):
    """Test the get_best_combination method with a feature that has no np.nan values"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])

    xagg = pd.Series({"a": [0, 2, 0]})
    evaluator.max_n_mod = 2
    result = evaluator.get_best_combination(feature, xagg, xagg)
    assert result is None


def test_get_best_combination_viable_with_nan_without_feature_nan(
    evaluator: ContinuousCombinationEvaluator,
):
    """Test the get_best_combination method with a feature that has no np.nan values"""

    feature = OrdinalFeature("feature", ["a", "b", "c", "d"])
    feature.has_nan = False

    xagg = pd.Series({"a": [0, 2, 0], "b": [2, 1], "c": [2, 0], "d": [-1, 5]})
    evaluator.max_n_mod = 2

    result = evaluator.get_best_combination(feature, xagg)
    print(result)
    assert evaluator.feature == feature
    assert feature.dropna is evaluator.dropna
    expected = {
        "xagg": pd.Series({"a": [0, 2, 0], "b": [2, 1, 2, 0, -1, 5]}),
        "combination": [["a"], ["b", "c", "d"]],
        "kruskal": 0.2857142857142847,
    }
    assert result["xagg"].equals(expected["xagg"])
    assert result["combination"] == expected["combination"]
    assert result["kruskal"] == expected["kruskal"]


def test_get_best_combination_viable_with_nan_without_dropna(
    evaluator: ContinuousCombinationEvaluator,
):
    """Test the get_best_combination method with a feature that has no np.nan values"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])
    feature.has_nan = True

    xagg = pd.Series({"a": [0, 2, 0], "b": [2, 1], "c": [2, 0], feature.nan: [-1, 5]})
    evaluator.max_n_mod = 2
    evaluator.dropna = False

    result = evaluator.get_best_combination(feature, xagg, xagg)
    print(result)
    assert evaluator.feature == feature
    assert feature.dropna is evaluator.dropna
    expected = {
        "xagg": pd.Series({"a": [0, 2, 0], "b": [2, 1, 2, 0]}),
        "combination": [["a"], ["b", "c"]],
        "kruskal": 0.5833333333333333,
    }
    assert result["xagg"].equals(expected["xagg"])
    assert result["combination"] == expected["combination"]
    assert result["kruskal"] == expected["kruskal"]

    expected = pd.Series({"a": [0, 2, 0], "b to c": [2, 1, 2, 0], feature.nan: [-1, 5]})
    print(evaluator.samples.train.xagg)
    assert feature.labels == list(expected.index)
    assert evaluator.samples.train.xagg.equals(expected)
    assert evaluator.samples.dev.xagg.equals(expected)
    assert evaluator.samples.dev.raw.equals(expected)
    assert evaluator.samples.train.raw.equals(expected)


def test_get_best_combination_viable_with_nan(evaluator: ContinuousCombinationEvaluator):
    """Test the get_best_combination method with a feature that has np.nan values"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])
    feature.has_nan = True
    assert feature.dropna is False

    xagg = pd.Series({"a": [0, 2, 0], "b": [2, 1], "c": [2, 0], feature.nan: [-1, 5]})
    evaluator.max_n_mod = 2
    evaluator.dropna = True

    result = evaluator.get_best_combination(feature, xagg, xagg)
    print(result)
    assert evaluator.feature == feature
    assert feature.dropna is True
    assert feature.dropna is evaluator.dropna

    expected = {
        "xagg": pd.Series({"a": [0, 2, 0], "b to c": [2, 1, 2, 0, -1, 5]}),
        "combination": [["a"], ["b to c", "__NAN__"]],
        "kruskal": 0.2857142857142847,
    }
    assert result["xagg"].equals(expected["xagg"])
    assert result["combination"] == expected["combination"]
    assert result["kruskal"] == expected["kruskal"]

    expected = pd.Series({"a": [0, 2, 0], "b to c, __NAN__": [2, 1, 2, 0, -1, 5]})
    print(evaluator.samples.train.xagg)
    assert feature.labels == list(expected.index)
    assert evaluator.samples.train.xagg.equals(expected)
    assert evaluator.samples.dev.xagg.equals(expected)
    assert evaluator.samples.dev.raw.equals(expected)
    assert evaluator.samples.train.raw.equals(expected)
