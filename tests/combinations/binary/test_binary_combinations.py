"""Tests for the binary_combinations module."""

import json

from numpy import allclose, nan, sqrt
from pandas import DataFrame, Series, isna
from pytest import FixtureRequest, fixture, raises
from scipy.stats import chi2_contingency

from AutoCarver.carvers.binary_carver import get_crosstab
from AutoCarver.combinations.binary.binary_combination_evaluators import (
    BinaryCombinationEvaluator,
    CramervCombinations,
    TschuprowtCombinations,
)
from AutoCarver.combinations.continuous.continuous_combination_evaluators import KruskalCombinations
from AutoCarver.combinations.utils.combination_evaluator import AggregatedSample
from AutoCarver.combinations.utils.combinations import consecutive_combinations, nan_combinations
from AutoCarver.combinations.utils.testing import TestKeys, TestMessages
from AutoCarver.features import OrdinalFeature

MAX_N_MOD = 5
MIN_FREQ = 0.2


@fixture(params=[TschuprowtCombinations, CramervCombinations])
def evaluator(request: FixtureRequest) -> BinaryCombinationEvaluator:
    combi_eval = request.param(max_n_mod=MAX_N_MOD)
    combi_eval.min_freq = MIN_FREQ
    return combi_eval


def test_init(evaluator: BinaryCombinationEvaluator):
    assert evaluator.is_y_binary is True
    assert evaluator.is_y_continuous is False
    assert evaluator.sort_by in ["cramerv", "tschuprowt"]

    assert evaluator.max_n_mod == MAX_N_MOD
    assert evaluator.min_freq == MIN_FREQ
    assert evaluator.dropna is False
    assert evaluator.verbose is False
    assert evaluator.feature is None
    assert evaluator.samples.train.xagg is None
    assert evaluator.samples.dev.xagg is None
    assert evaluator.samples.train.raw is None
    assert evaluator.samples.dev.raw is None


def test_to_json(evaluator: BinaryCombinationEvaluator):
    expected_json = {
        "sort_by": evaluator.sort_by,
        "max_n_mod": MAX_N_MOD,
        "dropna": evaluator.dropna,
        "min_freq": MIN_FREQ,
        "verbose": evaluator.verbose,
    }
    assert evaluator.to_json() == expected_json


def test_save(evaluator: BinaryCombinationEvaluator, tmp_path):
    file_name = tmp_path / "test.json"
    evaluator.save(str(file_name))

    with open(file_name, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    expected_json = {
        "sort_by": evaluator.sort_by,
        "max_n_mod": MAX_N_MOD,
        "dropna": evaluator.dropna,
        "min_freq": MIN_FREQ,
        "verbose": evaluator.verbose,
    }
    assert data == expected_json


def test_save_invalid_filename(evaluator: BinaryCombinationEvaluator):
    with raises(ValueError):
        evaluator.save("invalid_file.txt")


def test_load_tschuprowt(tmp_path):
    """Test the load method for TschuprowtCombinations"""
    file_name = tmp_path / "test.json"
    data = {
        "sort_by": "tschuprowt",
        "max_n_mod": MAX_N_MOD,
        "dropna": True,
        "min_freq": MIN_FREQ,
        "verbose": True,
    }

    with open(file_name, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file)

    loaded_evaluator = TschuprowtCombinations.load(str(file_name))

    assert loaded_evaluator.max_n_mod == MAX_N_MOD
    assert loaded_evaluator.dropna is True
    assert loaded_evaluator.min_freq == MIN_FREQ
    assert loaded_evaluator.verbose is True
    assert loaded_evaluator.sort_by == "tschuprowt"
    assert loaded_evaluator.is_y_binary is True
    assert loaded_evaluator.is_y_continuous is False

    with raises(ValueError):
        CramervCombinations.load(str(file_name))

    with raises(ValueError):
        KruskalCombinations.load(str(file_name))


def test_load_cramerv(tmp_path):
    """Test the load method for CrmervCombinations"""
    file_name = tmp_path / "test.json"
    data = {
        "sort_by": "cramerv",
        "max_n_mod": MAX_N_MOD,
        "dropna": True,
        "min_freq": MIN_FREQ,
        "verbose": True,
    }

    with open(file_name, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file)

    loaded_evaluator = CramervCombinations.load(str(file_name))

    assert loaded_evaluator.max_n_mod == MAX_N_MOD
    assert loaded_evaluator.dropna is True
    assert loaded_evaluator.min_freq == MIN_FREQ
    assert loaded_evaluator.verbose is True
    assert loaded_evaluator.sort_by == "cramerv"
    assert loaded_evaluator.is_y_binary is True
    assert loaded_evaluator.is_y_continuous is False

    with raises(ValueError):
        TschuprowtCombinations.load(str(file_name))

    with raises(ValueError):
        KruskalCombinations.load(str(file_name))


def test_compute_target_rates_basic(evaluator: BinaryCombinationEvaluator):
    xagg = DataFrame({0: [10, 20, 30], 1: [5, 15, 25]}, index=["a", "b", "c"])
    result = evaluator._compute_target_rates(xagg)
    expected = DataFrame(
        {
            "target_rate": [0.333333, 0.428571, 0.454545],
            "frequency": [0.142857, 0.333333, 0.523810],
        },
        index=["a", "b", "c"],
    )
    print(result)
    assert allclose(result, expected)


def test_compute_target_rates_empty(evaluator: BinaryCombinationEvaluator):
    xagg = DataFrame(columns=[0, 1])
    result = evaluator._compute_target_rates(xagg)
    expected = DataFrame(columns=["target_rate", "frequency"])
    assert result.equals(expected)


def test_compute_target_rates_none(evaluator: BinaryCombinationEvaluator):
    result = evaluator._compute_target_rates(None)
    assert result is None


def test_compute_target_rates_single_row(evaluator: BinaryCombinationEvaluator):
    xagg = DataFrame({0: [10], 1: [5]}, index=["a"])
    result = evaluator._compute_target_rates(xagg)
    expected = DataFrame({"target_rate": [0.333333], "frequency": [1.0]}, index=["a"])
    print(result)
    assert allclose(result, expected)


def test_compute_target_rates_single_column(evaluator: BinaryCombinationEvaluator):
    xagg = DataFrame({1: [5, 15, 25]}, index=["a", "b", "c"])
    result = evaluator._compute_target_rates(xagg)
    expected = DataFrame(
        {"target_rate": [1.0, 1.0, 1.0], "frequency": [0.111111, 0.333333, 0.555556]},
        index=["a", "b", "c"],
    )
    print(result)
    assert allclose(result, expected)


def test_compute_target_rates_with_nan(evaluator: BinaryCombinationEvaluator):
    X = DataFrame({"feature": ["a", "b", "a", "b", nan]})
    y = Series([1, 0, 1, 0, 1])
    feature = OrdinalFeature("feature", ["a", "b", "c"])
    xagg = get_crosstab(X, y, feature)
    print(xagg)
    result = evaluator._compute_target_rates(xagg)
    expected = DataFrame(
        {"target_rate": [1.0, 0.0, nan], "frequency": [0.5, 0.5, 0.0]},
        index=["a", "b", "c"],
    )
    print(result)
    assert result.equals(expected)


def test_compute_target_rates_all_nan(evaluator: BinaryCombinationEvaluator):
    """Test with a xagg with all nan values"""
    X = DataFrame({"feature": ["a", "b", "a", "b", "c"]})
    y = Series([nan, nan, nan, nan, nan])
    feature = OrdinalFeature("feature", ["a", "b", "c"])
    xagg = get_crosstab(X, y, feature)
    with raises(KeyError):
        evaluator._compute_target_rates(xagg)


def test_compute_target_rates_some_nan(evaluator: BinaryCombinationEvaluator):
    """Test with a xagg with some nan values"""
    X = DataFrame({"feature": ["a", "b", "a", "b", "c"]})
    y = Series([1, nan, 1, 0, 1])
    feature = OrdinalFeature("feature", ["a", "b", "c"])
    xagg = get_crosstab(X, y, feature)
    result = evaluator._compute_target_rates(xagg)
    expected = DataFrame(
        {"target_rate": [1.0, 0, 1.0], "frequency": [0.5, 0.25, 0.25]},
        index=["a", "b", "c"],
    )
    print(result)
    assert allclose(result, expected)


def test_association_measure_basic(evaluator: BinaryCombinationEvaluator):
    """Test with a basic xagg"""
    xagg = DataFrame({"A": [10, 20, 30], "B": [5, 15, 25]})
    n_obs = 105
    result = evaluator._association_measure(xagg, n_obs)

    tol = 1e-10
    chi2 = chi2_contingency(xagg.values + tol)[0]
    cramerv = sqrt(chi2 / n_obs)
    tschuprowt = cramerv / sqrt(sqrt(xagg.shape[0] - 1))
    expected = {"cramerv": round(cramerv / tol) * tol, "tschuprowt": round(tschuprowt / tol) * tol}
    assert result == expected


def test_association_measure_with_zeros(evaluator: BinaryCombinationEvaluator):
    """Test with a xagg with zeros"""
    xagg = DataFrame({"A": [10, 0, 30], "B": [5, 15, 0]})
    n_obs = 105
    result = evaluator._association_measure(xagg, n_obs)

    tol = 1e-10
    chi2 = chi2_contingency(xagg.values + tol)[0]
    cramerv = sqrt(chi2 / n_obs)
    tschuprowt = cramerv / sqrt(sqrt(xagg.shape[0] - 1))
    expected = {"cramerv": round(cramerv / tol) * tol, "tschuprowt": round(tschuprowt / tol) * tol}

    assert result == expected


def test_association_measure_with_nan(evaluator: BinaryCombinationEvaluator):
    """Test with a xagg with nan values"""
    xagg = DataFrame({"A": [10, nan, 30], "B": [5, 15, nan]})
    n_obs = 105
    result = evaluator._association_measure(xagg, n_obs)

    assert isna(result.get("cramerv"))
    assert isna(result.get("tschuprowt"))


def test_association_measure_single_row(evaluator: BinaryCombinationEvaluator):
    """Test with a single row"""
    xagg = DataFrame({"A": [10], "B": [5]})
    n_obs = 15
    result = evaluator._association_measure(xagg, n_obs)
    assert isna(result.get("tschuprowt"))
    assert result.get("cramerv") == 0


def test_association_measure_single_column(evaluator: BinaryCombinationEvaluator):
    """Test with a single column"""
    xagg = DataFrame({"A": [10, 20, 30]})
    n_obs = 60
    result = evaluator._association_measure(xagg, n_obs)
    expected = {"cramerv": 0, "tschuprowt": 0}
    assert result == expected


def test_grouper_basic(evaluator: BinaryCombinationEvaluator):
    """Test with a basic groupby"""
    xagg = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=["a", "b", "c"])
    groupby = {"a": "group1", "b": "group1", "c": "group2"}
    result = evaluator._grouper(xagg, groupby)
    expected = DataFrame({"A": [3, 3], "B": [9, 6]}, index=["group1", "group2"])
    print(result)
    assert all(result.index == expected.index)
    assert all(result.columns == expected.columns)
    for column in result.columns:
        assert (result[column] == expected[column]).all()


def test_grouper_with_duplicates(evaluator: BinaryCombinationEvaluator):
    """Test with a groupby that has duplicates"""
    xagg = DataFrame({"A": [1, 2, 3, 4], "B": [4, 5, 6, 7]}, index=["a", "b", "a", "c"])
    groupby = {"a": "group1", "b": "group1", "c": "group2"}
    result = evaluator._grouper(xagg, groupby)
    expected = DataFrame({"A": [6, 4], "B": [15, 7]}, index=["group1", "group2"])
    assert allclose(result, expected)


def test_grouper_no_groupby(evaluator: BinaryCombinationEvaluator):
    """Test with no groupby"""
    xagg = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=["a", "b", "c"])
    groupby = {}
    result = evaluator._grouper(xagg, groupby)
    expected = xagg.copy()
    assert allclose(result, expected)


def test_grouper_partial_groupby(evaluator: BinaryCombinationEvaluator):
    """Test with a groupby that does not cover all the xagg index"""
    xagg = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=["a", "b", "c"])
    groupby = {"a": "group1"}
    result = evaluator._grouper(xagg, groupby)
    expected = DataFrame({"A": [2, 3, 1], "B": [5, 6, 4]}, index=["b", "c", "group1"])
    print(result)
    assert allclose(result, expected)


def test_group_xagg_by_combinations(evaluator: BinaryCombinationEvaluator):
    """Test the _group_xagg_by_combinations method"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])
    xagg = DataFrame({0: [0, 2, 0], 1: [2, 0, 1]}, index=["a", "b", "c"])

    combinations = consecutive_combinations(feature.labels, 2)

    evaluator.samples.train = AggregatedSample(xagg)

    result = evaluator._group_xagg_by_combinations(combinations)
    print(result)

    expected = [
        {
            "xagg": DataFrame({0: [0, 2], 1: [2, 1]}, index=["a", "b"]),
            "combination": [["a"], ["b", "c"]],
            "index_to_groupby": {"a": "a", "b": "b", "c": "b"},
        },
        {
            "xagg": DataFrame({0: [2, 0], 1: [2, 1]}, index=["a", "c"]),
            "combination": [["a", "b"], ["c"]],
            "index_to_groupby": {"a": "a", "b": "a", "c": "c"},
        },
    ]
    for res, exp in zip(result, expected):
        assert allclose(res["xagg"], exp["xagg"])
        assert res["combination"] == exp["combination"]
        assert res["index_to_groupby"] == exp["index_to_groupby"]


def test_group_xagg_by_combinations_with_nan(evaluator: BinaryCombinationEvaluator):
    """Test with a xagg with nan values"""

    feature = OrdinalFeature("feature", ["A", "B", "C"])
    xagg = DataFrame({0: [0, 2, 0], 1: [2, 0, 0]}, index=["A", "B", "C"])

    combinations = consecutive_combinations(feature.labels, 3)

    evaluator.samples.train = AggregatedSample(xagg)

    result = evaluator._group_xagg_by_combinations(combinations)
    print(result)

    expected = [
        {
            "xagg": DataFrame({0: [0, 2, 0], 1: [2, 0, 0]}, index=["A", "B", "C"]),
            "combination": [["A"], ["B"], ["C"]],
            "index_to_groupby": {"A": "A", "B": "B", "C": "C"},
        },
        {
            "xagg": DataFrame({0: [0, 2], 1: [2, 0]}, index=["A", "B"]),
            "combination": [["A"], ["B", "C"]],
            "index_to_groupby": {"A": "A", "B": "B", "C": "B"},
        },
        {
            "xagg": DataFrame({0: [2, 0], 1: [2, 0]}, index=["A", "C"]),
            "combination": [["A", "B"], ["C"]],
            "index_to_groupby": {"A": "A", "B": "A", "C": "C"},
        },
    ]
    for res, exp in zip(result, expected):
        assert allclose(res["xagg"], exp["xagg"])
        assert res["combination"] == exp["combination"]
        assert res["index_to_groupby"] == exp["index_to_groupby"]


def test_compute_associations(evaluator: BinaryCombinationEvaluator):
    """Test the compute_associations method"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])
    xagg = DataFrame({0: [0, 2, 0], 1: [2, 0, 1]}, index=["a", "b", "c"])

    combinations = consecutive_combinations(feature.labels, 2)

    evaluator.samples.train = AggregatedSample(xagg)

    grouped_xaggs = evaluator._group_xagg_by_combinations(combinations)
    result = evaluator._compute_associations(grouped_xaggs)
    print(result)

    expected = [
        {
            "xagg": DataFrame({0: [0, 2], 1: [2, 1]}, index=["a", "b"]),
            "combination": [["a"], ["b", "c"]],
            "index_to_groupby": {"a": "a", "b": "b", "c": "b"},
            "cramerv": 0.25,
            "tschuprowt": 0.25,
        },
        {
            "xagg": DataFrame({0: [2, 0], 1: [2, 1]}, index=["a", "c"]),
            "combination": [["a", "b"], ["c"]],
            "index_to_groupby": {"a": "a", "b": "a", "c": "c"},
            "cramerv": 0.0,
            "tschuprowt": 0.0,
        },
    ]
    for res, exp in zip(result, expected):
        assert allclose(res["xagg"], exp["xagg"])
        assert res["combination"] == exp["combination"]
        assert res["index_to_groupby"] == exp["index_to_groupby"]
        assert res["cramerv"] == exp["cramerv"]
        assert res["tschuprowt"] == exp["tschuprowt"]


def test_compute_associations_with_three_rows(evaluator: BinaryCombinationEvaluator):
    """Test with a small xagg"""

    feature = OrdinalFeature("feature", ["A", "B", "C"])
    xagg = DataFrame({0: [0, 2, 0], 1: [2, 0, 0]}, index=["A", "B", "C"])
    evaluator.samples.train = AggregatedSample(xagg)

    combinations = consecutive_combinations(feature.labels, 3)

    grouped_xaggs = evaluator._group_xagg_by_combinations(combinations)

    # adding an observation to the xagg
    xagg = DataFrame({0: [0, 2, 0], 1: [2, 0, 1]}, index=["A", "B", "C"])
    evaluator.samples.train = AggregatedSample(xagg)

    grouped_xaggs = evaluator._group_xagg_by_combinations(combinations)
    result = evaluator._compute_associations(grouped_xaggs)
    print(result)

    expected = [
        {
            "xagg": DataFrame({0: [0, 2, 0], 1: [2, 0, 1]}, index=["A", "B", "C"]),
            "combination": [["A"], ["B"], ["C"]],
            "index_to_groupby": {"A": "A", "B": "B", "C": "C"},
            "cramerv": 0.9999999999,
            "tschuprowt": 0.8408964152,
        },
        {
            "xagg": DataFrame({0: [0, 2], 1: [2, 1]}, index=["A", "B"]),
            "combination": [["A"], ["B", "C"]],
            "index_to_groupby": {"A": "A", "B": "B", "C": "B"},
            "cramerv": 0.25,
            "tschuprowt": 0.25,
        },
        {
            "xagg": DataFrame({0: [2, 0], 1: [2, 1]}, index=["A", "C"]),
            "combination": [["A", "B"], ["C"]],
            "index_to_groupby": {"A": "A", "B": "A", "C": "C"},
            "cramerv": 0.0,
            "tschuprowt": 0.0,
        },
    ]
    for res, exp in zip(result, expected):
        assert allclose(res["xagg"], exp["xagg"])
        assert res["combination"] == exp["combination"]
        assert res["index_to_groupby"] == exp["index_to_groupby"]
        assert res["cramerv"] == exp["cramerv"]
        assert res["tschuprowt"] == exp["tschuprowt"]


def test_compute_associations_with_twenty_rows(evaluator: BinaryCombinationEvaluator):
    """Test with a larger xagg"""
    feature = OrdinalFeature("feature", [chr(i) for i in range(65, 85)])  # A to T
    xagg = DataFrame(
        {
            0: [0, 2, 0, 1, 3, 0, 2, 1, 6, 2, 20, 0, 2, 1, 0, 3, 30, 0, 10, 1],
            1: [5, 6, 1, 1, 2, 1, 0, 2, 1, 4, 5, 1, 0, 2, 1, 6, 2, 8, 0, 2],
        },
        index=[chr(i) for i in range(65, 85)],
    )  # A to T
    evaluator.samples.train = AggregatedSample(xagg)

    combinations = consecutive_combinations(feature.labels, 7)

    grouped_xaggs = evaluator._group_xagg_by_combinations(combinations)
    result = evaluator._compute_associations(grouped_xaggs)
    print("-------sorting by", evaluator.sort_by)
    print(result[0])

    expected = {
        "tschuprowt": {
            "xagg": DataFrame(
                {
                    0: [2, 37, 4, 30, 0, 11],
                    1: [12, 17, 9, 2, 8, 2],
                },
                index=["A", "D", "N", "Q", "R", "S"],
            ),
            "combination": [
                ["A", "B", "C"],
                ["D", "E", "F", "G", "H", "I", "J", "K", "L", "M"],
                ["N", "O", "P"],
                ["Q"],
                ["R"],
                ["S", "T"],
            ],
            "index_to_groupby": {
                "A": "A",
                "B": "A",
                "C": "A",
                "D": "D",
                "E": "D",
                "F": "D",
                "G": "D",
                "H": "D",
                "I": "D",
                "J": "D",
                "K": "D",
                "L": "D",
                "M": "D",
                "N": "N",
                "O": "N",
                "P": "N",
                "Q": "Q",
                "R": "R",
                "S": "S",
                "T": "S",
            },
            "cramerv": 0.6095153634,
            "tschuprowt": 0.40760749,
        },
        "cramerv": {
            "xagg": DataFrame(
                {
                    0: [2, 37, 4, 30, 0, 10, 1],
                    1: [12, 17, 9, 2, 8, 0, 2],
                },
                index=["A", "D", "N", "Q", "R", "S", "T"],
            ),
            "combination": [
                ["A", "B", "C"],
                ["D", "E", "F", "G", "H", "I", "J", "K", "L", "M"],
                ["N", "O", "P"],
                ["Q"],
                ["R"],
                ["S"],
                ["T"],
            ],
            "index_to_groupby": {
                "A": "A",
                "B": "A",
                "C": "A",
                "D": "D",
                "E": "D",
                "F": "D",
                "G": "D",
                "H": "D",
                "I": "D",
                "J": "D",
                "K": "D",
                "L": "D",
                "M": "D",
                "N": "N",
                "O": "N",
                "P": "N",
                "Q": "Q",
                "R": "R",
                "S": "S",
                "T": "T",
            },
            "cramerv": 0.6357922703000001,
            "tschuprowt": 0.40623508680000003,
        },
    }
    res = result[0]
    exp = expected[evaluator.sort_by]
    assert allclose(res["xagg"], exp["xagg"])
    assert res["combination"] == exp["combination"]
    assert res["index_to_groupby"] == exp["index_to_groupby"]
    assert res["cramerv"] == exp["cramerv"]
    assert res["tschuprowt"] == exp["tschuprowt"]


def test_viability_train(evaluator: BinaryCombinationEvaluator):
    """Test the viability of the combination on xagg_train"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])
    xagg = DataFrame({0: [0, 2, 0], 1: [2, 0, 1]}, index=["a", "b", "c"])

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
            "train": {TestKeys.VIABLE.value: True},
            "train_rates": DataFrame(
                {"target_rate": [1.0, 0.333333], "frequency": [0.4, 0.6]}, index=["a", "c"]
            ),
        },
        {
            "train": {TestKeys.VIABLE.value: True},
            "train_rates": DataFrame(
                {"target_rate": [0.5, 1.0], "frequency": [0.8, 0.2]}, index=["a", "c"]
            ),
        },
    ]
    for res, exp in zip(result, expected):
        assert allclose(res["train_rates"], exp["train_rates"])
        assert res["train"][TestKeys.VIABLE.value] == exp["train"][TestKeys.VIABLE.value]


def test_viability_dev(evaluator: BinaryCombinationEvaluator):
    """Test the viability of the combination on xagg_dev"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])
    xagg = DataFrame({0: [0, 2, 0], 1: [2, 0, 1]}, index=["a", "b", "c"])

    combinations = consecutive_combinations(feature.labels, 2)

    evaluator.samples.train = AggregatedSample(xagg)

    grouped_xaggs = evaluator._group_xagg_by_combinations(combinations)
    associations = evaluator._compute_associations(grouped_xaggs)

    # test with no xagg_dev
    for combination in associations:
        test_results = evaluator._test_viability_train(combination)
        test_results = evaluator._test_viability_dev(test_results, combination)
        assert test_results.get("dev").get(TestKeys.VIABLE.value) is None

    # test with xagg_dev but not viable on train
    evaluator.samples.dev = AggregatedSample(
        DataFrame({0: [0, 2, 0], 1: [2, 0, 1]}, index=["a", "b", "c"])
    )
    for combination in associations:
        test_results = evaluator._test_viability_dev(
            {"train": {TestKeys.VIABLE.value: False}}, combination
        )
        assert test_results.get("dev").get(TestKeys.VIABLE.value) is None

    # test with xagg_dev same as train
    result = []
    for combination in associations:
        test_results = evaluator._test_viability_train(combination)
        result += [evaluator._test_viability_dev(test_results, combination)]
    for res in result:
        assert res["train"][TestKeys.VIABLE.value] is True
        assert res["dev"][TestKeys.VIABLE.value] is True

    # test with xagg_dev wrong
    evaluator.samples.dev = AggregatedSample(
        DataFrame({0: [5, 0, 10], 1: [2, 5, 1]}, index=["a", "b", "c"])
    )
    result = []
    for combination in associations:
        test_results = evaluator._test_viability_train(combination)
        result += [evaluator._test_viability_dev(test_results, combination)]
    print(result)

    expected = [
        {
            "train": {TestKeys.VIABLE.value: True, TestKeys.INFO: TestMessages.PASSED_TESTS},
            "dev": {TestKeys.VIABLE.value: False, TestKeys.INFO: TestMessages.INVERSION_RATES},
        },
        {
            "train": {TestKeys.VIABLE.value: True, TestKeys.INFO: TestMessages.PASSED_TESTS},
            "dev": {TestKeys.VIABLE.value: False, TestKeys.INFO: TestMessages.INVERSION_RATES},
        },
    ]
    for res, exp in zip(result, expected):
        assert res["train"] == exp["train"]
        assert res["dev"] == exp["dev"]


def test_get_viable_combination_without_dev(evaluator: BinaryCombinationEvaluator):
    """Test the get_viable_combination method without a dev xagg"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])
    xagg = DataFrame({0: [0, 2, 0], 1: [2, 0, 1]}, index=["a", "b", "c"])

    combinations = consecutive_combinations(feature.labels, 2)

    evaluator.samples.train = AggregatedSample(xagg)
    evaluator.feature = feature

    grouped_xaggs = evaluator._group_xagg_by_combinations(combinations)
    associations = evaluator._compute_associations(grouped_xaggs)

    # test with no xagg_dev
    result = evaluator._get_viable_combination(associations)
    print(result)
    expected = {
        "xagg": DataFrame({0: [0, 2], 1: [2, 1]}, index=["a", "b"]),
        "combination": [["a"], ["b", "c"]],
        "index_to_groupby": {"a": "a", "b": "b", "c": "b"},
        "cramerv": 0.25,
        "tschuprowt": 0.25,
    }
    assert allclose(result["xagg"], expected["xagg"])
    assert result["combination"] == expected["combination"]
    assert result["index_to_groupby"] == expected["index_to_groupby"]
    assert result["cramerv"] == expected["cramerv"]
    assert result["tschuprowt"] == expected["tschuprowt"]


def test_get_viable_combination_with_non_viable_train(evaluator: BinaryCombinationEvaluator):
    """Test the get_viable_combination method with a non-viable train"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])
    xagg = DataFrame({0: [0, 2, 0], 1: [2, 0, 1]}, index=["a", "b", "c"])
    evaluator.samples.train = AggregatedSample(xagg)

    combinations = consecutive_combinations(feature.labels, 2)

    grouped_xaggs = evaluator._group_xagg_by_combinations(combinations)
    associations = evaluator._compute_associations(grouped_xaggs)

    # test with xagg_dev but not viable on train
    evaluator.min_freq = 0.6
    evaluator.samples.dev = AggregatedSample(xagg)
    result = evaluator._get_viable_combination(associations)
    print(result)

    assert result is None


def test_get_viable_combination_with_viable_train(evaluator: BinaryCombinationEvaluator):
    """Test the get_viable_combination method with a viable train"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])
    xagg = DataFrame({0: [0, 2, 0], 1: [2, 0, 1]}, index=["a", "b", "c"])
    evaluator.samples.train = AggregatedSample(xagg)

    combinations = consecutive_combinations(feature.labels, 2)

    grouped_xaggs = evaluator._group_xagg_by_combinations(combinations)
    associations = evaluator._compute_associations(grouped_xaggs)

    # test with xagg_dev same as train
    evaluator.samples.dev = AggregatedSample(xagg)
    result = evaluator._get_viable_combination(associations)
    print(result)

    expected = {
        "xagg": DataFrame({0: [0, 2], 1: [2, 1]}, index=["a", "b"]),
        "combination": [["a"], ["b", "c"]],
        "index_to_groupby": {"a": "a", "b": "b", "c": "b"},
        "cramerv": 0.25,
        "tschuprowt": 0.25,
    }
    assert allclose(result["xagg"], expected["xagg"])
    assert result["combination"] == expected["combination"]
    assert result["index_to_groupby"] == expected["index_to_groupby"]
    assert result["cramerv"] == expected["cramerv"]
    assert result["tschuprowt"] == expected["tschuprowt"]


def test_get_viable_combination_with_not_viable_dev(evaluator: BinaryCombinationEvaluator):
    """Test the get_viable_combination method with a non-viable dev"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])
    xagg = DataFrame({0: [0, 2, 0], 1: [2, 0, 1]}, index=["a", "b", "c"])
    evaluator.samples.train = AggregatedSample(xagg)

    combinations = consecutive_combinations(feature.labels, 2)

    grouped_xaggs = evaluator._group_xagg_by_combinations(combinations)
    associations = evaluator._compute_associations(grouped_xaggs)

    # test with xagg_dev wrong
    evaluator.samples.dev = AggregatedSample(
        DataFrame({0: [5, 0, 10], 1: [2, 5, 1]}, index=["a", "b", "c"])
    )
    result = evaluator._get_viable_combination(associations)
    print(result)
    assert result is None


def test_apply_best_combination_with_viable(evaluator: BinaryCombinationEvaluator):
    """Test the apply_best_combination method with a viable combination"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])
    evaluator.feature = feature
    xagg = DataFrame({0: [0, 2, 0], 1: [2, 0, 1]}, index=["a", "b", "c"])
    evaluator.samples.train = AggregatedSample(xagg)

    combinations = consecutive_combinations(feature.labels, 2)

    grouped_xaggs = evaluator._group_xagg_by_combinations(combinations)
    associations = evaluator._compute_associations(grouped_xaggs)

    # test with xagg_dev same as train
    evaluator.samples.dev = AggregatedSample(xagg)

    best_combination = evaluator._get_viable_combination(associations)

    evaluator._apply_best_combination(best_combination)

    expected = DataFrame({0: [0, 2], 1: [2, 1]}, index=["a", "b to c"])
    assert feature.labels == list(expected.index)
    assert allclose(evaluator.samples.train.xagg, expected)
    assert allclose(evaluator.samples.dev.xagg, expected)
    assert allclose(evaluator.samples.dev.raw, expected)
    assert allclose(evaluator.samples.train.raw, expected)


def test_apply_best_combination_with_non_viable(evaluator: BinaryCombinationEvaluator):
    """Test the apply best combination with a non-viable combination"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])
    evaluator.feature = feature
    xagg = DataFrame({0: [0, 2, 0], 1: [2, 0, 1]}, index=["a", "b", "c"])
    evaluator.samples.train = AggregatedSample(xagg)

    # test with xagg_dev same as train
    evaluator.samples.dev = AggregatedSample(xagg)

    best_combination = None

    evaluator._apply_best_combination(best_combination)

    expected = xagg
    assert feature.labels == list(expected.index)
    assert allclose(evaluator.samples.train.xagg, expected)
    assert allclose(evaluator.samples.dev.xagg, expected)
    assert allclose(evaluator.samples.dev.raw, xagg)
    assert allclose(evaluator.samples.train.raw, xagg)


def test_best_association_with_combinations_viable(evaluator: BinaryCombinationEvaluator):
    """Test the best association with viable combinations"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])
    evaluator.feature = feature

    xagg = DataFrame({0: [0, 2, 0], 1: [2, 0, 1]}, index=["a", "b", "c"])
    evaluator.samples.train = AggregatedSample(xagg)
    evaluator.samples.dev = AggregatedSample(xagg)

    combinations = consecutive_combinations(feature.labels, 2)

    result = evaluator._get_best_association(combinations)
    print(result)

    expected = {
        "xagg": DataFrame({0: [0, 2], 1: [2, 1]}, index=["a", "b"]),
        "combination": [["a"], ["b", "c"]],
        "cramerv": 0.25,
        "tschuprowt": 0.25,
    }
    assert allclose(result["xagg"], expected["xagg"])
    assert result["combination"] == expected["combination"]
    assert result["cramerv"] == expected["cramerv"]
    assert result["tschuprowt"] == expected["tschuprowt"]

    expected = DataFrame({0: [0, 2], 1: [2, 1]}, index=["a", "b to c"])
    assert feature.labels == list(expected.index)
    assert allclose(evaluator.samples.train.xagg, expected)
    assert allclose(evaluator.samples.dev.xagg, expected)
    assert allclose(evaluator.samples.dev.raw, expected)
    assert allclose(evaluator.samples.train.raw, expected)


def test_best_association_with_combinations_non_viable(evaluator: BinaryCombinationEvaluator):
    """Test the best association with no viable combinations"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])
    evaluator.feature = feature

    xagg = DataFrame({0: [0, 2, 0], 1: [2, 0, 1]}, index=["a", "b", "c"])
    evaluator.samples.train = AggregatedSample(xagg)

    xagg_dev = DataFrame({0: [5, 0, 10], 1: [2, 5, 1]}, index=["a", "b", "c"])
    evaluator.samples.dev = AggregatedSample(xagg_dev)

    combinations = consecutive_combinations(feature.labels, 2)

    result = evaluator._get_best_association(combinations)
    print(result)

    assert result is None
    assert feature.labels == list(xagg.index)
    assert allclose(evaluator.samples.train.xagg, xagg)
    assert allclose(evaluator.samples.dev.xagg, xagg_dev)
    assert allclose(evaluator.samples.dev.raw, xagg_dev)
    assert allclose(evaluator.samples.train.raw, xagg)


def test_best_association_with_nan_combinations_viable(evaluator: BinaryCombinationEvaluator):
    """Test the best association with a feature that has NaN values"""

    feature = OrdinalFeature("feature", ["a", "b"])
    feature.has_nan = True
    feature.dropna = True
    evaluator.feature = feature

    xagg = DataFrame({0: [0, 2, 0], 1: [2, 0, 1]}, index=["a", "b", feature.nan])
    evaluator.samples.train = AggregatedSample(xagg)

    evaluator.samples.dev = AggregatedSample(xagg)

    combinations = nan_combinations(feature, 2)

    result = evaluator._get_best_association(combinations)
    print(result)

    expected = {
        "xagg": DataFrame({0: [0, 2], 1: [3, 0]}, index=["a", "b"]),
        "combination": [["a", "__NAN__"], ["b"]],
        "cramerv": 0.5833333333,
        "tschuprowt": 0.5833333333,
    }
    assert allclose(result["xagg"], expected["xagg"])
    assert result["combination"] == expected["combination"]
    assert result["cramerv"] == expected["cramerv"]
    assert result["tschuprowt"] == expected["tschuprowt"]
    print(evaluator.samples.train.xagg)

    expected = DataFrame({0: [0, 2], 1: [3, 0]}, index=[f"a, {feature.nan}", "b"])
    assert feature.labels == list(expected.index)
    assert allclose(evaluator.samples.train.xagg, expected)
    assert allclose(evaluator.samples.dev.xagg, expected)
    assert allclose(evaluator.samples.dev.raw, expected)
    assert allclose(evaluator.samples.train.raw, expected)


def test_get_best_combination_non_nan_viable(evaluator: BinaryCombinationEvaluator):
    """Test the get_best_combination method with a feature that has no NaN values"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])
    evaluator.feature = feature

    xagg = DataFrame({0: [0, 2, 0], 1: [2, 0, 1]}, index=["a", "b", "c"])
    evaluator.samples.train = AggregatedSample(xagg)

    evaluator.samples.dev = AggregatedSample(xagg)

    evaluator.max_n_mod = 2

    result = evaluator._get_best_combination_non_nan()
    print(result)

    expected = {
        "xagg": DataFrame({0: [0, 2], 1: [2, 1]}, index=["a", "b"]),
        "combination": [["a"], ["b", "c"]],
        "cramerv": 0.25,
        "tschuprowt": 0.25,
    }
    assert allclose(result["xagg"], expected["xagg"])
    assert result["combination"] == expected["combination"]
    assert result["cramerv"] == expected["cramerv"]
    assert result["tschuprowt"] == expected["tschuprowt"]

    expected = DataFrame({0: [0, 2], 1: [2, 1]}, index=["a", "b to c"])
    assert feature.labels == list(expected.index)
    assert allclose(evaluator.samples.train.xagg, expected)
    assert allclose(evaluator.samples.dev.xagg, expected)
    assert allclose(evaluator.samples.dev.raw, expected)
    assert allclose(evaluator.samples.train.raw, expected)


def test_get_best_combination_non_nan_not_viable(evaluator: BinaryCombinationEvaluator):
    """Test the get_best_combination method with a feature that has no NaN values"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])
    evaluator.feature = feature

    xagg = DataFrame({0: [0], 1: [2]}, index=["a"])
    evaluator.samples.train = AggregatedSample(xagg)

    evaluator.samples.dev = AggregatedSample(xagg)

    evaluator.max_n_mod = 2

    result = evaluator._get_best_combination_non_nan()
    assert result is None


def test_get_best_combination_non_nan_viable_with_nan(evaluator: BinaryCombinationEvaluator):
    """Test the get_best_combination method with a feature that has no NaN values"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])
    feature.has_nan = True
    feature.dropna = True
    evaluator.feature = feature

    xagg = DataFrame({0: [0, 2, 0, 0], 1: [2, 0, 1, 3]}, index=["a", "b", "c", feature.nan])
    evaluator.samples.train = AggregatedSample(xagg)

    evaluator.samples.dev = AggregatedSample(xagg)

    evaluator.max_n_mod = 2

    result = evaluator._get_best_combination_non_nan()
    print(result)

    expected = {
        "xagg": DataFrame({0: [0, 2], 1: [2, 1]}, index=["a", "b"]),
        "combination": [["a"], ["b", "c"]],
        "cramerv": 0.25,
        "tschuprowt": 0.25,
    }
    assert allclose(result["xagg"], expected["xagg"])
    assert result["combination"] == expected["combination"]
    assert result["cramerv"] == expected["cramerv"]
    assert result["tschuprowt"] == expected["tschuprowt"]

    expected = DataFrame({0: [0, 2, 0], 1: [2, 1, 3]}, index=["a", "b to c", feature.nan])
    assert feature.labels == list(expected.index)
    assert allclose(evaluator.samples.train.xagg, expected)
    assert allclose(evaluator.samples.dev.xagg, expected)
    assert allclose(evaluator.samples.dev.raw, expected)
    assert allclose(evaluator.samples.train.raw, expected)


def test_get_best_combination_with_nan_viable(evaluator: BinaryCombinationEvaluator):
    """Test the get_best_combination method with a feature that has no NaN values"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])
    evaluator.feature = feature

    xagg = DataFrame({0: [0, 2, 0], 1: [2, 0, 1]}, index=["a", "b", "c"])
    evaluator.samples.train = AggregatedSample(xagg)

    evaluator.samples.dev = AggregatedSample(xagg)

    evaluator.max_n_mod = 2

    best_combination = evaluator._get_best_combination_non_nan()

    result = evaluator._get_best_combination_with_nan(best_combination)
    print(result)

    expected = {
        "xagg": DataFrame({0: [0, 2], 1: [2, 1]}, index=["a", "b"]),
        "combination": [["a"], ["b", "c"]],
        "index_to_groupby": {"a": "a", "b": "b", "c": "b"},
        "cramerv": 0.25,
        "tschuprowt": 0.25,
    }
    assert allclose(result["xagg"], expected["xagg"])
    assert result["combination"] == expected["combination"]
    assert result["index_to_groupby"] == expected["index_to_groupby"]
    assert result["cramerv"] == expected["cramerv"]
    assert result["tschuprowt"] == expected["tschuprowt"]

    expected = DataFrame({0: [0, 2], 1: [2, 1]}, index=["a", "b to c"])
    assert feature.labels == list(expected.index)
    assert allclose(evaluator.samples.train.xagg, expected)
    assert allclose(evaluator.samples.dev.xagg, expected)
    assert allclose(evaluator.samples.dev.raw, expected)
    assert allclose(evaluator.samples.train.raw, expected)


def test_get_best_combination_with_nan_not_viable(evaluator: BinaryCombinationEvaluator):
    """Test the get_best_combination method with a feature that has no NaN values"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])
    evaluator.feature = feature

    xagg = DataFrame({0: [0], 1: [2]}, index=["a"])
    evaluator.samples.train = AggregatedSample(xagg)

    evaluator.samples.dev = AggregatedSample(xagg)

    evaluator.max_n_mod = 2

    best_combination = evaluator._get_best_combination_non_nan()
    result = evaluator._get_best_combination_with_nan(best_combination)
    assert result is None


def test_get_best_combination_with_nan_viable_with_nan_without_combi(
    evaluator: BinaryCombinationEvaluator,
):
    """Test the get_best_combination method with a feature that has no NaN values"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])
    feature.has_nan = True
    feature.dropna = True
    evaluator.feature = feature

    xagg = DataFrame({0: [0, 2, 0, 0], 1: [2, 0, 1, 3]}, index=["a", "b", "c", feature.nan])
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
    evaluator: BinaryCombinationEvaluator,
):
    """Test the get_best_combination method with a feature that has no NaN values"""

    feature = OrdinalFeature("feature", ["a", "b", "c", "d"])
    feature.has_nan = False
    evaluator.feature = feature

    xagg = DataFrame({0: [0, 2, 0, 0], 1: [2, 0, 1, 3]}, index=["a", "b", "c", "d"])
    evaluator.samples.train = AggregatedSample(xagg)

    evaluator.samples.dev = AggregatedSample(xagg)

    evaluator.max_n_mod = 2

    best_combination = evaluator._get_best_combination_non_nan()

    result = evaluator._get_best_combination_with_nan(best_combination)
    assert result == best_combination


def test_get_best_combination_with_nan_viable_with_nan_without_dropna(
    evaluator: BinaryCombinationEvaluator,
):
    """Test the get_best_combination method with a feature that has no NaN values"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])
    feature.has_nan = False
    evaluator.feature = feature

    xagg = DataFrame({0: [0, 2, 0, 0], 1: [2, 0, 1, 3]}, index=["a", "b", "c", feature.nan])
    evaluator.samples.train = AggregatedSample(xagg)
    evaluator.samples.dev = AggregatedSample(xagg)
    evaluator.max_n_mod = 2

    best_combination = evaluator._get_best_combination_non_nan()

    result = evaluator._get_best_combination_with_nan(best_combination)
    assert result == best_combination


def test_get_best_combination_with_nan_viable_with_nan(evaluator: BinaryCombinationEvaluator):
    """Test the get_best_combination method with a feature that has no NaN values"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])
    feature.has_nan = True
    feature.dropna = True  # has to be set to True
    evaluator.feature = feature
    evaluator.dropna = True

    xagg = DataFrame({0: [0, 2, 0, 0], 1: [2, 0, 1, 3]}, index=["a", "b", "c", feature.nan])
    evaluator.samples.train = AggregatedSample(xagg)

    evaluator.samples.dev = AggregatedSample(xagg)

    evaluator.max_n_mod = 2

    best_combination = evaluator._get_best_combination_non_nan()

    result = evaluator._get_best_combination_with_nan(best_combination)
    print(result)
    assert feature.dropna is True

    expected = {
        "xagg": DataFrame({0: [0, 2], 1: [5, 1]}, index=["a", "b to c"]),
        "combination": [["a", "__NAN__"], ["b to c"]],
        "cramerv": 0.4472135955,
        "tschuprowt": 0.4472135955,
    }

    assert allclose(result["xagg"], expected["xagg"])
    assert result["combination"] == expected["combination"]
    assert result["cramerv"] == expected["cramerv"]
    assert result["tschuprowt"] == expected["tschuprowt"]

    expected = DataFrame({0: [0, 2], 1: [5, 1]}, index=["a, __NAN__", "b to c"])
    assert feature.labels == list(expected.index)
    assert allclose(evaluator.samples.train.xagg, expected)
    assert allclose(evaluator.samples.dev.xagg, expected)
    assert allclose(evaluator.samples.dev.raw, expected)
    assert allclose(evaluator.samples.train.raw, expected)


def test_get_best_combination_viable(evaluator: BinaryCombinationEvaluator):
    """Test the get_best_combination method with a feature that has no NaN values"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])

    xagg = DataFrame({0: [0, 2, 0], 1: [2, 0, 1]}, index=["a", "b", "c"])
    evaluator.max_n_mod = 2

    result = evaluator.get_best_combination(feature, xagg, xagg)
    print(result)
    assert evaluator.feature == feature
    assert feature.dropna is evaluator.dropna

    expected = {
        "xagg": DataFrame({0: [0, 2], 1: [2, 1]}, index=["a", "b"]),
        "combination": [["a"], ["b", "c"]],
        "cramerv": 0.25,
        "tschuprowt": 0.25,
    }
    assert allclose(result["xagg"], expected["xagg"])
    assert result["combination"] == expected["combination"]
    assert result["cramerv"] == expected["cramerv"]
    assert result["tschuprowt"] == expected["tschuprowt"]

    expected = DataFrame({0: [0, 2], 1: [2, 1]}, index=["a", "b to c"])
    assert feature.labels == list(expected.index)
    assert allclose(evaluator.samples.train.xagg, expected)
    assert allclose(evaluator.samples.dev.xagg, expected)
    assert allclose(evaluator.samples.dev.raw, expected)
    assert allclose(evaluator.samples.train.raw, expected)


def test_get_best_combination_viable_without_dev(evaluator: BinaryCombinationEvaluator):
    """Test the get_best_combination method with a feature that has no NaN values"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])

    xagg = DataFrame({0: [0, 2, 0], 1: [2, 0, 1]}, index=["a", "b", "c"])
    evaluator.max_n_mod = 2

    result = evaluator.get_best_combination(feature, xagg)
    print(result)
    assert evaluator.feature == feature
    assert feature.dropna is evaluator.dropna

    expected = {
        "xagg": DataFrame({0: [0, 2], 1: [2, 1]}, index=["a", "b"]),
        "combination": [["a"], ["b", "c"]],
        "cramerv": 0.25,
        "tschuprowt": 0.25,
    }
    assert allclose(result["xagg"], expected["xagg"])
    assert result["combination"] == expected["combination"]
    assert result["cramerv"] == expected["cramerv"]
    assert result["tschuprowt"] == expected["tschuprowt"]

    expected = DataFrame({0: [0, 2], 1: [2, 1]}, index=["a", "b to c"])
    assert feature.labels == list(expected.index)
    assert allclose(evaluator.samples.train.xagg, expected)
    assert evaluator.samples.dev.xagg is None
    assert allclose(evaluator.samples.train.raw, expected)


def test_get_best_combination_not_viable(evaluator: BinaryCombinationEvaluator):
    """Test the get_best_combination method with a feature that has no NaN values"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])

    xagg = DataFrame({0: [0], 1: [2]}, index=["a"])
    evaluator.max_n_mod = 2

    result = evaluator.get_best_combination(feature, xagg, xagg)
    assert result is None


def test_get_best_combination_viable_with_nan_without_feature_nan(
    evaluator: BinaryCombinationEvaluator,
):
    """Test the get_best_combination method with a feature that has no NaN values"""

    feature = OrdinalFeature("feature", ["a", "b", "c", "d"])
    feature.has_nan = False

    xagg = DataFrame({0: [0, 2, 0, 0], 1: [2, 0, 1, 3]}, index=["a", "b", "c", "d"])
    evaluator.max_n_mod = 2

    result = evaluator.get_best_combination(feature, xagg)
    print(result)
    assert evaluator.feature == feature
    assert feature.dropna is evaluator.dropna
    expected = {
        "xagg": DataFrame({0: [2, 0], 1: [2, 4]}, index=["a", "c"]),
        "combination": [["a", "b"], ["c", "d"]],
        "cramerv": 0.2886751346,
        "tschuprowt": 0.2886751346,
    }

    assert allclose(result["xagg"], expected["xagg"])
    assert result["combination"] == expected["combination"]
    assert result["cramerv"] == expected["cramerv"]
    assert result["tschuprowt"] == expected["tschuprowt"]


def test_get_best_combination_viable_with_nan_without_dropna(
    evaluator: BinaryCombinationEvaluator,
):
    """Test the get_best_combination method with a feature that has no NaN values"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])
    feature.has_nan = True

    xagg = DataFrame({0: [0, 2, 0, 0], 1: [2, 0, 1, 3]}, index=["a", "b", "c", feature.nan])
    evaluator.max_n_mod = 2
    evaluator.dropna = False

    result = evaluator.get_best_combination(feature, xagg, xagg)
    print(result)
    assert evaluator.feature == feature
    assert feature.dropna is evaluator.dropna
    expected = {
        "xagg": DataFrame({0: [0, 2], 1: [2, 1]}, index=["a", "b"]),
        "combination": [["a"], ["b", "c"]],
        "cramerv": 0.25,
        "tschuprowt": 0.25,
    }

    assert allclose(result["xagg"], expected["xagg"])
    assert result["combination"] == expected["combination"]
    assert result["cramerv"] == expected["cramerv"]
    assert result["tschuprowt"] == expected["tschuprowt"]

    expected = DataFrame({0: [0, 2, 0], 1: [2, 1, 3]}, index=["a", "b to c", feature.nan])
    assert list(evaluator.samples.train.index) == list(expected.index)
    assert allclose(evaluator.samples.train.xagg, expected)
    assert allclose(evaluator.samples.dev.xagg, expected)
    assert allclose(evaluator.samples.dev.raw, expected)
    assert allclose(evaluator.samples.train.raw, expected)


def test_get_best_combination_viable_with_nan(evaluator: BinaryCombinationEvaluator):
    """Test the get_best_combination method with a feature that has NaN values"""

    feature = OrdinalFeature("feature", ["a", "b", "c"])
    feature.has_nan = True
    assert feature.dropna is False

    xagg = DataFrame({0: [0, 2, 0, 0], 1: [2, 0, 1, 3]}, index=["a", "b", "c", feature.nan])
    evaluator.max_n_mod = 2
    evaluator.dropna = True

    result = evaluator.get_best_combination(feature, xagg, xagg)
    print(result)
    assert evaluator.feature == feature
    assert feature.dropna is True
    assert feature.dropna is evaluator.dropna

    expected = {
        "xagg": DataFrame({0: [0, 2], 1: [5, 1]}, index=["a", "b to c"]),
        "combination": [["a", "__NAN__"], ["b to c"]],
        "cramerv": 0.4472135955,
        "tschuprowt": 0.4472135955,
    }

    assert allclose(result["xagg"], expected["xagg"])
    assert result["combination"] == expected["combination"]
    assert result["cramerv"] == expected["cramerv"]
    assert result["tschuprowt"] == expected["tschuprowt"]

    expected = DataFrame({0: [0, 2], 1: [5, 1]}, index=["a, __NAN__", "b to c"])
    assert feature.labels == list(expected.index)
    assert allclose(evaluator.samples.train.xagg, expected)
    assert allclose(evaluator.samples.dev.xagg, expected)
    assert allclose(evaluator.samples.dev.raw, expected)
    assert allclose(evaluator.samples.train.raw, expected)
