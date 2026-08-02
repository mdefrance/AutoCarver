"""Tests for the binary_combinations module."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
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
from AutoCarver.combinations.utils.combinations import (
    combination_formatter,
    consecutive_combinations,
    nan_combinations,
)
from AutoCarver.combinations.utils.testing import Keys, Messages
from AutoCarver.features import OrdinalFeature

MAX_N_MOD = 5
MIN_FREQ = 0.2


@fixture(params=[TschuprowtCombinations])
def evaluator(request: FixtureRequest) -> BinaryCombinationEvaluator:
    """Fixture for BinaryCombinationEvaluator used in tests."""
    return request.param()


def test_init(evaluator: BinaryCombinationEvaluator):
    """Test initialization of BinaryCombinationEvaluator."""
    assert evaluator.is_y_binary is True
    assert evaluator.is_y_continuous is False
    assert evaluator.sort_by in ["cramerv", "tschuprowt"]
    assert evaluator.verbose is False
    # feature/xagg are properties that raise when unset; check the predicates
    assert evaluator._feature is None
    assert not evaluator.samples.train.has_xagg
    assert not evaluator.samples.dev.has_xagg
    assert evaluator.samples.train.raw is None
    assert evaluator.samples.dev.raw is None


def test_to_json(evaluator: BinaryCombinationEvaluator):
    """Test to_json method of BinaryCombinationEvaluator."""
    expected_json = {
        "sort_by": evaluator.sort_by,
        "target_rate": evaluator.target_rate.__name__,
        "verbose": evaluator.verbose,
    }
    assert evaluator.to_json() == expected_json


def test_save(evaluator: BinaryCombinationEvaluator, tmp_path):
    """Test save method of BinaryCombinationEvaluator."""
    file_name = tmp_path / "test.json"
    evaluator.save(file_name)

    with file_name.open(encoding="utf-8") as json_file:
        data = json.load(json_file)

    expected_json = {
        "sort_by": evaluator.sort_by,
        "target_rate": evaluator.target_rate.__name__,
        "verbose": evaluator.verbose,
    }
    assert data == expected_json


def test_save_invalid_filename(evaluator: BinaryCombinationEvaluator):
    """Test save method with an invalid filename."""
    with raises(ValueError):
        evaluator.save(Path("invalid_file.txt"))


def test_load_tschuprowt(tmp_path):
    """Test the load method for TschuprowtCombinations."""
    file_name = tmp_path / "test.json"
    data = {
        "sort_by": "tschuprowt",
        "target_rate": "odds_ratio",
        "verbose": True,
    }

    with file_name.open("w", encoding="utf-8") as json_file:
        json.dump(data, json_file)

    loaded_evaluator = TschuprowtCombinations.load(file_name)

    assert loaded_evaluator.verbose is True
    assert loaded_evaluator.sort_by == "tschuprowt"
    assert loaded_evaluator.target_rate.__name__ == "odds_ratio"
    assert loaded_evaluator.is_y_binary is True
    assert loaded_evaluator.is_y_continuous is False

    with raises(ValueError):
        CramervCombinations.load(file_name)

    with raises(ValueError):
        KruskalCombinations.load(file_name)


def test_load_cramerv(tmp_path):
    """Test the load method for CramervCombinations."""
    file_name = tmp_path / "test.json"
    data = {
        "sort_by": "cramerv",
        "target_rate": "woe",
        "verbose": True,
    }

    with file_name.open("w", encoding="utf-8") as json_file:
        json.dump(data, json_file)

    loaded_evaluator = CramervCombinations.load(file_name)

    assert loaded_evaluator.verbose is True
    assert loaded_evaluator.sort_by == "cramerv"
    assert loaded_evaluator.target_rate.__name__ == "woe"
    assert loaded_evaluator.is_y_binary is True
    assert loaded_evaluator.is_y_continuous is False

    with raises(ValueError):
        TschuprowtCombinations.load(file_name)

    with raises(ValueError):
        KruskalCombinations.load(file_name)


def test_compute_target_rates_basic(evaluator: BinaryCombinationEvaluator):
    """Test _compute_target_rates with a basic xagg."""
    xagg = pd.DataFrame({0: [10, 20, 30], 1: [5, 15, 25]}, index=["a", "b", "c"])
    result = evaluator.target_rate.compute(xagg)
    expected = pd.DataFrame(
        {
            "target_mean": [0.333333, 0.428571, 0.454545],
            "frequency": [0.142857, 0.333333, 0.523810],
            "count": [15, 35, 55],
        },
        index=["a", "b", "c"],
    )
    print(result)
    assert np.allclose(result, expected)


def test_compute_target_rates_empty(evaluator: BinaryCombinationEvaluator):
    """Test _compute_target_rates with an empty xagg."""
    xagg = pd.DataFrame(columns=[0, 1])
    result = evaluator.target_rate.compute(xagg)
    print(result)
    expected = pd.DataFrame(columns=["target_mean", "frequency", "count"])
    assert list(result.columns) == list(expected.columns)
    assert len(result) == 0


def test_compute_target_rates_none(evaluator: BinaryCombinationEvaluator):
    """Test _compute_target_rates with None."""
    result = evaluator.target_rate.compute(None)
    assert result is None


def test_compute_target_rates_single_row(evaluator: BinaryCombinationEvaluator):
    """Test _compute_target_rates with a single row."""
    xagg = pd.DataFrame({0: [10], 1: [5]}, index=["a"])
    result = evaluator.target_rate.compute(xagg)
    expected = pd.DataFrame({"target_mean": [0.333333], "frequency": [1.0], "count": [15]}, index=["a"])
    print(result)
    assert np.allclose(result, expected)


def test_compute_target_rates_single_column(evaluator: BinaryCombinationEvaluator):
    """Test _compute_target_rates with a single column."""
    xagg = pd.DataFrame({1: [5, 15, 25]}, index=["a", "b", "c"])
    result = evaluator.target_rate.compute(xagg)
    expected = pd.DataFrame(
        {"target_mean": [1.0, 1.0, 1.0], "frequency": [0.111111, 0.333333, 0.555556], "count": [5, 15, 25]},
        index=["a", "b", "c"],
    )
    print(result)
    assert np.allclose(result, expected)


def test_compute_target_rates_with_nan(evaluator: BinaryCombinationEvaluator):
    """Test _compute_target_rates with np.nan values."""
    X = pd.DataFrame({"feature": ["a", "b", "a", "b", np.nan]})
    y = pd.Series([1, 0, 1, 0, 1])
    feature = OrdinalFeature("feature", ["a", "b", "c"])
    xagg = get_crosstab(X, y, feature)
    print(xagg)
    result = evaluator.target_rate.compute(xagg)
    expected = pd.DataFrame(
        {"target_mean": [1.0, 0.0, np.nan], "frequency": [0.5, 0.5, 0.0], "count": [2, 2, 0]},
        index=["a", "b", "c"],
    )
    print(result)
    assert result.equals(expected)


def test_compute_target_rates_all_nan(evaluator: BinaryCombinationEvaluator):
    """Test _compute_target_rates with all np.nan values."""
    X = pd.DataFrame({"feature": ["a", "b", "a", "b", "c"]})
    y = pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan])
    feature = OrdinalFeature("feature", ["a", "b", "c"])
    xagg = get_crosstab(X, y, feature)
    with raises(KeyError):
        evaluator.target_rate.compute(xagg)


def test_compute_target_rates_some_nan(evaluator: BinaryCombinationEvaluator):
    """Test _compute_target_rates with some np.nan values."""
    X = pd.DataFrame({"feature": ["a", "b", "a", "b", "c"]})
    y = pd.Series([1, np.nan, 1, 0, 1])
    feature = OrdinalFeature("feature", ["a", "b", "c"])
    xagg = get_crosstab(X, y, feature)
    result = evaluator.target_rate.compute(xagg)
    expected = pd.DataFrame(
        {"target_mean": [1.0, 0, 1.0], "frequency": [0.5, 0.25, 0.25], "count": [2, 1, 1]},
        index=["a", "b", "c"],
    )
    print(result)
    assert np.allclose(result, expected)


def test_association_measure_basic(evaluator: BinaryCombinationEvaluator):
    """Test _association_measure with a basic xagg."""
    xagg = pd.DataFrame({"A": [10, 20, 30], "B": [5, 15, 25]})
    n_obs = 105
    result = evaluator._association_measure(xagg, n_obs)

    tol = 1e-10
    chi2 = chi2_contingency(xagg.values + tol)[0]
    cramerv = np.sqrt(chi2 / n_obs)
    tschuprowt = cramerv / np.sqrt(np.sqrt(xagg.shape[0] - 1))
    expected = {"cramerv": round(cramerv / tol) * tol, "tschuprowt": round(tschuprowt / tol) * tol}
    assert result == expected


def test_association_measure_with_zeros(evaluator: BinaryCombinationEvaluator):
    """Test _association_measure with zeros in xagg."""
    xagg = pd.DataFrame({"A": [10, 0, 30], "B": [5, 15, 0]})
    n_obs = 105
    result = evaluator._association_measure(xagg, n_obs)

    tol = 1e-10
    chi2 = chi2_contingency(xagg.values + tol)[0]
    cramerv = np.sqrt(chi2 / n_obs)
    tschuprowt = cramerv / np.sqrt(np.sqrt(xagg.shape[0] - 1))
    expected = {"cramerv": round(cramerv / tol) * tol, "tschuprowt": round(tschuprowt / tol) * tol}

    assert result == expected


def test_association_measure_with_nan(evaluator: BinaryCombinationEvaluator):
    """Test _association_measure with np.nan values in xagg."""
    xagg = pd.DataFrame({"A": [10, np.nan, 30], "B": [5, 15, np.nan]})
    n_obs = 105
    result = evaluator._association_measure(xagg, n_obs)

    assert pd.isna(result.get("cramerv"))
    assert pd.isna(result.get("tschuprowt"))


def test_association_measure_single_row(evaluator: BinaryCombinationEvaluator):
    """Test _association_measure with a single row."""
    xagg = pd.DataFrame({"A": [10], "B": [5]})
    n_obs = 15
    result = evaluator._association_measure(xagg, n_obs)
    assert pd.isna(result.get("tschuprowt"))
    assert result.get("cramerv") == 0


def test_association_measure_single_column(evaluator: BinaryCombinationEvaluator):
    """Test _association_measure with a single column."""
    xagg = pd.DataFrame({"A": [10, 20, 30]})
    n_obs = 60
    result = evaluator._association_measure(xagg, n_obs)
    expected = {"cramerv": 0, "tschuprowt": 0}
    assert result == expected


def test_grouper_basic(evaluator: BinaryCombinationEvaluator):
    """Test _grouper with a basic groupby."""
    xagg = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=["a", "b", "c"])
    groupby = {"a": "group1", "b": "group1", "c": "group2"}
    result = evaluator._grouper(xagg, groupby)
    expected = pd.DataFrame({"A": [3, 3], "B": [9, 6]}, index=["group1", "group2"])
    print(result)
    assert all(result.index == expected.index)
    assert all(result.columns == expected.columns)
    for column in result.columns:
        assert (result[column] == expected[column]).all()


def test_grouper_with_duplicates(evaluator: BinaryCombinationEvaluator):
    """Test _grouper with duplicates in groupby."""
    xagg = pd.DataFrame({"A": [1, 2, 3, 4], "B": [4, 5, 6, 7]}, index=["a", "b", "a", "c"])
    groupby = {"a": "group1", "b": "group1", "c": "group2"}
    result = evaluator._grouper(xagg, groupby)
    expected = pd.DataFrame({"A": [6, 4], "B": [15, 7]}, index=["group1", "group2"])
    assert np.allclose(result, expected)


def test_grouper_no_groupby(evaluator: BinaryCombinationEvaluator):
    """Test _grouper with no groupby."""
    xagg = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=["a", "b", "c"])
    groupby = {}
    result = evaluator._grouper(xagg, groupby)
    expected = xagg.copy()
    assert np.allclose(result, expected)


def test_grouper_partial_groupby(evaluator: BinaryCombinationEvaluator):
    """Test _grouper with a partial groupby."""
    xagg = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=["a", "b", "c"])
    groupby = {"a": "group1"}
    result = evaluator._grouper(xagg, groupby)
    expected = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=["group1", "b", "c"])
    print(result)
    assert np.allclose(result, expected)


def _sorted_assocs(evaluator, combinations):
    """Score each combination via the closed-form ``_association_measure`` (the
    surviving scipy-backed formula, kept for the raw-distribution historization and
    the NaN-fanout scoring path) and sort by sort_by desc — matches the order
    `_get_best_association` would feed into the viability walk, which is what
    these tests assert against."""
    n_obs = evaluator.samples.train.xagg.apply(sum).sum()
    scored = []
    for combination in combinations:
        index_to_groupby = combination_formatter(combination)
        grouped = evaluator._grouper(evaluator.samples.train, index_to_groupby)
        measure = evaluator._association_measure(grouped, n_obs=n_obs)
        scored.append({"combination": combination, "index_to_groupby": index_to_groupby, **measure})

    sort_by = evaluator.sort_by
    return sorted(
        scored,
        key=lambda r: r[sort_by] if r[sort_by] is not None and r[sort_by] == r[sort_by] else -float("inf"),
        reverse=True,
    )


def test_compute_associations(evaluator: BinaryCombinationEvaluator):
    """`_association_measure` scores each combination's grouped crosstab; sorting
    by sort_by desc matches what the viability walk consumes."""
    feature = OrdinalFeature("feature", ["a", "b", "c"])
    xagg = pd.DataFrame({0: [0, 2, 0], 1: [2, 0, 1]}, index=["a", "b", "c"])

    combinations = consecutive_combinations(feature.labels, 2)

    evaluator.samples.train = AggregatedSample(xagg)

    result = _sorted_assocs(evaluator, combinations)

    expected = [
        {
            "combination": [["a"], ["b", "c"]],
            "index_to_groupby": {"a": "a", "b": "b", "c": "b"},
            "cramerv": 0.25,
            "tschuprowt": 0.25,
        },
        {
            "combination": [["a", "b"], ["c"]],
            "index_to_groupby": {"a": "a", "b": "a", "c": "c"},
            "cramerv": 0.0,
            "tschuprowt": 0.0,
        },
    ]
    for res, exp in zip(result, expected):
        assert res["combination"] == exp["combination"]
        assert res["index_to_groupby"] == exp["index_to_groupby"]
        assert res["cramerv"] == exp["cramerv"]
        assert res["tschuprowt"] == exp["tschuprowt"]


def test_compute_associations_with_three_rows(evaluator: BinaryCombinationEvaluator):
    """Sorted-by-metric variant on the 3-modality case."""
    feature = OrdinalFeature("feature", ["A", "B", "C"])
    xagg = pd.DataFrame({0: [0, 2, 0], 1: [2, 0, 1]}, index=["A", "B", "C"])
    evaluator.samples.train = AggregatedSample(xagg)

    combinations = consecutive_combinations(feature.labels, 3)

    result = _sorted_assocs(evaluator, combinations)

    expected = [
        {
            "combination": [["A"], ["B"], ["C"]],
            "index_to_groupby": {"A": "A", "B": "B", "C": "C"},
            "cramerv": 0.9999999999,
            "tschuprowt": 0.8408964152,
        },
        {
            "combination": [["A"], ["B", "C"]],
            "index_to_groupby": {"A": "A", "B": "B", "C": "B"},
            "cramerv": 0.25,
            "tschuprowt": 0.25,
        },
        {
            "combination": [["A", "B"], ["C"]],
            "index_to_groupby": {"A": "A", "B": "A", "C": "C"},
            "cramerv": 0.0,
            "tschuprowt": 0.0,
        },
    ]
    for res, exp in zip(result, expected):
        assert res["combination"] == exp["combination"]
        assert res["index_to_groupby"] == exp["index_to_groupby"]
        assert res["cramerv"] == exp["cramerv"]
        assert res["tschuprowt"] == exp["tschuprowt"]


def test_compute_associations_with_ten_labels(evaluator: BinaryCombinationEvaluator):
    """Sorted top-1 across 84 combinations."""
    feature = OrdinalFeature("feature", [chr(i) for i in range(65, 75)])  # A to J
    xagg = pd.DataFrame(
        {
            0: [0, 2, 0, 1, 3, 0, 2, 1, 6, 2],
            1: [5, 6, 1, 1, 2, 1, 0, 2, 1, 4],
        },
        index=[chr(i) for i in range(65, 75)],
    )
    evaluator.samples.train = AggregatedSample(xagg)

    combinations = consecutive_combinations(feature.labels, 4)

    result = _sorted_assocs(evaluator, combinations)

    expected = {
        "combination": [["A", "B", "C"], ["D", "E", "F", "G", "H", "I"], ["J"]],
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
            "J": "J",
        },
        "cramerv": 0.4719639494,
        "tschuprowt": 0.3968727932,
    }
    res = result[0]
    assert res["combination"] == expected["combination"]
    assert res["index_to_groupby"] == expected["index_to_groupby"]
    assert res["cramerv"] == expected["cramerv"]
    assert res["tschuprowt"] == expected["tschuprowt"]


def test_compute_associations_evaluators_differ():
    """TschuprowtCombinations and CramervCombinations pick different best combinations
    because TschuprowT penalises more groups via the sqrt(sqrt(k-1)) factor."""
    labels = [chr(i) for i in range(65, 75)]  # A to J
    xagg = pd.DataFrame(
        {
            0: [0, 2, 0, 1, 3, 0, 2, 1, 6, 2],
            1: [5, 6, 1, 1, 2, 1, 0, 2, 1, 4],
        },
        index=labels,
    )

    results = {}
    for cls in [TschuprowtCombinations, CramervCombinations]:
        ev = cls()
        ev.max_n_mod = 4
        ev.min_freq = MIN_FREQ
        ev.samples.train = AggregatedSample(xagg)
        combis = consecutive_combinations(labels, 4)
        grouped = ev._group_xagg_by_combinations(combis)
        assocs = ev._compute_associations(grouped)
        best = sorted(assocs, key=lambda r: r[ev.sort_by] if r[ev.sort_by] == r[ev.sort_by] else -1, reverse=True)[0]
        results[ev.sort_by] = best

    # tschuprowt merges I into D-I (3 groups); cramerv keeps I separate (4 groups)
    assert len(results["tschuprowt"]["combination"]) != len(results["cramerv"]["combination"])
    assert results["tschuprowt"]["combination"] == [["A", "B", "C"], ["D", "E", "F", "G", "H", "I"], ["J"]]
    assert results["cramerv"]["combination"] == [["A", "B", "C"], ["D", "E", "F", "G", "H"], ["I"], ["J"]]


def test_viability_train(evaluator: BinaryCombinationEvaluator):
    """Test the viability of the combination on xagg_train."""
    feature = OrdinalFeature("feature", ["a", "b", "c"])
    xagg = pd.DataFrame({0: [0, 2, 0], 1: [2, 0, 1]}, index=["a", "b", "c"])

    combinations = consecutive_combinations(feature.labels, 2)

    evaluator.samples.train = AggregatedSample(xagg)
    evaluator.min_freq = MIN_FREQ

    associations = _sorted_assocs(evaluator, combinations)
    result = []
    for combination in associations:
        result += [evaluator._test_viability_train(combination)]
    print(result)

    expected = [
        {
            "train": {Keys.VIABLE.value: True},
            "train_rates": pd.DataFrame(
                {"target_mean": [1.0, 0.333333], "frequency": [0.4, 0.6], "count": [2, 3]}, index=["a", "c"]
            ),
        },
        {
            "train": {Keys.VIABLE.value: True},
            "train_rates": pd.DataFrame(
                {"target_mean": [0.5, 1.0], "frequency": [0.8, 0.2], "count": [4, 1]}, index=["a", "c"]
            ),
        },
    ]
    for res, exp in zip(result, expected):
        assert np.allclose(res["train_rates"], exp["train_rates"])
        assert res["train"][Keys.VIABLE.value] == exp["train"][Keys.VIABLE.value]


def test_viability_dev(evaluator: BinaryCombinationEvaluator):
    """Test the viability of the combination on xagg_dev."""
    feature = OrdinalFeature("feature", ["a", "b", "c"])
    xagg = pd.DataFrame({0: [0, 2, 0], 1: [2, 0, 1]}, index=["a", "b", "c"])

    combinations = consecutive_combinations(feature.labels, 2)

    evaluator.samples.train = AggregatedSample(xagg)
    evaluator.min_freq = MIN_FREQ

    associations = _sorted_assocs(evaluator, combinations)

    # test with no xagg_dev
    for combination in associations:
        test_results = evaluator._test_viability_train(combination)
        test_results = evaluator._test_viability_dev(test_results, combination)
        assert test_results.get("dev").get(Keys.VIABLE.value) is None

    # test with xagg_dev but not viable on train
    evaluator.samples.dev = AggregatedSample(pd.DataFrame({0: [0, 2, 0], 1: [2, 0, 1]}, index=["a", "b", "c"]))
    for combination in associations:
        test_results = evaluator._test_viability_dev(
            {"train": {Keys.VIABLE.value: False}, Keys.VIABLE.value: False}, combination
        )
        assert test_results.get("dev").get(Keys.VIABLE.value) is None

    # test with xagg_dev same as train
    result = []
    for combination in associations:
        test_results = evaluator._test_viability_train(combination)
        result += [evaluator._test_viability_dev(test_results, combination)]
    for res in result:
        assert res["train"][Keys.VIABLE.value] is True
        assert res["dev"][Keys.VIABLE.value] is True

    # test with xagg_dev wrong
    evaluator.samples.dev = AggregatedSample(pd.DataFrame({0: [5, 0, 10], 1: [2, 5, 1]}, index=["a", "b", "c"]))
    result = []
    for combination in associations:
        test_results = evaluator._test_viability_train(combination)
        result += [evaluator._test_viability_dev(test_results, combination)]
    print(result)

    expected = [
        {
            "train": {Keys.VIABLE.value: True, Keys.INFO: Messages.PASSED_TESTS},
            "dev": {Keys.VIABLE.value: False, Keys.INFO: Messages.INVERSION_RATES},
        },
        {
            "train": {Keys.VIABLE.value: True, Keys.INFO: Messages.PASSED_TESTS},
            "dev": {Keys.VIABLE.value: False, Keys.INFO: Messages.INVERSION_RATES},
        },
    ]
    for res, exp in zip(result, expected):
        assert res["train"] == exp["train"]
        assert res["dev"] == exp["dev"]


def test_get_viable_combination_without_dev(evaluator: BinaryCombinationEvaluator):
    """Test the get_viable_combination method without a dev xagg."""
    feature = OrdinalFeature("feature", ["a", "b", "c"])
    xagg = pd.DataFrame({0: [0, 2, 0], 1: [2, 0, 1]}, index=["a", "b", "c"])

    combinations = consecutive_combinations(feature.labels, 2)

    evaluator.samples.train = AggregatedSample(xagg)
    evaluator.feature = feature
    evaluator.min_freq = MIN_FREQ
    evaluator.max_n_mod = MAX_N_MOD

    associations = _sorted_assocs(evaluator, combinations)

    # test with no xagg_dev
    result = evaluator._get_viable_combination(associations)
    print(result)
    expected = {
        "xagg": pd.DataFrame({0: [0, 2], 1: [2, 1]}, index=["a", "b"]),
        "combination": [["a"], ["b", "c"]],
        "cramerv": 0.25,
        "tschuprowt": 0.25,
    }
    assert np.allclose(result["xagg"], expected["xagg"])
    assert result["combination"] == expected["combination"]
    assert result["cramerv"] == expected["cramerv"]
    assert result["tschuprowt"] == expected["tschuprowt"]


def test_get_viable_combination_with_non_viable_train(evaluator: BinaryCombinationEvaluator):
    """Test the get_viable_combination method with a non-viable train.

    Scaled to a larger sample so the Wilson CI is tight enough to flag the
    singleton modality as significantly below ``min_freq=0.6``.
    """
    feature = OrdinalFeature("feature", ["a", "b", "c"])
    xagg = pd.DataFrame({0: [0, 200, 0], 1: [200, 0, 1]}, index=["a", "b", "c"])
    evaluator.samples.train = AggregatedSample(xagg)
    evaluator.samples.dev = AggregatedSample(xagg.copy())
    evaluator.feature = feature
    evaluator.max_n_mod = 2

    combinations = consecutive_combinations(feature.labels, 2)
    associations = _sorted_assocs(evaluator, combinations)

    evaluator.min_freq = 0.6
    result = evaluator._get_viable_combination(associations)
    assert result is None


def test_get_viable_combination_with_viable_train(evaluator: BinaryCombinationEvaluator):
    """Test the get_viable_combination method with a viable train."""
    feature = OrdinalFeature("feature", ["a", "b", "c"])
    xagg = pd.DataFrame({0: [0, 2, 0], 1: [2, 0, 1]}, index=["a", "b", "c"])
    evaluator.samples.train = AggregatedSample(xagg)
    evaluator.feature = feature
    evaluator.min_freq = MIN_FREQ
    evaluator.max_n_mod = MAX_N_MOD

    combinations = consecutive_combinations(feature.labels, 2)

    associations = _sorted_assocs(evaluator, combinations)

    # test with xagg_dev same as train
    evaluator.samples.dev = AggregatedSample(xagg)
    result = evaluator._get_viable_combination(associations)
    print(result)

    expected = {
        "xagg": pd.DataFrame({0: [0, 2], 1: [2, 1]}, index=["a", "b"]),
        "combination": [["a"], ["b", "c"]],
        "cramerv": 0.25,
        "tschuprowt": 0.25,
    }
    assert np.allclose(result["xagg"], expected["xagg"])
    assert result["combination"] == expected["combination"]
    assert result["cramerv"] == expected["cramerv"]
    assert result["tschuprowt"] == expected["tschuprowt"]


def test_get_viable_combination_with_not_viable_dev(evaluator: BinaryCombinationEvaluator):
    """Test the get_viable_combination method with a non-viable dev."""
    feature = OrdinalFeature("feature", ["a", "b", "c"])
    xagg = pd.DataFrame({0: [0, 2, 0], 1: [2, 0, 1]}, index=["a", "b", "c"])
    evaluator.samples.train = AggregatedSample(xagg)
    evaluator.feature = feature
    evaluator.min_freq = MIN_FREQ

    combinations = consecutive_combinations(feature.labels, 2)

    associations = _sorted_assocs(evaluator, combinations)

    # test with xagg_dev wrong
    evaluator.samples.dev = AggregatedSample(pd.DataFrame({0: [5, 0, 10], 1: [2, 5, 1]}, index=["a", "b", "c"]))
    result = evaluator._get_viable_combination(associations)
    print(result)
    assert result is None


def test_apply_best_combination_with_viable(evaluator: BinaryCombinationEvaluator):
    """Test the apply_best_combination method with a viable combination."""
    feature = OrdinalFeature("feature", ["a", "b", "c"])
    evaluator.feature = feature
    evaluator.min_freq = MIN_FREQ
    evaluator.max_n_mod = MAX_N_MOD
    xagg = pd.DataFrame({0: [0, 2, 0], 1: [2, 0, 1]}, index=["a", "b", "c"])
    evaluator.samples.train = AggregatedSample(xagg)

    combinations = consecutive_combinations(feature.labels, 2)

    associations = _sorted_assocs(evaluator, combinations)

    # test with xagg_dev same as train
    evaluator.samples.dev = AggregatedSample(xagg)

    best_combination = evaluator._get_viable_combination(associations)

    evaluator._apply_best_combination(best_combination)

    expected = pd.DataFrame({0: [0, 2], 1: [2, 1]}, index=["a", "b to c"])
    assert feature.labels == list(expected.index)
    assert np.allclose(evaluator.samples.train.xagg, expected)
    assert np.allclose(evaluator.samples.dev.xagg, expected)
    assert np.allclose(evaluator.samples.dev.raw, expected)
    assert np.allclose(evaluator.samples.train.raw, expected)


def test_apply_best_combination_with_non_viable(evaluator: BinaryCombinationEvaluator):
    """Test the apply best combination with a non-viable combination."""
    feature = OrdinalFeature("feature", ["a", "b", "c"])
    evaluator.feature = feature
    xagg = pd.DataFrame({0: [0, 2, 0], 1: [2, 0, 1]}, index=["a", "b", "c"])
    evaluator.samples.train = AggregatedSample(xagg)

    # test with xagg_dev same as train
    evaluator.samples.dev = AggregatedSample(xagg)

    best_combination = None

    evaluator._apply_best_combination(best_combination)

    expected = xagg
    assert feature.labels == list(expected.index)
    assert np.allclose(evaluator.samples.train.xagg, expected)
    assert np.allclose(evaluator.samples.dev.xagg, expected)
    assert np.allclose(evaluator.samples.dev.raw, xagg)
    assert np.allclose(evaluator.samples.train.raw, xagg)


def test_best_association_with_combinations_viable(evaluator: BinaryCombinationEvaluator):
    """Test the best association with viable combinations."""
    feature = OrdinalFeature("feature", ["a", "b", "c"])
    evaluator.feature = feature
    evaluator.min_freq = MIN_FREQ
    evaluator.max_n_mod = MAX_N_MOD

    xagg = pd.DataFrame({0: [0, 2, 0], 1: [2, 0, 1]}, index=["a", "b", "c"])
    evaluator.samples.train = AggregatedSample(xagg)
    evaluator.samples.dev = AggregatedSample(xagg)

    combinations = consecutive_combinations(feature.labels, 2)

    result = evaluator._get_best_association(combinations)
    print(result)

    expected = {
        "xagg": pd.DataFrame({0: [0, 2], 1: [2, 1]}, index=["a", "b"]),
        "combination": [["a"], ["b", "c"]],
        "cramerv": 0.25,
        "tschuprowt": 0.25,
    }
    assert np.allclose(result["xagg"], expected["xagg"])
    assert result["combination"] == expected["combination"]
    assert result["cramerv"] == expected["cramerv"]
    assert result["tschuprowt"] == expected["tschuprowt"]

    expected = pd.DataFrame({0: [0, 2], 1: [2, 1]}, index=["a", "b to c"])
    assert feature.labels == list(expected.index)
    assert np.allclose(evaluator.samples.train.xagg, expected)
    assert np.allclose(evaluator.samples.dev.xagg, expected)
    assert np.allclose(evaluator.samples.dev.raw, expected)
    assert np.allclose(evaluator.samples.train.raw, expected)


def test_best_association_with_combinations_non_viable(evaluator: BinaryCombinationEvaluator):
    """Test the best association with no viable combinations."""
    feature = OrdinalFeature("feature", ["a", "b", "c"])
    evaluator.feature = feature
    evaluator.min_freq = MIN_FREQ

    xagg = pd.DataFrame({0: [0, 2, 0], 1: [2, 0, 1]}, index=["a", "b", "c"])
    evaluator.samples.train = AggregatedSample(xagg)

    xagg_dev = pd.DataFrame({0: [5, 0, 10], 1: [2, 5, 1]}, index=["a", "b", "c"])
    evaluator.samples.dev = AggregatedSample(xagg_dev)

    combinations = consecutive_combinations(feature.labels, 2)

    result = evaluator._get_best_association(combinations)
    print(result)

    assert result is None
    assert feature.labels == list(xagg.index)
    assert np.allclose(evaluator.samples.train.xagg, xagg)
    assert np.allclose(evaluator.samples.dev.xagg, xagg_dev)
    assert np.allclose(evaluator.samples.dev.raw, xagg_dev)
    assert np.allclose(evaluator.samples.train.raw, xagg)


def test_best_association_with_nan_combinations_viable(evaluator: BinaryCombinationEvaluator):
    """Test the best association with a feature that has np.nan values."""
    feature = OrdinalFeature("feature", ["a", "b"])
    feature.has_nan = True
    feature.dropna = True
    evaluator.feature = feature
    evaluator.min_freq = MIN_FREQ
    evaluator.max_n_mod = MAX_N_MOD

    xagg = pd.DataFrame({0: [0, 2, 0], 1: [2, 0, 1]}, index=["a", "b", feature.nan])
    evaluator.samples.train = AggregatedSample(xagg)

    evaluator.samples.dev = AggregatedSample(xagg)

    combinations = nan_combinations(feature, 2)

    result = evaluator._get_best_association(combinations)
    print(result)

    expected = {
        "xagg": pd.DataFrame({0: [0, 2], 1: [3, 0]}, index=["a", "b"]),
        "combination": [["a", "__NAN__"], ["b"]],
        "cramerv": 0.5833333333,
        "tschuprowt": 0.5833333333,
    }
    assert np.allclose(result["xagg"], expected["xagg"])
    assert result["combination"] == expected["combination"]
    assert result["cramerv"] == expected["cramerv"]
    assert result["tschuprowt"] == expected["tschuprowt"]
    print(evaluator.samples.train.xagg)

    expected = pd.DataFrame({0: [0, 2], 1: [3, 0]}, index=[f"a, {feature.nan}", "b"])
    assert feature.labels == list(expected.index)
    assert np.allclose(evaluator.samples.train.xagg, expected)
    assert np.allclose(evaluator.samples.dev.xagg, expected)
    assert np.allclose(evaluator.samples.dev.raw, expected)
    assert np.allclose(evaluator.samples.train.raw, expected)


def test_get_best_combination_non_nan_viable(evaluator: BinaryCombinationEvaluator):
    """Test the get_best_combination method with a feature that has no np.nan values."""
    feature = OrdinalFeature("feature", ["a", "b", "c"])
    evaluator.feature = feature

    xagg = pd.DataFrame({0: [0, 2, 0], 1: [2, 0, 1]}, index=["a", "b", "c"])
    evaluator.samples.train = AggregatedSample(xagg)

    evaluator.samples.dev = AggregatedSample(xagg)

    evaluator.min_freq = MIN_FREQ
    evaluator.max_n_mod = 2

    result = evaluator._get_best_combination_non_nan()
    print(result)

    expected = {
        "xagg": pd.DataFrame({0: [0, 2], 1: [2, 1]}, index=["a", "b"]),
        "combination": [["a"], ["b", "c"]],
        "cramerv": 0.25,
        "tschuprowt": 0.25,
    }
    assert np.allclose(result["xagg"], expected["xagg"])
    assert result["combination"] == expected["combination"]
    assert result["cramerv"] == expected["cramerv"]
    assert result["tschuprowt"] == expected["tschuprowt"]

    expected = pd.DataFrame({0: [0, 2], 1: [2, 1]}, index=["a", "b to c"])
    assert feature.labels == list(expected.index)
    assert np.allclose(evaluator.samples.train.xagg, expected)
    assert np.allclose(evaluator.samples.dev.xagg, expected)
    assert np.allclose(evaluator.samples.dev.raw, expected)
    assert np.allclose(evaluator.samples.train.raw, expected)


def test_get_best_combination_non_nan_not_viable(evaluator: BinaryCombinationEvaluator):
    """Test the get_best_combination method with a feature that has no np.nan values."""
    feature = OrdinalFeature("feature", ["a", "b", "c"])
    evaluator.feature = feature

    xagg = pd.DataFrame({0: [0], 1: [2]}, index=["a"])
    evaluator.samples.train = AggregatedSample(xagg)

    evaluator.samples.dev = AggregatedSample(xagg)

    evaluator.max_n_mod = 2

    result = evaluator._get_best_combination_non_nan()
    assert result is None


def test_get_best_combination_non_nan_viable_with_nan(evaluator: BinaryCombinationEvaluator):
    """Test the get_best_combination method with a feature that has no np.nan values."""
    feature = OrdinalFeature("feature", ["a", "b", "c"])
    feature.has_nan = True
    feature.dropna = True
    evaluator.feature = feature

    xagg = pd.DataFrame({0: [0, 2, 0, 0], 1: [2, 0, 1, 3]}, index=["a", "b", "c", feature.nan])
    evaluator.samples.train = AggregatedSample(xagg)

    evaluator.samples.dev = AggregatedSample(xagg)

    evaluator.min_freq = MIN_FREQ
    evaluator.max_n_mod = 2

    result = evaluator._get_best_combination_non_nan()
    print(result)

    expected = {
        "xagg": pd.DataFrame({0: [0, 2], 1: [2, 1]}, index=["a", "b"]),
        "combination": [["a"], ["b", "c"]],
        "cramerv": 0.25,
        "tschuprowt": 0.25,
    }
    assert np.allclose(result["xagg"], expected["xagg"])
    assert result["combination"] == expected["combination"]
    assert result["cramerv"] == expected["cramerv"]
    assert result["tschuprowt"] == expected["tschuprowt"]
    print("trainxagg", evaluator.samples.train.xagg, "\n")
    expected = pd.DataFrame({0: [0, 2, 0], 1: [2, 1, 3]}, index=["a", "b to c", feature.nan])
    assert feature.labels == list(expected.index)
    assert np.allclose(evaluator.samples.train.xagg, expected)
    assert np.allclose(evaluator.samples.dev.xagg, expected)
    assert np.allclose(evaluator.samples.dev.raw, expected)
    assert np.allclose(evaluator.samples.train.raw, expected)


def test_get_best_combination_with_nan_viable(evaluator: BinaryCombinationEvaluator):
    """Test the get_best_combination method with a feature that has no np.nan values."""
    feature = OrdinalFeature("feature", ["a", "b", "c"])
    evaluator.feature = feature

    xagg = pd.DataFrame({0: [0, 2, 0], 1: [2, 0, 1]}, index=["a", "b", "c"])
    evaluator.samples.train = AggregatedSample(xagg)

    evaluator.samples.dev = AggregatedSample(xagg)

    evaluator.min_freq = MIN_FREQ
    evaluator.dropna = False
    evaluator.max_n_mod = 2

    best_combination = evaluator._get_best_combination_non_nan()

    result = evaluator._get_best_combination_with_nan(best_combination)
    print(result)

    expected = {
        "xagg": pd.DataFrame({0: [0, 2], 1: [2, 1]}, index=["a", "b"]),
        "combination": [["a"], ["b", "c"]],
        "cramerv": 0.25,
        "tschuprowt": 0.25,
    }
    assert np.allclose(result["xagg"], expected["xagg"])
    assert result["combination"] == expected["combination"]
    assert result["cramerv"] == expected["cramerv"]
    assert result["tschuprowt"] == expected["tschuprowt"]

    expected = pd.DataFrame({0: [0, 2], 1: [2, 1]}, index=["a", "b to c"])
    assert feature.labels == list(expected.index)
    assert np.allclose(evaluator.samples.train.xagg, expected)
    assert np.allclose(evaluator.samples.dev.xagg, expected)
    assert np.allclose(evaluator.samples.dev.raw, expected)
    assert np.allclose(evaluator.samples.train.raw, expected)


def test_get_best_combination_with_nan_not_viable(evaluator: BinaryCombinationEvaluator):
    """Test the get_best_combination method with a feature that has no np.nan values."""
    feature = OrdinalFeature("feature", ["a", "b", "c"])
    evaluator.feature = feature

    xagg = pd.DataFrame({0: [0], 1: [2]}, index=["a"])
    evaluator.samples.train = AggregatedSample(xagg)

    evaluator.samples.dev = AggregatedSample(xagg)

    evaluator.dropna = False
    evaluator.max_n_mod = 2

    best_combination = evaluator._get_best_combination_non_nan()
    result = evaluator._get_best_combination_with_nan(best_combination)
    assert result is None


def test_get_best_combination_with_nan_viable_with_nan_without_combi(
    evaluator: BinaryCombinationEvaluator,
):
    """Test the get_best_combination method with a feature that has no np.nan values."""
    xagg = pd.DataFrame({0: [0, 2, 0, 0], 1: [2, 0, 1, 3]}, index=["a", "b", "c", "__NAN__"])
    evaluator.max_n_mod = 2
    evaluator.min_freq = MIN_FREQ

    # dropna=False, best_combination=None -> the only safe rescue candidate (all non-nan
    # merged, nan on its own -- nan can never be folded into a partial group, see
    # Features.unfillna) is still tested when the non-nan search found nothing. Unlike
    # before this fix, where dropna=False meant the nan path never ran at all.
    feature = OrdinalFeature("feature", ["a", "b", "c"])
    feature.has_nan = True
    feature.dropna = False
    evaluator.feature = feature
    evaluator.samples.train = AggregatedSample(xagg)
    evaluator.samples.dev = AggregatedSample(xagg)
    evaluator.dropna = False
    result = evaluator._get_best_combination_with_nan(None)
    assert result is not None
    assert result["combination"] == [["a", "b", "c"], [feature.nan]]

    # dropna=True, best_combination=None -> the full nan fan-out runs (the fix):
    # [[a, nan], [b, c]] is a viable rescue combination that was never tested before.
    feature = OrdinalFeature("feature", ["a", "b", "c"])
    feature.has_nan = True
    feature.dropna = True
    evaluator.feature = feature
    evaluator.samples.train = AggregatedSample(xagg)
    evaluator.samples.dev = AggregatedSample(xagg)
    evaluator.dropna = True
    result = evaluator._get_best_combination_with_nan(None)
    assert result is not None
    assert result["combination"] == [["a", feature.nan], ["b", "c"]]

    # has_nan=False -> nan path never runs
    feature.has_nan = False
    result = evaluator._get_best_combination_with_nan(None)
    assert result is None


def test_get_best_combination_with_nan_viable_with_nan_without_feature_nan(
    evaluator: BinaryCombinationEvaluator,
):
    """Test the get_best_combination method with a feature that has no np.nan values."""
    feature = OrdinalFeature("feature", ["a", "b", "c", "d"])
    feature.has_nan = False
    evaluator.feature = feature

    xagg = pd.DataFrame({0: [0, 2, 0, 0], 1: [2, 0, 1, 3]}, index=["a", "b", "c", "d"])
    evaluator.samples.train = AggregatedSample(xagg)

    evaluator.samples.dev = AggregatedSample(xagg)

    evaluator.min_freq = MIN_FREQ
    evaluator.dropna = False
    evaluator.max_n_mod = 2

    best_combination = evaluator._get_best_combination_non_nan()

    result = evaluator._get_best_combination_with_nan(best_combination)
    assert result == best_combination


def test_get_best_combination_with_nan_viable_with_nan_without_dropna(
    evaluator: BinaryCombinationEvaluator,
):
    """No viable combination is found when ``dropna=False`` and the nan row is
    kept — the high ``min_freq`` ensures every 2-group partition fails
    viability under the Wilson CI."""
    feature = OrdinalFeature("feature", ["a", "b", "c"])
    feature.has_nan = False
    evaluator.feature = feature

    xagg = pd.DataFrame({0: [0, 2, 0, 0], 1: [2, 0, 1, 3]}, index=["a", "b", "c", feature.nan])
    evaluator.samples.train = AggregatedSample(xagg)
    evaluator.samples.dev = AggregatedSample(xagg)
    evaluator.min_freq = 0.9
    evaluator.dropna = False
    evaluator.max_n_mod = 2

    best_combination = evaluator._get_best_combination_non_nan()
    result = evaluator._get_best_combination_with_nan(best_combination)
    assert result == best_combination
    assert result is None


def test_get_best_combination_with_nan_viable_with_nan(evaluator: BinaryCombinationEvaluator):
    """Test the get_best_combination method with a feature that has no np.nan values."""
    feature = OrdinalFeature("feature", ["a", "b", "c"])
    feature.has_nan = True
    feature.dropna = True  # has to be set to True
    evaluator.feature = feature
    evaluator.dropna = True

    xagg = pd.DataFrame({0: [0, 2, 0, 0], 1: [2, 0, 1, 3]}, index=["a", "b", "c", feature.nan])
    evaluator.samples.train = AggregatedSample(xagg)

    evaluator.samples.dev = AggregatedSample(xagg)

    evaluator.min_freq = MIN_FREQ
    evaluator.max_n_mod = 2

    best_combination = evaluator._get_best_combination_non_nan()

    result = evaluator._get_best_combination_with_nan(best_combination)
    print(result)
    assert feature.dropna is True

    expected = {
        "xagg": pd.DataFrame({0: [0, 2], 1: [5, 1]}, index=["a", "b to c"]),
        "combination": [["a", "__NAN__"], ["b to c"]],
        "cramerv": 0.4472135955,
        "tschuprowt": 0.4472135955,
    }

    assert np.allclose(result["xagg"], expected["xagg"])
    assert result["combination"] == expected["combination"]
    assert result["cramerv"] == expected["cramerv"]
    assert result["tschuprowt"] == expected["tschuprowt"]

    expected = pd.DataFrame({0: [0, 2], 1: [5, 1]}, index=["a, __NAN__", "b to c"])
    assert feature.labels == list(expected.index)
    assert np.allclose(evaluator.samples.train.xagg, expected)
    assert np.allclose(evaluator.samples.dev.xagg, expected)
    assert np.allclose(evaluator.samples.dev.raw, expected)
    assert np.allclose(evaluator.samples.train.raw, expected)


def test_get_best_combination_viable(evaluator: BinaryCombinationEvaluator):
    """Test the get_best_combination method with a feature that has no np.nan values."""
    feature = OrdinalFeature("feature", ["a", "b", "c"])

    xagg = pd.DataFrame({0: [0, 2, 0], 1: [2, 0, 1]}, index=["a", "b", "c"])

    result = evaluator.get_best_combination(feature, xagg, xagg, max_n_mod=2, min_freq=MIN_FREQ, dropna=False)
    print(result)
    assert evaluator.feature == feature
    assert feature.dropna is evaluator.dropna

    expected = {
        "xagg": pd.DataFrame({0: [0, 2], 1: [2, 1]}, index=["a", "b"]),
        "combination": [["a"], ["b", "c"]],
        "cramerv": 0.25,
        "tschuprowt": 0.25,
    }
    assert np.allclose(result["xagg"], expected["xagg"])
    assert result["combination"] == expected["combination"]
    assert result["cramerv"] == expected["cramerv"]
    assert result["tschuprowt"] == expected["tschuprowt"]

    expected = pd.DataFrame({0: [0, 2], 1: [2, 1]}, index=["a", "b to c"])
    assert feature.labels == list(expected.index)
    assert np.allclose(evaluator.samples.train.xagg, expected)
    assert np.allclose(evaluator.samples.dev.xagg, expected)
    assert np.allclose(evaluator.samples.dev.raw, expected)
    assert np.allclose(evaluator.samples.train.raw, expected)


def test_get_best_combination_viable_without_dev(evaluator: BinaryCombinationEvaluator):
    """Test the get_best_combination method with a feature that has no np.nan values."""
    feature = OrdinalFeature("feature", ["a", "b", "c"])

    xagg = pd.DataFrame({0: [0, 2, 0], 1: [2, 0, 1]}, index=["a", "b", "c"])

    result = evaluator.get_best_combination(feature, xagg, max_n_mod=2, min_freq=MIN_FREQ, dropna=False)
    print(result)
    assert evaluator.feature == feature
    assert feature.dropna is evaluator.dropna

    expected = {
        "xagg": pd.DataFrame({0: [0, 2], 1: [2, 1]}, index=["a", "b"]),
        "combination": [["a"], ["b", "c"]],
        "cramerv": 0.25,
        "tschuprowt": 0.25,
    }
    assert np.allclose(result["xagg"], expected["xagg"])
    assert result["combination"] == expected["combination"]
    assert result["cramerv"] == expected["cramerv"]
    assert result["tschuprowt"] == expected["tschuprowt"]

    expected = pd.DataFrame({0: [0, 2], 1: [2, 1]}, index=["a", "b to c"])
    assert feature.labels == list(expected.index)
    assert np.allclose(evaluator.samples.train.xagg, expected)
    assert not evaluator.samples.dev.has_xagg
    assert np.allclose(evaluator.samples.train.raw, expected)


def test_get_best_combination_not_viable(evaluator: BinaryCombinationEvaluator):
    """Test the get_best_combination method with a feature that has no np.nan values."""
    feature = OrdinalFeature("feature", ["a", "b", "c"])

    xagg = pd.DataFrame({0: [0], 1: [2]}, index=["a"])

    result = evaluator.get_best_combination(feature, xagg, xagg, max_n_mod=2, min_freq=MIN_FREQ, dropna=False)
    assert result is None


# distribution that inverts the train target-rate ordering, so the winning
# combination [["a"], ["b", "c"]] is non-viable on it (reused across CV tests).
_NON_REPRESENTATIVE_XAGG = pd.DataFrame({0: [5, 0, 10], 1: [2, 5, 1]}, index=["a", "b", "c"])


def test_get_best_combination_cv_all_folds_viable(evaluator: BinaryCombinationEvaluator):
    """CV folds that agree with train don't change the carving (ranks from train)."""
    feature = OrdinalFeature("feature", ["a", "b", "c"])
    xagg = pd.DataFrame({0: [0, 2, 0], 1: [2, 0, 1]}, index=["a", "b", "c"])

    result = evaluator.get_best_combination(
        feature, xagg, xagg, max_n_mod=2, min_freq=MIN_FREQ, dropna=False, folds_xagg=[xagg, xagg]
    )

    assert result is not None
    assert result["combination"] == [["a"], ["b", "c"]]
    assert len(evaluator.samples.folds) == 2


def test_get_best_combination_cv_one_fold_vetoes(evaluator: BinaryCombinationEvaluator):
    """A single non-representative fold vetoes a combination that passes train+dev."""
    feature = OrdinalFeature("feature", ["a", "b", "c"])
    xagg = pd.DataFrame({0: [0, 2, 0], 1: [2, 0, 1]}, index=["a", "b", "c"])

    # dev is fine, but one fold inverts the ordering -> all-folds-must-pass drops it
    result = evaluator.get_best_combination(
        feature,
        xagg,
        xagg,
        max_n_mod=2,
        min_freq=MIN_FREQ,
        dropna=False,
        folds_xagg=[xagg, _NON_REPRESENTATIVE_XAGG],
    )
    assert result is None


def test_get_best_combination_cv_folds_only_no_dev(evaluator: BinaryCombinationEvaluator):
    """Folds act as the robustness veto even with no explicit dev sample."""
    feature = OrdinalFeature("feature", ["a", "b", "c"])
    xagg = pd.DataFrame({0: [0, 2, 0], 1: [2, 0, 1]}, index=["a", "b", "c"])

    # folds all agree -> viable
    viable = evaluator.get_best_combination(
        feature, xagg, max_n_mod=2, min_freq=MIN_FREQ, dropna=False, folds_xagg=[xagg, xagg]
    )
    assert viable is not None
    assert viable["combination"] == [["a"], ["b", "c"]]

    # a bad fold (no dev) -> dropped
    fresh = OrdinalFeature("feature", ["a", "b", "c"])
    dropped = type(evaluator)().get_best_combination(
        fresh, xagg, max_n_mod=2, min_freq=MIN_FREQ, dropna=False, folds_xagg=[_NON_REPRESENTATIVE_XAGG]
    )
    assert dropped is None


def test_get_best_combination_viable_with_nan_without_feature_nan(
    evaluator: BinaryCombinationEvaluator,
):
    """Test the get_best_combination method with a feature that has no np.nan values."""
    feature = OrdinalFeature("feature", ["a", "b", "c", "d"])
    feature.has_nan = False

    xagg = pd.DataFrame({0: [0, 2, 0, 0], 1: [2, 0, 1, 3]}, index=["a", "b", "c", "d"])

    result = evaluator.get_best_combination(feature, xagg, max_n_mod=2, min_freq=MIN_FREQ, dropna=False)
    print(result)
    assert evaluator.feature == feature
    assert feature.dropna is evaluator.dropna
    expected = {
        "xagg": pd.DataFrame({0: [2, 0], 1: [2, 4]}, index=["a", "c"]),
        "combination": [["a", "b"], ["c", "d"]],
        "cramerv": 0.2886751346,
        "tschuprowt": 0.2886751346,
    }

    assert np.allclose(result["xagg"], expected["xagg"])
    assert result["combination"] == expected["combination"]
    assert result["cramerv"] == expected["cramerv"]
    assert result["tschuprowt"] == expected["tschuprowt"]


def test_get_best_combination_viable_with_nan_without_dropna(
    evaluator: BinaryCombinationEvaluator,
):
    """Test the get_best_combination method with a feature that has no np.nan values."""
    feature = OrdinalFeature("feature", ["a", "b", "c"])
    feature.has_nan = True

    xagg = pd.DataFrame({0: [0, 2, 0, 0], 1: [2, 0, 1, 3]}, index=["a", "b", "c", feature.nan])

    result = evaluator.get_best_combination(feature, xagg, xagg, max_n_mod=2, min_freq=MIN_FREQ, dropna=False)
    print(result)
    assert evaluator.feature == feature
    assert feature.dropna is evaluator.dropna
    expected = {
        "xagg": pd.DataFrame({0: [0, 2], 1: [2, 1]}, index=["a", "b"]),
        "combination": [["a"], ["b", "c"]],
        "cramerv": 0.25,
        "tschuprowt": 0.25,
    }

    assert np.allclose(result["xagg"], expected["xagg"])
    assert result["combination"] == expected["combination"]
    assert result["cramerv"] == expected["cramerv"]
    assert result["tschuprowt"] == expected["tschuprowt"]

    expected = pd.DataFrame({0: [0, 2, 0], 1: [2, 1, 3]}, index=["a", "b to c", feature.nan])
    assert list(evaluator.samples.train.index) == list(expected.index)
    assert np.allclose(evaluator.samples.train.xagg, expected)
    assert np.allclose(evaluator.samples.dev.xagg, expected)
    assert np.allclose(evaluator.samples.dev.raw, expected)
    assert np.allclose(evaluator.samples.train.raw, expected)


def test_get_best_combination_viable_with_nan(evaluator: BinaryCombinationEvaluator):
    """Test the get_best_combination method with a feature that has np.nan values."""
    feature = OrdinalFeature("feature", ["a", "b", "c"])
    feature.has_nan = True
    assert feature.dropna is False

    xagg = pd.DataFrame({0: [0, 2, 0, 0], 1: [2, 0, 1, 3]}, index=["a", "b", "c", feature.nan])

    result = evaluator.get_best_combination(feature, xagg, xagg, max_n_mod=2, min_freq=MIN_FREQ, dropna=True)
    print(result)
    assert evaluator.feature == feature
    assert feature.dropna is True
    assert feature.dropna is evaluator.dropna

    expected = {
        "xagg": pd.DataFrame({0: [0, 2], 1: [5, 1]}, index=["a", "b to c"]),
        "combination": [["a", "__NAN__"], ["b to c"]],
        "cramerv": 0.4472135955,
        "tschuprowt": 0.4472135955,
    }

    assert np.allclose(result["xagg"], expected["xagg"])
    assert result["combination"] == expected["combination"]
    assert result["cramerv"] == expected["cramerv"]
    assert result["tschuprowt"] == expected["tschuprowt"]

    expected = pd.DataFrame({0: [0, 2], 1: [5, 1]}, index=["a, __NAN__", "b to c"])
    assert feature.labels == list(expected.index)
    assert np.allclose(evaluator.samples.train.xagg, expected)
    assert np.allclose(evaluator.samples.dev.xagg, expected)
    assert np.allclose(evaluator.samples.dev.raw, expected)
    assert np.allclose(evaluator.samples.train.raw, expected)
