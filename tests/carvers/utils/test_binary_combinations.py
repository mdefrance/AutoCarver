from numpy import nan
from pandas import DataFrame, Series
from pytest import fixture, FixtureRequest
from AutoCarver.features import CategoricalFeature
from numpy import allclose
from AutoCarver.carvers.utils.associations import (
    BinaryCombinationEvaluator,
    TschuprowtCombinations,
    CramervCombinations,
)

from numpy import add, array, searchsorted, sqrt, unique, zeros
from pandas import DataFrame, Series, crosstab, isna
from scipy.stats import chi2_contingency


@fixture(params=[TschuprowtCombinations, CramervCombinations])
def evaluator(request: FixtureRequest) -> str:
    xagg = DataFrame({"A": [1, 3], "B": [4, 6]}, index=["a", "c"])
    return request.param(max_n_mod=5, min_freq=0.2, feature=CategoricalFeature("test"), xagg=xagg)


def test_init(evaluator: BinaryCombinationEvaluator):

    assert evaluator.is_y_binary is True
    assert evaluator.is_y_continuous is False
    assert evaluator.sort_by in ["cramerv", "tschuprowt"]


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


# def test_compute_target_rates_with_nan(evaluator: BinaryCombinationEvaluator):
#     xagg = DataFrame({0: [10, nan, 30], 1: [5, 15, nan]}, index=["a", "b", "c"])
#     result = evaluator._compute_target_rates(xagg)
#     expected = DataFrame(
#         {"target_rate": [0.333333, 1.0, 0.5], "frequency": [0.15, 0.15, 0.35]},
#         index=["a", "b", "c"],
#     )
#     print(result)
#     assert allclose(result, expected)


# def test_compute_target_rates_all_nan(evaluator: BinaryCombinationEvaluator):
#     xagg = DataFrame({0: [nan, nan, nan], 1: [nan, nan, nan]}, index=["a", "b", "c"])
#     result = evaluator._compute_target_rates(xagg)
#     expected = DataFrame(
#         {"target_rate": [nan, nan, nan], "frequency": [0.0, 0.0, 0.0]},
#         index=["a", "b", "c"],
#     )
#     print(result)
#     pd.testing.assert_frame_equal(result, expected)


# def test_compute_target_rates_some_nan(evaluator: BinaryCombinationEvaluator):
#     xagg = DataFrame({0: [10, nan, 30], 1: [nan, 15, 25]}, index=["a", "b", "c"])
#     result = evaluator._compute_target_rates(xagg)
#     expected = DataFrame(
#         {"target_rate": [1.0, 1.0, 0.545455], "frequency": [0.1, 0.15, 0.75]}, index=["a", "b", "c"]
#     )
#     print(result)
#     assert allclose(result, expected)


def test_association_measure_basic(evaluator: BinaryCombinationEvaluator):
    xagg = DataFrame({"A": [10, 20, 30], "B": [5, 15, 25]})
    n_obs = 105
    result = evaluator._association_measure(xagg, n_obs)

    chi2 = chi2_contingency(xagg)[0]
    cramerv = sqrt(chi2 / n_obs)
    tschuprowt = cramerv / sqrt(sqrt(xagg.shape[0] - 1))
    expected = {"cramerv": cramerv, "tschuprowt": tschuprowt}
    assert result == expected


def test_association_measure_with_zeros(evaluator: BinaryCombinationEvaluator):
    xagg = DataFrame({"A": [10, 0, 30], "B": [5, 15, 0]})
    n_obs = 105
    result = evaluator._association_measure(xagg, n_obs)
    chi2 = chi2_contingency(xagg)[0]
    cramerv = sqrt(chi2 / n_obs)
    tschuprowt = cramerv / sqrt(sqrt(xagg.shape[0] - 1))
    expected = {"cramerv": cramerv, "tschuprowt": tschuprowt}

    assert result == expected


def test_association_measure_with_nan(evaluator: BinaryCombinationEvaluator):
    xagg = DataFrame({"A": [10, nan, 30], "B": [5, 15, nan]})
    n_obs = 105
    result = evaluator._association_measure(xagg, n_obs)

    assert isna(result.get("cramerv"))
    assert isna(result.get("tschuprowt"))


def test_association_measure_single_row(evaluator: BinaryCombinationEvaluator):
    xagg = DataFrame({"A": [10], "B": [5]})
    n_obs = 15
    result = evaluator._association_measure(xagg, n_obs)
    assert isna(result.get("tschuprowt"))
    assert result.get("cramerv") == 0


def test_association_measure_single_column(evaluator: BinaryCombinationEvaluator):
    xagg = DataFrame({"A": [10, 20, 30]})
    n_obs = 60
    result = evaluator._association_measure(xagg, n_obs)
    expected = {"cramerv": 0, "tschuprowt": 0}
    assert result == expected


def test_grouper_basic(evaluator: BinaryCombinationEvaluator):
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
    xagg = DataFrame({"A": [1, 2, 3, 4], "B": [4, 5, 6, 7]}, index=["a", "b", "a", "c"])
    groupby = {"a": "group1", "b": "group1", "c": "group2"}
    result = evaluator._grouper(xagg, groupby)
    expected = DataFrame({"A": [6, 4], "B": [15, 7]}, index=["group1", "group2"])
    assert allclose(result, expected)


def test_grouper_no_groupby(evaluator: BinaryCombinationEvaluator):
    xagg = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=["a", "b", "c"])
    groupby = {}
    result = evaluator._grouper(xagg, groupby)
    expected = xagg.copy()
    assert allclose(result, expected)


def test_grouper_partial_groupby(evaluator: BinaryCombinationEvaluator):
    xagg = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=["a", "b", "c"])
    groupby = {"a": "group1"}
    result = evaluator._grouper(xagg, groupby)
    expected = DataFrame({"A": [2, 3, 1], "B": [5, 6, 4]}, index=["b", "c", "group1"])
    print(result)
    assert allclose(result, expected)
