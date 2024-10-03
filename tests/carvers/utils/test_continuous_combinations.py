from numpy import nan
from pandas import DataFrame, Series
from pytest import fixture, FixtureRequest
from AutoCarver.features import CategoricalFeature
from numpy import allclose
from AutoCarver.carvers.utils.associations import (
    ContiunousCombinationEvaluator,
    KruksalCombinations,
)

from numpy import mean, unique
from pandas import DataFrame, Series
from scipy.stats import kruskal


@fixture(params=[KruksalCombinations])
def evaluator(request: FixtureRequest) -> ContiunousCombinationEvaluator:
    xagg = DataFrame({"A": [1, 3], "B": [4, 6]}, index=["a", "c"])
    return request.param(max_n_mod=5, min_freq=0.2, feature=CategoricalFeature("test"), xagg=xagg)


# def test_association_measure_basic(evaluator: ContiunousCombinationEvaluator):
#     xagg = Series([10, 20, 30, 40, 50])
#     n_obs = 100

#     result = evaluator._association_measure(xagg, n_obs)
#     expected = {"kruskal": kruskal(*tuple(xagg.values))[0]}
#     assert result == expected


# def test_association_measure_with_nan(evaluator: ContiunousCombinationEvaluator):
#     xagg = Series([10, np.nan, 30, 40, 50])
#     n_obs = 100

#     result = evaluator._association_measure(xagg.dropna(), n_obs)
#     expected = {"kruskal": kruskal(*tuple(xagg.dropna().values))[0]}
#     assert result == expected


# def test_association_measure_all_nan(evaluator: ContiunousCombinationEvaluator):
#     xagg = Series([np.nan, np.nan, np.nan])
#     n_obs = 100

#     with pytest.raises(ValueError):
#         evaluator._association_measure(xagg.dropna(), n_obs)


# def test_association_measure_single_value(evaluator: ContiunousCombinationEvaluator):
#     xagg = Series([10])
#     n_obs = 1

#     with pytest.raises(ValueError):
#         evaluator._association_measure(xagg, n_obs)


# def test_association_measure_identical_values(evaluator: ContiunousCombinationEvaluator):
#     xagg = Series([10, 10, 10, 10, 10])
#     n_obs = 5

#     result = evaluator._association_measure(xagg, n_obs)
#     expected = {"kruskal": kruskal(*tuple(xagg.values))[0]}
#     assert result == expected
