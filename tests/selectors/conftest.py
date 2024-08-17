""" Defines fixtures for carvers pytests"""

# from pytest import FixtureRequest, fixture


# @fixture
# def quantitative_features() -> list[str]:
#     """List of quantitative raw features to be carved"""
#     return [
#         "Quantitative",
#         "Discrete_Quantitative_highnan",
#         "Discrete_Quantitative_lownan",
#         "Discrete_Quantitative",
#         "Discrete_Quantitative_rarevalue",
#     ]


# @fixture
# def qualitative_features() -> list[str]:
#     """List of qualitative raw features to be carved"""
#     return [
#         "Qualitative",
#         "Qualitative_grouped",
#         "Qualitative_lownan",
#         "Qualitative_highnan",
#         "Discrete_Qualitative_noorder",
#         "Discrete_Qualitative_lownan_noorder",
#         "Discrete_Qualitative_rarevalue_noorder",
#     ]


# @fixture
# def ordinal_features() -> list[str]:
#     """List of ordinal raw features to be carved"""
#     return [
#         "Qualitative_Ordinal",
#         "Qualitative_Ordinal_lownan",
#         "Discrete_Qualitative_highnan",
#     ]


# @fixture(params=[3, 5])
# def n_best(request: FixtureRequest) -> int:
#     """Number of features to be selected"""
#     return request.param


from pytest import fixture, FixtureRequest

from pandas import DataFrame, Series
from AutoCarver.features import Features, BaseFeature
from AutoCarver.selectors import BaseFilter, BaseMeasure
from AutoCarver.selectors.measures import (
    NanMeasure,
    ModeMeasure,
    Chi2Measure,
    CramervMeasure,
    KruskalMeasure,
    PearsonMeasure,
    DistanceMeasure,
    SpearmanMeasure,
    IqrOutlierMeasure,
    TschuprowtMeasure,
    ZscoreOutlierMeasure,
)
from AutoCarver.selectors.filters import (
    ValidFilter,
    CramervFilter,
    PearsonFilter,
    SpearmanFilter,
    TschuprowtFilter,
)
from AutoCarver.selectors.base_selector import BaseSelector

# setting BaseSelector as non abstract classes for the duration of the test
BaseSelector.__abstractmethods__ = set()

quanti_measures = [KruskalMeasure, PearsonMeasure, DistanceMeasure, SpearmanMeasure]
quali_measures = [Chi2Measure, CramervMeasure, TschuprowtMeasure]


@fixture(params=quanti_measures + quali_measures)
def measure(request: FixtureRequest) -> BaseMeasure:
    return request.param()


@fixture(params=quali_measures)
def quali_measure(request: FixtureRequest) -> BaseMeasure:
    return request.param()


@fixture(params=quanti_measures)
def quanti_measure(request: FixtureRequest) -> BaseMeasure:
    return request.param()


@fixture(params=[NanMeasure, ModeMeasure, ZscoreOutlierMeasure, IqrOutlierMeasure])
def default_measure(request: FixtureRequest) -> BaseMeasure:
    return request.param()


@fixture
def measures(
    quanti_measure: BaseMeasure, quali_measure: BaseMeasure, default_measure: BaseMeasure
) -> list[BaseMeasure]:
    return [default_measure, quanti_measure, quali_measure]


@fixture(params=[PearsonFilter, SpearmanFilter])
def quanti_filter(request: FixtureRequest) -> BaseFilter:
    return request.param()


@fixture(params=[CramervFilter, TschuprowtFilter])
def quali_filter(request: FixtureRequest) -> BaseFilter:
    return request.param()


@fixture(params=[ValidFilter])
def default_filter(request: FixtureRequest) -> BaseFilter:
    return request.param()


@fixture
def filters(
    quanti_filter: BaseFilter, quali_filter: BaseFilter, default_filter: BaseFilter
) -> list[BaseFilter]:
    return [default_filter, quanti_filter, quali_filter]


@fixture
def features():
    feature1 = BaseFeature("feature1")
    feature1.is_qualitative = True
    feature1.is_categorical = True
    feature2 = BaseFeature("feature2")
    feature2.is_qualitative = True
    feature2.is_ordinal = True
    feature3 = BaseFeature("feature3")
    feature3.is_quantitative = True
    feature4 = BaseFeature("feature4")
    feature4.is_quantitative = True
    return [feature1, feature2, feature3, feature4]


@fixture
def X():
    return DataFrame(
        {
            "feature1": [0, 1, 0, 1, 0, 1],
            "feature2": [2, 0, 2, 0, 0, 0],
            "feature3": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "feature4": [-4.0, -2.0, -0.0, -2.0, -4.0, -8.0],
        }
    )


@fixture
def y():
    return Series([7, 8, 9, 10, 11, 12])


@fixture
def features_object(features: list[BaseFeature]) -> Features:
    """mock Features"""
    return Features(features)
