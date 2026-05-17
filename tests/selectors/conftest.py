"""Defines fixtures for carvers pytests"""

import pandas as pd

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
from pytest import FixtureRequest, fixture

from AutoCarver.features import BaseFeature, Features
from AutoCarver.selectors import BaseFilter, BaseMeasure
from AutoCarver.selectors.filters import (
    PearsonFilter,
    TschuprowtFilter,
    ValidFilter,
)
from AutoCarver.selectors.measures import (
    CramervMeasure,
    KruskalMeasure,
    NanMeasure,
    SpearmanMeasure,
    TschuprowtMeasure,
    ZscoreOutlierMeasure,
)
from AutoCarver.selectors.utils.base_selector import BaseSelector

# setting BaseSelector, BaseFeature as non abstract classes for the duration of the test
BaseSelector.__abstractmethods__ = set()
BaseFeature.__abstractmethods__ = set()

quanti_measures = [KruskalMeasure, SpearmanMeasure]
quali_measures = [CramervMeasure, TschuprowtMeasure]


@fixture(params=quanti_measures + quali_measures)
def measure(request: FixtureRequest) -> BaseMeasure:
    return request.param()


@fixture(params=quali_measures)
def quali_measure(request: FixtureRequest) -> BaseMeasure:
    return request.param()


@fixture(params=quanti_measures)
def quanti_measure(request: FixtureRequest) -> BaseMeasure:
    return request.param()


@fixture(params=[NanMeasure, ZscoreOutlierMeasure])
def default_measure(request: FixtureRequest) -> BaseMeasure:
    # Two params needed: NanMeasure (regular default) vs ZscoreOutlierMeasure
    # (outlier-only default that triggers the qualitative conditional branch)
    return request.param()


@fixture
def measures(default_measure: BaseMeasure) -> list[BaseMeasure]:
    # Always includes 2 quanti + 2 quali so tie-breaking between same-type measures is tested.
    # Parametrised only over default_measure (2 variants); individual measure implementations
    # are covered by test_quantitative_measures.py / test_qualitative_measures.py.
    return [default_measure, KruskalMeasure(), SpearmanMeasure(), CramervMeasure(), TschuprowtMeasure()]


@fixture(params=[PearsonFilter])
def quanti_filter(request: FixtureRequest) -> BaseFilter:
    return request.param()


@fixture(params=[TschuprowtFilter])
def quali_filter(request: FixtureRequest) -> BaseFilter:
    return request.param()


@fixture(params=[ValidFilter])
def default_filter(request: FixtureRequest) -> BaseFilter:
    return request.param()


@fixture
def filters(quanti_filter: BaseFilter, quali_filter: BaseFilter, default_filter: BaseFilter) -> list[BaseFilter]:
    return [default_filter, quanti_filter, quali_filter]


@fixture
def features() -> list[BaseFeature]:
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
    return pd.DataFrame(
        {
            "feature1": [0, 1, 0, 1, 0, 1],
            "feature2": [2, 0, 2, 0, 0, 0],
            "feature3": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "feature4": [-4.0, -2.0, -0.0, -2.0, -4.0, -8.0],
        }
    )


@fixture
def y():
    return pd.Series([7, 8, 9, 10, 11, 12])


@fixture
def features_object(features: list[BaseFeature]) -> Features:
    """mock Features"""
    return Features.from_list(features)
