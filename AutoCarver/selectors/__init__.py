"""Loads feature selection tools."""

from AutoCarver.selectors.classification_selector import ClassificationSelector
from AutoCarver.selectors.filters import (
    BaseFilter,
    CramervFilter,
    PearsonFilter,
    QualitativeFilter,
    QuantitativeFilter,
    SpearmanFilter,
    TschuprowtFilter,
)
from AutoCarver.selectors.measures import (
    AbsoluteMeasure,
    BaseMeasure,
    Chi2Measure,
    CramervMeasure,
    DistanceMeasure,
    IqrOutlierMeasure,
    KruskalEpsilonSquaredMeasure,
    KruskalEtaSquaredMeasure,
    KruskalMeasure,
    ModeMeasure,
    NanMeasure,
    OutlierMeasure,
    PearsonMeasure,
    RMeasure,
    SpearmanMeasure,
    TschuprowtMeasure,
    ZscoreOutlierMeasure,
)
from AutoCarver.selectors.regression_selector import RegressionSelector
from AutoCarver.selectors.utils.base_selector import BaseSelector

__all__ = [
    # selectors
    "BaseSelector",
    "RegressionSelector",
    "ClassificationSelector",
    # filters
    "BaseFilter",
    "QuantitativeFilter",
    "SpearmanFilter",
    "PearsonFilter",
    "QualitativeFilter",
    "CramervFilter",
    "TschuprowtFilter",
    # measures
    "AbsoluteMeasure",
    "ModeMeasure",
    "NanMeasure",
    "BaseMeasure",
    "OutlierMeasure",
    "Chi2Measure",
    "CramervMeasure",
    "PearsonMeasure",
    "SpearmanMeasure",
    "TschuprowtMeasure",
    "RMeasure",
    "DistanceMeasure",
    "IqrOutlierMeasure",
    "KruskalMeasure",
    "KruskalEpsilonSquaredMeasure",
    "KruskalEtaSquaredMeasure",
    "ZscoreOutlierMeasure",
]
