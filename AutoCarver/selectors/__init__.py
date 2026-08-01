"""Loads feature selection tools."""

from AutoCarver.selectors.classification_selector import ClassificationSelector
from AutoCarver.selectors.filters import (
    BaseFilter,
    CramervFilter,
    NonDefaultValidFilter,
    PearsonFilter,
    QualitativeFilter,
    QuantitativeFilter,
    SpearmanFilter,
    TschuprowtFilter,
    ValidFilter,
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
    ReversibleMeasure,
    RMeasure,
    SpearmanMeasure,
    TschuprowtMeasure,
    ZscoreOutlierMeasure,
)
from AutoCarver.selectors.ordinal_selector import OrdinalSelector
from AutoCarver.selectors.regression_selector import RegressionSelector
from AutoCarver.selectors.utils.base_selector import BaseSelector, SelectionConfig

__all__ = [
    # selectors
    "BaseSelector",
    "SelectionConfig",
    "RegressionSelector",
    "ClassificationSelector",
    "OrdinalSelector",
    # filters
    "BaseFilter",
    "QuantitativeFilter",
    "SpearmanFilter",
    "PearsonFilter",
    "QualitativeFilter",
    "CramervFilter",
    "TschuprowtFilter",
    "ValidFilter",
    "NonDefaultValidFilter",
    # measures
    "AbsoluteMeasure",
    "ModeMeasure",
    "NanMeasure",
    "BaseMeasure",
    "OutlierMeasure",
    "ReversibleMeasure",
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
