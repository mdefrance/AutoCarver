""" Loads feature selection tools."""

# from .base_selector import BaseSelector
from .classification_selector import ClassificationSelector
from .filters import (
    BaseFilter,
    CramervFilter,
    PearsonFilter,
    QualitativeFilter,
    QuantitativeFilter,
    SpearmanFilter,
    TschuprowtFilter,
)
from .measures import (
    AbsoluteMeasure,
    BaseMeasure,
    Chi2Measure,
    CramervMeasure,
    DistanceMeasure,
    IqrOutlierMeasure,
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
from .regression_selector import RegressionSelector

__all__ = [
    # selectors
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
    "ZscoreOutlierMeasure",
]
