""" Loads feature selection tools."""

# from .base_selector import BaseSelector
from .classification_selector import ClassificationSelector
from .filters import (
    QualitativeFilter,
    CramervFilter,
    TschuprowtFilter,
    BaseFilter,
    QuantitativeFilter,
    SpearmanFilter,
    PearsonFilter,
)

from .measures import (
    AbsoluteMeasure,
    BaseMeasure,
    OutlierMeasure,
    Chi2Measure,
    CramervMeasure,
    TschuprowtMeasure,
    ModeMeasure,
    NanMeasure,
    RMeasure,
    DistanceMeasure,
    KruskalMeasure,
    PearsonMeasure,
    SpearmanMeasure,
    IqrOutlierMeasure,
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
