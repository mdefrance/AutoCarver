"""Loads association measures."""

from AutoCarver.selectors.measures.base_measures import (
    AbsoluteMeasure,
    BaseMeasure,
    ModeMeasure,
    NanMeasure,
    OutlierMeasure,
)
from AutoCarver.selectors.measures.qualitative_measures import (
    Chi2Measure,
    CramervMeasure,
    TschuprowtMeasure,
)
from AutoCarver.selectors.measures.quantitative_measures import (
    DistanceMeasure,
    IqrOutlierMeasure,
    KruskalEffectSizeMeasure,
    KruskalEtaSquaredMeasure,
    KruskalMeasure,
    PearsonMeasure,
    ReversibleMeasure,
    RMeasure,
    SpearmanMeasure,
    ZscoreOutlierMeasure,
)

__all__ = [
    "AbsoluteMeasure",
    "NanMeasure",
    "ModeMeasure",
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
    "KruskalEffectSizeMeasure",
    "KruskalEtaSquaredMeasure",
    "ZscoreOutlierMeasure",
    "ReversibleMeasure",
]
