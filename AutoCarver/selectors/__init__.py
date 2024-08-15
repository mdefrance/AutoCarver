""" Loads feature selection tools."""

# from .base_selector import BaseSelector
# from .classification_selector import ClassificationSelector
from .filters import (
    #     cramerv_filter,
    #     thresh_filter,
    #     tschuprowt_filter,
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
    dtype_measure,
    make_measure,
    mode_measure,
    nans_measure,
    RMeasure,
    DistanceMeasure,
    KruskalMeasure,
    PearsonMeasure,
    SpearmanMeasure,
    IqrOutlierMeasure,
    ZscoreOutlierMeasure,
)

# from .regression_selector import RegressionSelector

__all__ = [
    # "BaseSelector",
    # "ClassificationSelector",
    # "thresh_filter",
    # "cramerv_filter",
    # "tschuprowt_filter",
    # "pearson_filter",
    # "spearman_filter",
    # filters
    "BaseFilter",
    "QuantitativeFilter",
    "SpearmanFilter",
    "PearsonFilter",
    # measures
    "AbsoluteMeasure",
    "dtype_measure",
    "make_measure",
    "mode_measure",
    "nans_measure",
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
    # "RegressionSelector",
]
