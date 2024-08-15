""" Loads association filters."""

from .base_filters import thresh_filter, BaseFilter

# from .qualitative_filters import cramerv_filter, tschuprowt_filter
from .quantitative_filters import QuantitativeFilter, SpearmanFilter, PearsonFilter

__all__ = [
    "BaseFilter",
    "QuantitativeFilter",
    "thresh_filter",
    "SpearmanFilter",
    "PearsonFilter",
    # "cramerv_filter",
    # "tschuprowt_filter",
    # "pearson_filter",
    # "spearman_filter",
]
