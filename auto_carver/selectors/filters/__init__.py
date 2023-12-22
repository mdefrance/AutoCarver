""" Loads association filters."""

from .base_filters import thresh_filter
from .qualitative_filters import cramerv_filter, tschuprowt_filter
from .quantitative_filters import pearson_filter, spearman_filter

__all__ = [
    "thresh_filter",
    "cramerv_filter",
    "tschuprowt_filter",
    "pearson_filter",
    "spearman_filter",
]
