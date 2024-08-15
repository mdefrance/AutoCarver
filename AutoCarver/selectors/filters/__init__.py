""" Loads association filters."""

from .base_filters import thresh_filter, BaseFilter

from .qualitative_filters import QualitativeFilter, CramervFilter, TschuprowtFilter
from .quantitative_filters import QuantitativeFilter, SpearmanFilter, PearsonFilter

__all__ = [
    "BaseFilter",
    "QuantitativeFilter",
    "thresh_filter",
    "SpearmanFilter",
    "PearsonFilter",
    "QualitativeFilter",
    "CramervFilter",
    "TschuprowtFilter",
]
