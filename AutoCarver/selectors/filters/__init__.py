""" Loads association filters."""

from .base_filters import BaseFilter, ValidFilter

from .qualitative_filters import QualitativeFilter, CramervFilter, TschuprowtFilter
from .quantitative_filters import QuantitativeFilter, SpearmanFilter, PearsonFilter

__all__ = [
    "BaseFilter",
    "QuantitativeFilter",
    "ValidFilter",
    "SpearmanFilter",
    "PearsonFilter",
    "QualitativeFilter",
    "CramervFilter",
    "TschuprowtFilter",
]
