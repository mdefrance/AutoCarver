""" Loads Quantitative Discretization tools."""

from .continuous_discretizer import ContinuousDiscretizer
from .quantitative_discretizer import QuantitativeDiscretizer

__all__ = [
    "QuantitativeDiscretizer",
    "ContinuousDiscretizer",
]
