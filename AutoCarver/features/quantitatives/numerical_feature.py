""" Defines a continuous/discrete feature"""

from .quantitative_feature import QuantitativeFeature


class NumericalFeature(QuantitativeFeature):
    """Defines a numerical feature"""

    __name__ = "Numerical"
