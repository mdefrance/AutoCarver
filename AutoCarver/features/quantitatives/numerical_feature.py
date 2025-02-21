""" Defines a continuous/discrete feature"""

from ..utils.base_feature import BaseFeature
from .quantitative_feature import QuantitativeFeature


class NumericalFeature(QuantitativeFeature):
    """Defines a numerical feature"""

    __name__ = "Numerical"

    is_numerical = True


def get_numerical_features(features: list[BaseFeature]) -> list[NumericalFeature]:
    """returns numerical features amongst provided features"""
    return [feature for feature in features if feature.is_numerical]
