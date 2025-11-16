""" Defines a categorical feature"""

from ..utils.base_feature import BaseFeature
from .qualitative_feature import QualitativeFeature


class CategoricalFeature(QualitativeFeature):
    """Defines a categorical feature"""

    __name__ = "Categorical"
    is_categorical = True

    def _specific_formatting(self, ordered_content: list[str]) -> str:
        """categorical features' specific label formatting"""

        # list label for categorical feature
        label = ", ".join(ordered_content)
        if len(label) > self.max_n_chars:
            label = label[: self.max_n_chars] + "..."

        return label


def get_categorical_features(features: list[BaseFeature]) -> list[CategoricalFeature]:
    """returns categorical features amongst provided features"""
    return [feature for feature in features if feature.is_categorical]
