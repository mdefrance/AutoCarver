"""Defines a categorical feature"""

from typing import Any

from AutoCarver.features.qualitatives.qualitative_feature import QualitativeFeature
from AutoCarver.features.utils.base_feature import BaseFeature


class CategoricalFeature(QualitativeFeature):
    """Defines a categorical feature"""

    __name__ = "Categorical"
    is_categorical = True

    def __init__(self, name: str, *, max_n_chars: int = 50) -> None:
        super().__init__(name)
        # max number of characters per label
        self.max_n_chars: int = max_n_chars

    def to_json(self, light_mode: bool = False) -> dict[str, Any]:
        feature = super().to_json(light_mode=light_mode)
        feature["max_n_chars"] = self.max_n_chars
        return feature

    def _restore_from_json(self, feature_json: dict) -> None:
        # max_n_chars must be restored before super() since super() may trigger
        # update_labels(), which reads self.max_n_chars via _specific_formatting
        self.max_n_chars = feature_json.get("max_n_chars", 50)
        super()._restore_from_json(feature_json)

    def _specific_formatting(self, ordered_content: list[str]) -> str:
        """categorical features' specific label formatting"""

        label = ", ".join(ordered_content)
        if len(label) > self.max_n_chars:
            label = label[: self.max_n_chars] + "..."

        return label


def get_categorical_features(features: list[BaseFeature]) -> list[CategoricalFeature]:
    """returns categorical features amongst provided features"""
    return [feature for feature in features if isinstance(feature, CategoricalFeature)]
