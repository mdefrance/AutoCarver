"""Defines a categorical feature"""

from typing import Any

from AutoCarver.config import Constants
from AutoCarver.features.qualitatives.qualitative_feature import QualitativeFeature
from AutoCarver.features.utils.base_feature import BaseFeature


class CategoricalFeature(QualitativeFeature):
    """Defines a categorical feature"""

    __name__ = "Categorical"
    is_categorical = True

    def __init__(
        self,
        name: str,
        *,
        nan: str = Constants.NAN,
        default: str = Constants.DEFAULT,
        ordinal_encoding: bool = False,
        is_fitted: bool = False,
        version: str | None = None,
        version_tag: str | None = None,
        has_nan: bool = False,
        has_default: bool = False,
        dropna: bool = False,
        max_n_chars: int = 50,
    ) -> None:
        super().__init__(
            name,
            nan=nan,
            default=default,
            ordinal_encoding=ordinal_encoding,
            is_fitted=is_fitted,
            version=version,
            version_tag=version_tag,
            has_nan=has_nan,
            has_default=has_default,
            dropna=dropna,
        )
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
    return [feature for feature in features if feature.is_categorical]
