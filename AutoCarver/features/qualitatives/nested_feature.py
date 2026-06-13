"""Defines a nested feature.

A nested feature is built from several columns of increasing granularity (e.g.
``col_a`` ⊃ ``col_b`` ⊃ ``col_c``), where each fine modality is included in exactly
one coarser modality. After fitting (see :class:`NestedDiscretizer`) it collapses to a
single output column — the finest column (``col_c``) — in which rare modalities have been
rolled up to their data-derived parent until frequent enough. From then on it behaves like
an ordinary single-column qualitative feature.
"""

from typing import Any

from AutoCarver.features.qualitatives.qualitative_feature import QualitativeFeature
from AutoCarver.features.utils.base_feature import BaseFeature
from AutoCarver.features.utils.grouped_list import GroupedList


class NestedFeature(QualitativeFeature):
    """Defines a nested feature built from nested columns finest-to-coarsest."""

    __name__ = "Nested"
    is_nested = True

    def __init__(self, name: str, parents: list[str], *, max_n_chars: int = 50) -> None:
        """
        Parameters
        ----------
        name : str
            Finest column name — also the single output column after fitting.
        parents : list[str]
            Coarser-ward parent columns (from nearest to farthest). Each value of a
            level is included in exactly one value of the next level.
        max_n_chars : int, optional
            Maximum number of characters per label, by default ``50``.
        """
        super().__init__(name)

        if not parents:
            raise ValueError(f"[{self}] Please provide at least one parent column.")
        if not all(isinstance(parent, str) for parent in parents):
            raise ValueError(f"[{self}] Please make sure to provide str parent column names.")
        if name in parents:
            raise ValueError(f"[{self}] Output column {name!r} can't also be a parent column.")

        self.parents = list(parents)
        # max number of characters per label
        self.max_n_chars: int = max_n_chars

    @property
    def levels(self) -> list[str]:
        """Ordered level columns finest-to-coarsest (output column first)."""
        return [self.name] + self.parents

    def to_json(self, light_mode: bool = False) -> dict[str, Any]:
        feature = super().to_json(light_mode=light_mode)
        feature["parents"] = self.parents
        feature["max_n_chars"] = self.max_n_chars
        return feature

    def _restore_from_json(self, feature_json: dict) -> None:
        # parents / max_n_chars must be restored before super() since super() may trigger
        # update_labels(), which reads self.max_n_chars via _specific_formatting
        self.parents = list(feature_json.get("parents") or [])
        self.max_n_chars = feature_json.get("max_n_chars", 50)
        super()._restore_from_json(feature_json)

    def make_labels(self) -> GroupedList:
        """Builds labels: each bucket is labelled by its leader value — a kept finest modality
        or the coarser parent it rolled up into — which is the meaningful name for a nested
        feature (rather than a join of the rolled-up children)."""
        return GroupedList(list(self.content))

    def _specific_formatting(self, ordered_content: list[str]) -> str:
        """nested features' specific label formatting (matches categorical)

        Only used as a fallback; :meth:`make_labels` labels buckets by their leader value.
        """
        label = ", ".join(ordered_content)
        if len(label) > self.max_n_chars:
            label = label[: self.max_n_chars] + "..."

        return label


def get_nested_features(features: list[BaseFeature]) -> list[NestedFeature]:
    """returns nested features amongst provided features"""
    return [feature for feature in features if isinstance(feature, NestedFeature)]
