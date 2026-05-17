"""Defines an ordinal feature"""

from AutoCarver.features.qualitatives.qualitative_feature import QualitativeFeature
from AutoCarver.features.utils.base_feature import BaseFeature
from AutoCarver.features.utils.grouped_list import GroupedList
from AutoCarver.utils import extend_docstring


class OrdinalFeature(QualitativeFeature):
    """Defines an ordinal feature"""

    __name__ = "Ordinal"

    is_ordinal = True

    @extend_docstring(BaseFeature.__init__, append=False)
    def __init__(self, name: str, values: list[str]) -> None:
        """
        Parameters
        ----------
        values : list[str]
            Ordered list of all unique values for the feature
        """
        super().__init__(name)

        if values is None or len(values) == 0:
            raise ValueError(f"[{self}] Please make sure to provide values.")

        if self.nan in values:
            raise ValueError(f"[{self}] Ordering for '{self.nan}' can't be set by user, only fitted on data.")

        if not all(isinstance(value, str) for value in values):
            raise ValueError(f"[{self}] Please make sure to provide str values.")

        self.raw_order = values[:]
        super().update(GroupedList(values))

    def _specific_formatting(self, ordered_content: list[str]) -> str:
        """ordinal features' specific label formatting"""

        return f"{ordered_content[0]} to {ordered_content[-1]}"


def get_ordinal_features(features: list[BaseFeature]) -> list[OrdinalFeature]:
    """returns ordinal features amongst provided features"""
    return [feature for feature in features if isinstance(feature, OrdinalFeature)]
