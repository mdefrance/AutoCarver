""" Defines an ordinal feature"""

from ...utils import extend_docstring
from ..utils.base_feature import BaseFeature
from ..utils.grouped_list import GroupedList
from .qualitative_feature import QualitativeFeature


class OrdinalFeature(QualitativeFeature):
    """Defines an ordinal feature"""

    __name__ = "Ordinal"

    is_ordinal = True

    @extend_docstring(BaseFeature.__init__, append=False)
    def __init__(self, name: str, values: list[str], **kwargs) -> None:
        """
        Parameters
        ----------
        values : list[str]
            Ordered list of all unique values for the feature
        """
        super().__init__(name, **kwargs)

        # checking for values
        if values is None or len(values) == 0:
            raise ValueError(f"[{self}] Please make sure to provide values.")

        # checking for nan ordering
        if self.nan in values and not kwargs.get("load_mode", False):
            raise ValueError(
                f"[{self}] Ordering for '{self.nan}' can't be set by user, only fitted on data."
            )

        # checking for str values
        if not all(isinstance(value, str) for value in values):
            raise ValueError(f"[{self}] Please make sure to provide str values.")

        # saving up raw ordering for labeling
        self.raw_order = kwargs.get("raw_order", values[:])

        # setting values and labels
        super().update(GroupedList(values))

    def _specific_formatting(self, ordered_content: list[str]) -> str:
        """ordinal features' specific label formatting"""

        # ordered label for ordinal features
        return f"{ordered_content[0]} to {ordered_content[-1]}"


def get_ordinal_features(features: list[BaseFeature]) -> list[OrdinalFeature]:
    """returns ordinal features amongst provided features"""
    return [feature for feature in features if feature.is_ordinal]
