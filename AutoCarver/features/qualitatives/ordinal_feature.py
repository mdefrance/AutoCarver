"""Defines an ordinal feature"""

from AutoCarver.config import Constants
from AutoCarver.features.qualitatives.qualitative_feature import QualitativeFeature
from AutoCarver.features.utils.base_feature import BaseFeature
from AutoCarver.features.utils.grouped_list import GroupedList
from AutoCarver.utils import extend_docstring


class OrdinalFeature(QualitativeFeature):
    """Defines an ordinal feature"""

    __name__ = "Ordinal"

    is_ordinal = True

    @extend_docstring(BaseFeature.__init__, append=False)
    def __init__(
        self,
        name: str,
        values: list[str],
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
    ) -> None:
        """
        Parameters
        ----------
        values : list[str]
            Ordered list of all unique values for the feature
        """
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
    return [feature for feature in features if feature.is_ordinal]
