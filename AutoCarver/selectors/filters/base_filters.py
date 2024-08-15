""" Filters based on association measure between features and a binary target.
"""

from typing import Any

from pandas import DataFrame

from abc import ABC, abstractmethod
from ...features import BaseFeature


class BaseFilter(ABC):

    __name__ = "BaseFilter"

    is_measure = False
    is_filter = True
    is_x_quantitative = False
    is_x_qualitative = False

    # info
    higher_is_better = False

    def __init__(self, threshold: float = 1.0):
        self.measure = None
        self.threshold = threshold

    @abstractmethod
    def filter(self, X: DataFrame, ranks: list[BaseFeature]) -> list[BaseFeature]:
        pass

    def update_feature(
        self,
        feature: BaseFeature,
        value: float,
        valid: bool,
        info: dict,
    ) -> None:
        """adds measure to specified feature"""

        # existing stats
        filters = feature.statistics.get("filters", {})

        # updating statistics
        filters.update(
            {
                self.__name__: {
                    "value": value,
                    "threshold": self.threshold,
                    "valid": valid,
                    "info": dict(higher_is_better=self.higher_is_better, **info),
                }
            }
        )

        # updating statistics of the feature accordingly
        feature.statistics.update({"filters": filters})


class ValidFilter(BaseFilter):
    __name__ = "Valid"
    is_default = True

    def filter(self, X: DataFrame, ranks: list[BaseFeature]) -> list[BaseFeature]:
        """filters out all non-valid features fril ranks"""
        _ = X  # not used

        # iterating over each feature
        filtered = []
        for feature in ranks:

            # getting feature's measures
            measures = feature.statistics.get("measures")

            # checking for non-valid measures, keeping feature
            if all(measure.get("valid") for measure in measures.values()):
                filtered += [feature]

        return filtered
