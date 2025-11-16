""" Filters based on association measure between features and a binary target.
"""

from abc import ABC, abstractmethod

from pandas import DataFrame

from ...features import BaseFeature


class BaseFilter(ABC):
    """Base filter class."""

    __name__ = "BaseFilter"

    is_x_quantitative = False
    """ wether x is quantitative or not """

    is_x_qualitative = False
    """ wether x is qualitative or not """

    # info
    is_default = False
    """ wether the filter is an association measure or not """

    higher_is_better = False
    """ wether higher values are better or not """

    is_absolute = False
    """ wether the measure needs absolute value for comparison or not """

    def __init__(self, threshold: float = 1.0) -> None:
        """
        Parameters
        ----------
        threshold : float, optional
            Maximum threshold to reach, by default ``1.0``
        """

        self.measure = None
        self.threshold = threshold

    def __repr__(self) -> str:
        return self.__name__

    @abstractmethod
    def filter(self, X: DataFrame, ranks: list[BaseFeature]) -> list[BaseFeature]:
        """Filters out ranked features that reach the association threshold

        Parameters
        ----------
        X : DataFrame
            DataFrame containing features
        ranks : list[BaseFeature]
            List of ranked features to filter, from most to least associated

        Returns
        -------
        list[BaseFeature]
            Filtered list of features
        """

    def _update_feature(
        self,
        feature: BaseFeature,
        value: float,
        valid: bool,
        info: dict,
    ) -> None:
        """adds measure to specified feature"""

        # filter state info
        filter_state = {
            self.__name__: {
                "value": value,
                "threshold": self.threshold,
                "valid": valid,
                "info": {"higher_is_better": self.higher_is_better, **info},
            }
        }

        # updating statistics of the feature accordingly
        feature.filters.update(filter_state)


class ValidFilter(BaseFilter):
    """Valid filter class."""

    __name__ = "Valid"

    is_default = True
    is_x_quantitative = True
    is_x_qualitative = True

    def filter(self, X: DataFrame, ranks: list[BaseFeature]) -> list[BaseFeature]:
        """filters out all non-valid features fril ranks"""
        _ = X  # not used

        # iterating over each feature
        filtered = []
        for feature in ranks:
            # checking for non-valid measures, keeping feature
            if len(feature.measures) == 0 or all(
                measure.get("valid") for measure in feature.measures.values()
            ):
                filtered += [feature]

        return filtered


class NonDefaultValidFilter(ValidFilter):
    """Valid filter class for non-default metrics."""

    __name__ = "NonDefaultValid"

    is_default = False
