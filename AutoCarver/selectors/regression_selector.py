"""Tools to select the best Quantitative and Qualitative features for a Regression task."""

from .base_selector import BaseSelector
from .measures import BaseMeasure, NanMeasure, ModeMeasure, KruskalMeasure, SpearmanMeasure
from .filters import BaseFilter, TschuprowtFilter, SpearmanFilter, ValidFilter


class RegressionSelector(BaseSelector):
    """A pipeline of measures to perform a feature pre-selection that maximizes association
    with a quantitative target.

    Get your best features with ``RegressionSelector.select()``!
    """

    __name__ = "RegressionSelector"

    def _initiate_measures(self, requested_measures: list[BaseMeasure] = None) -> list[BaseMeasure]:
        """initiates the list of measures with default values"""

        # initating to requested ones
        measures = requested_measures

        # if no measure were passed
        if measures is None:
            measures = [NanMeasure(), ModeMeasure(), SpearmanMeasure(), KruskalMeasure()]

        # adding default measure of mode
        if all(measure.__name__ != "Mode" for measure in measures):
            measures = [ModeMeasure()] + measures

        # adding default measure of nan
        if all(measure.__name__ != "NaN" for measure in measures):
            measures = [NanMeasure()] + measures

        # checking x and y types for measures
        for measure in measures:

            # checking for quantitative target
            if measure.is_y_quantitative:

                # trying to reverse the measure's x and y if implemented
                if not measure.reverse_xy():
                    raise ValueError(
                        f" - [{self.__name__}] Provide measures for quantitative target!"
                    )

        return measures

    def _initiate_filters(self, requested_filters: list[BaseFilter] = None) -> list[BaseFilter]:
        """initiates the list of measures with default values"""

        # initating to requested ones
        filters = requested_filters

        # if no measure were passed
        if filters is None:
            filters = [ValidFilter(), TschuprowtFilter(), SpearmanFilter()]

        # adding default validity filter (based on measures)
        if all(filter.__name__ != "Valid" for filter in filters):
            filters = [ValidFilter()] + filters

        return filters
