"""Tools to select the best Quantitative and Qualitative features for a Classification task."""

from .filters import (
    BaseFilter,
    NonDefaultValidFilter,
    SpearmanFilter,
    TschuprowtFilter,
    ValidFilter,
)
from .measures import BaseMeasure, KruskalMeasure, ModeMeasure, NanMeasure, TschuprowtMeasure
from .utils.base_selector import BaseSelector


class ClassificationSelector(BaseSelector):
    """A pipeline of measures to perform a feature pre-selection that maximizes association
    with a qualitative target.

    """

    __name__ = "ClassificationSelector"

    def _initiate_measures(self, requested_measures: list[BaseMeasure] = None) -> list[BaseMeasure]:
        """initiates the list of measures with default values"""

        # initating to requested ones
        measures = requested_measures

        # if no measure were passed
        if measures is None:
            measures = [NanMeasure(), ModeMeasure(), TschuprowtMeasure(), KruskalMeasure()]

        # adding default measure of mode
        mode_measure = ModeMeasure()
        if all(measure.__name__ != mode_measure.__name__ for measure in measures):
            measures = [mode_measure] + measures

        # adding default measure of nan
        nan_measure = NanMeasure()
        if all(measure.__name__ != nan_measure.__name__ for measure in measures):
            measures = [nan_measure] + measures

        # checking for measures for quantitative target
        if not all(
            measure.is_y_qualitative
            or (measure.reverse_xy() and measure.is_y_qualitative)
            or measure.is_default
            for measure in measures
        ):
            raise ValueError(f"[{self}] Provide measures for qualitative target!")
        return measures

    def _initiate_filters(self, requested_filters: list[BaseFilter] = None) -> list[BaseFilter]:
        """initiates the list of measures with default values"""

        # initating to requested ones
        filters = requested_filters

        # if no measure were passed
        if filters is None:
            filters = [ValidFilter(), TschuprowtFilter(), SpearmanFilter()]

        # adding default validity filter (based on measures)
        valid_filter = ValidFilter()
        if all(filter.__name__ != valid_filter.__name__ for filter in filters):
            filters = [valid_filter] + filters

        # adding default validity filter (based on measures)
        valid_filter = NonDefaultValidFilter()
        if all(filter.__name__ != valid_filter.__name__ for filter in filters):
            filters = [valid_filter] + filters

        return filters
