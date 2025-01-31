"""Tools to select the best Quantitative and Qualitative features for a Classification task."""

from .filters import BaseFilter, SpearmanFilter, TschuprowtFilter, ValidFilter
from .measures import BaseMeasure, KruskalMeasure, ModeMeasure, NanMeasure, TschuprowtMeasure
from .utils.base_selector import BaseSelector
from ..features import Features
from ..utils import extend_docstring


class ClassificationSelector(BaseSelector):
    """A pipeline of measures to perform a feature pre-selection that maximizes association
    with a qualitative target.

    Get your best features with ``ClassificationSelector.select()``!
    """

    __name__ = "ClassificationSelector"

    @extend_docstring(BaseSelector.__init__)
    def __init__(self, features: Features, n_best_per_type: int, **kwargs) -> None:
        """
        Keyword Arguments
        -----------------
        measures : list[BaseMeasure], optional
            List of association measures to be used, by default ``None``.
            Ranks features based on last provided measure of the list.
            See :ref:`Measures`.
            Implemented measures are:

            * [Quantitative Features] For association evaluation: ``kruskal_measure`` (default), ``R_measure``
            * [Quantitative Features] For outlier detection: ``zscore_measure``, ``iqr_measure``
            * [Qualitative Features] For association evaluation: ``chi2_measure``, ``cramerv_measure``, ``tschuprowt_measure`` (default)

        filters : list[BaseFilter], optional
            List of filters to be used, by default ``None``.
            See :ref:`Filters`.
            Implemented filters are:

            * [Quantitative Features] For linear correlation: ``spearman_filter`` (default), ``pearson_filter``
            * [Qualitative Features] For correlation: ``cramerv_filter``, ``tschuprowt_filter`` (default)
        """
        super().__init__(features, n_best_per_type, **kwargs)

    def _initiate_measures(self, requested_measures: list[BaseMeasure] = None) -> list[BaseMeasure]:
        """initiates the list of measures with default values"""

        # initating to requested ones
        measures = requested_measures

        # if no measure were passed
        if measures is None:
            measures = [NanMeasure(), ModeMeasure(), TschuprowtMeasure(), KruskalMeasure()]

        # adding default measure of mode
        if all(measure.__name__ != ModeMeasure.__name__ for measure in measures):
            measures = [ModeMeasure()] + measures

        # adding default measure of nan
        if all(measure.__name__ != NanMeasure.__name__ for measure in measures):
            measures = [NanMeasure()] + measures

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
        if all(filter.__name__ != ValidFilter.__name__ for filter in filters):
            filters = [ValidFilter()] + filters

        return filters
