"""Tools to select the best Quantitative and Qualitative features for a Classification task."""

from typing import Callable

from ..discretizers import extend_docstring
from .base_selector import BaseSelector
from .filters.qualitative_filters import tschuprowt_filter
from .filters.quantitative_filters import spearman_filter
from .measures.qualitative_measures import tschuprowt_measure
from .measures.quantitative_measures import kruskal_measure


class ClassificationSelector(BaseSelector):
    """A pipeline of measures to perform a feature pre-selection that maximizes association
    with a qualitative target.

    Get your best features with ``ClassificationSelector.select()``!
    """

    @extend_docstring(BaseSelector.__init__)
    def __init__(
        self,
        n_best: int,
        qualitative_features: list[str] = None,
        quantitative_features: list[str] = None,
        *,
        quantitative_measures: list[Callable] = None,
        qualitative_measures: list[Callable] = None,
        quantitative_filters: list[Callable] = None,
        qualitative_filters: list[Callable] = None,
        colsample: float = 1.0,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        # default measures
        if quantitative_measures is None:
            quantitative_measures = [kruskal_measure]
        if qualitative_measures is None:
            qualitative_measures = [tschuprowt_measure]
        measures = {"float": quantitative_measures, "str": qualitative_measures}

        # default filters
        if quantitative_filters is None:
            quantitative_filters = [spearman_filter]
        if qualitative_filters is None:
            qualitative_filters = [tschuprowt_filter]
        filters = {"float": quantitative_filters, "str": qualitative_filters}

        # initiating BaseSelector with the corresponding list of measures
        super().__init__(
            n_best=n_best,
            quantitative_features=quantitative_features,
            qualitative_features=qualitative_features,
            measures=measures,
            filters=filters,
            colsample=colsample,
            verbose=verbose,
            **kwargs,
        )
