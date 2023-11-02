"""(For compatibility) Tools to select the best Quantitative and Qualitative features."""

from typing import Callable
from warnings import warn

from ..discretizers import GroupedList, extend_docstring
from ..selectors import ClassificationSelector


class FeatureSelector(ClassificationSelector):
    @extend_docstring(ClassificationSelector.__init__)
    def __init__(
        self,
        n_best: int,
        qualitative_features: list[str] = None,
        quantitative_features: list[str] = None,
        *,
        measures: list[Callable] = None,
        filters: list[Callable] = None,
        colsample: float = 1.0,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        warn(
            "feature_selection.FeatureSelector will be deprecated, please use selectors.ClassificationSelector instead.",
            DeprecationWarning,
        )
        # corresponding measures and filters
        quantitative_measures = None
        qualitative_measures = None
        if measures is not None:
            if (quantitative_features is not None) and any(quantitative_features):
                quantitative_measures = measures[:]
            elif (qualitative_features is not None) and any(qualitative_features):
                qualitative_measures = measures[:]

        quantitative_filters = None
        qualitative_filters = None
        if filters is not None:
            if (quantitative_features is not None) and any(quantitative_features):
                quantitative_filters = filters[:]
            elif (qualitative_features is not None) and any(qualitative_features):
                qualitative_filters = filters[:]

        # initiating BaseSelector with the corresponding list of measures
        super().__init__(
            n_best=n_best,
            quantitative_features=quantitative_features,
            qualitative_features=qualitative_features,
            quantitative_measures=quantitative_measures,
            qualitative_measures=qualitative_measures,
            quantitative_filters=quantitative_filters,
            qualitative_filters=qualitative_filters,
            colsample=colsample,
            verbose=verbose,
            **kwargs,
        )
