"""Tools to select the best Quantitative and Qualitative features for a Classification task."""

from AutoCarver.selectors.measures import BaseMeasure, KruskalEtaSquaredMeasure, TschuprowtMeasure
from AutoCarver.selectors.utils.base_selector import BaseSelector


class ClassificationSelector(BaseSelector):
    """A pipeline of measures to perform a feature pre-selection that maximizes association
    with a qualitative target.
    """

    __name__ = "ClassificationSelector"
    _target_is_qualitative = True

    def _default_measures(self) -> list[BaseMeasure]:
        """Tschuprow's T ranks qualitative features, Kruskal-η² ranks quantitative ones."""
        return [TschuprowtMeasure(), KruskalEtaSquaredMeasure()]
