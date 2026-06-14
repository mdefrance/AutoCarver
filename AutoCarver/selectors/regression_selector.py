"""Tools to select the best Quantitative and Qualitative features for a Regression task."""

from AutoCarver.selectors.measures import BaseMeasure, KruskalEtaSquaredMeasure, SpearmanMeasure
from AutoCarver.selectors.utils.base_selector import BaseSelector


class RegressionSelector(BaseSelector):
    """A pipeline of measures to perform a feature pre-selection that maximizes association
    with a quantitative target.
    """

    __name__ = "RegressionSelector"
    _target_is_qualitative = False

    def _default_measures(self) -> list[BaseMeasure]:
        """Spearman's rho ranks quantitative features, Kruskal-η² (reversed) ranks qualitative ones."""
        return [SpearmanMeasure(), KruskalEtaSquaredMeasure()]
