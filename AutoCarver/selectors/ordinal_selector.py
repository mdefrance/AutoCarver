"""Tools to select the best Quantitative and Qualitative features for an ordinal target."""

from AutoCarver.selectors.measures import BaseMeasure, KruskalEtaSquaredMeasure, SpearmanMeasure
from AutoCarver.selectors.utils.base_selector import BaseSelector


class OrdinalSelector(BaseSelector):
    """A pipeline of measures to perform a feature pre-selection that maximizes association
    with an **ordinal** target.

    The integer-encoded ordinal target is treated as a numeric rank, so the same
    rank-based measures used for regression apply and stay order-aware w.r.t. the
    target: Spearman's rho for quantitative features, Kruskal-η² (reversed) for
    qualitative ones. No ordinal-specific selection statistic is required.
    """

    __name__ = "OrdinalSelector"
    # the ordinal target is ranked numerically (like regression) for measure orientation
    _target_is_qualitative = False

    def _default_measures(self) -> list[BaseMeasure]:
        """Spearman's rho ranks quantitative features, Kruskal-η² (reversed) ranks qualitative ones."""
        return [SpearmanMeasure(), KruskalEtaSquaredMeasure()]
