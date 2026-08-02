"""Tools to select the best Quantitative and Qualitative features for an ordinal target."""

from AutoCarver.selectors.regression_selector import RegressionSelector


class OrdinalSelector(RegressionSelector):
    """A pipeline of measures to perform a feature pre-selection that maximizes association
    with an **ordinal** target.

    The integer-encoded ordinal target is treated as a numeric rank, so the same
    rank-based measures used for regression apply and stay order-aware w.r.t. the
    target: Spearman's rho for quantitative features, Kruskal-η² (reversed) for
    qualitative ones. No ordinal-specific selection statistic is required.
    """

    __name__ = "OrdinalSelector"
