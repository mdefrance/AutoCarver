"""Single home for statistical primitives shared by carvers, selectors and stability."""

from AutoCarver.stats.chi2 import cramerv_tschuprowt, cramerv_tschuprowt_unrounded, pearson_chi2
from AutoCarver.stats.correspondence_analysis import CAAxis, ca_row_scores, fit_ca_axis
from AutoCarver.stats.frequency_ci import is_significantly_below, wilson_upper_bound
from AutoCarver.stats.kruskal import h_from_rank_sums, tie_correction
from AutoCarver.stats.ridits import ridit_scores_for_levels, ridits_from_counts

__all__ = [
    "CAAxis",
    "ca_row_scores",
    "fit_ca_axis",
    "is_significantly_below",
    "wilson_upper_bound",
    "ridit_scores_for_levels",
    "ridits_from_counts",
    "pearson_chi2",
    "cramerv_tschuprowt",
    "cramerv_tschuprowt_unrounded",
    "tie_correction",
    "h_from_rank_sums",
]
