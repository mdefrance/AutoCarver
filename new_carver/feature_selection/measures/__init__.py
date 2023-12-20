""" Loads association measures."""

from ...selectors.measures.base_measures import (
    dtype_measure,
    make_measure,
    mode_measure,
    nans_measure,
)
from ...selectors.measures.qualitative_measures import (
    chi2_measure,
    cramerv_measure,
    tschuprowt_measure,
)
from ...selectors.measures.quantitative_measures import (
    R_measure,
    iqr_measure,
    kruskal_measure,
    zscore_measure,
)
