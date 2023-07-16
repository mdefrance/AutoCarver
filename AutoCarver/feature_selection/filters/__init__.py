""" Loads association filters."""

from .base_filters import measure_filter, thresh_filter
from .qualitative_filters import cramerv_filter, tschuprowt_filter
from .quantitative_filters import pearson_filter, spearman_filter, vif_filter
