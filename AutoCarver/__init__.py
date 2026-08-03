from importlib.metadata import PackageNotFoundError, version

from AutoCarver.carvers.binary_carver import BinaryCarver
from AutoCarver.carvers.continuous_carver import ContinuousCarver
from AutoCarver.carvers.multiclass_carver import MulticlassCarver
from AutoCarver.carvers.one_vs_rest_carver import OneVsRestCarver
from AutoCarver.carvers.ordinal_carver import OrdinalCarver
from AutoCarver.features import Features
from AutoCarver.selectors import ClassificationSelector, OrdinalSelector, RegressionSelector, SelectionConfig
from AutoCarver.stability import StabilityReport, evaluate_stability

try:
    __version__ = version("AutoCarver")
except PackageNotFoundError:  # package not installed
    __version__ = "unknown"

__all__ = [
    "__version__",
    "BinaryCarver",
    "ContinuousCarver",
    "Features",
    "MulticlassCarver",
    "OneVsRestCarver",
    "OrdinalCarver",
    "ClassificationSelector",
    "OrdinalSelector",
    "RegressionSelector",
    "StabilityReport",
    "evaluate_stability",
    "SelectionConfig",
]
