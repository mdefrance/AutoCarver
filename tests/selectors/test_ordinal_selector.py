"""Set of tests for OrdinalSelector module."""

from AutoCarver.features import Features
from AutoCarver.selectors import OrdinalSelector
from AutoCarver.selectors.measures import KruskalEtaSquaredMeasure, ModeMeasure, NanMeasure, SpearmanMeasure


def test_ordinal_selector_treats_target_as_ranked() -> None:
    """The ordinal target is ranked numerically (like regression), not as qualitative."""
    assert OrdinalSelector._target_is_qualitative is False


def test_ordinal_selector_default_measures(features_object: Features) -> None:
    """Defaults reuse the rank-based measures (Spearman + Kruskal-η²) plus the gate defaults."""
    selector = OrdinalSelector(n_best_per_type=2, features=features_object)
    names = {measure.__name__ for measure in selector.measures}
    assert SpearmanMeasure().__name__ in names
    assert KruskalEtaSquaredMeasure().__name__ in names
    assert NanMeasure().__name__ in names
    assert ModeMeasure().__name__ in names
