"""Set of tests for RegressionSelector module."""

from pytest import raises, warns

from AutoCarver.features import Features
from AutoCarver.selectors import RegressionSelector, SelectionConfig
from AutoCarver.selectors.measures import TschuprowtMeasure
from AutoCarver.selectors.utils.base_selector import remove_default_metrics

# NB: reading ``__name__`` off a metric *class* hits the metaclass descriptor, not the
# per-class override — only instances carry the real metric name (see the sibling
# classification tests), hence the literals below.
GATES = {"Nan", "Mode"}


def names(metrics: list) -> set[str]:
    """metric names of a per-type slot"""
    return {metric.__name__ for metric in metrics}


def test_regression_selector_default_measures(features_object: Features) -> None:
    """Kruskal-η² is *reversed* for a quantitative target, so it ranks qualitative features."""
    selector = RegressionSelector(features_object, 2)

    assert names(selector.measures["quantitatives"]) == GATES | {"SpearmanMeasure"}
    assert names(selector.measures["qualitatives"]) == GATES | {"KruskalEtaSquaredMeasure"}
    for kind in ("qualitatives", "quantitatives"):
        assert len(remove_default_metrics(selector.measures[kind])) == 1


def test_regression_selector_default_filters(features_object: Features) -> None:
    """Validity filters are always added; redundancy filters are routed per type."""
    selector = RegressionSelector(features_object, 2)

    # NonDefaultValid is not `is_default`: it re-runs the validity gate on the ranking pass
    assert names(selector.filters["qualitatives"]) == {"Valid", "NonDefaultValid", "TschuprowtFilter"}
    assert names(selector.filters["quantitatives"]) == {"Valid", "NonDefaultValid", "SpearmanFilter"}


def test_regression_selector_rejects_classification_measure(features_object: Features) -> None:
    """A non-reversible measure of the wrong target type is refused at construction."""
    with raises(ValueError, match="does not match the target type"):
        RegressionSelector(features_object, 2, config=SelectionConfig(qualitative_measures=[TschuprowtMeasure()]))


def test_regression_selector_falls_back_on_empty_measures(features_object: Features) -> None:
    """An explicitly empty per-type slot falls back to that type's default, with a warning."""
    with warns(UserWarning, match="no ranking measure applies to qualitative features"):
        selector = RegressionSelector(features_object, 2, config=SelectionConfig(qualitative_measures=[]))

    assert names(remove_default_metrics(selector.measures["qualitatives"])) == {"KruskalEtaSquaredMeasure"}


def test_regression_selector_default_filters_drop_a_duplicate() -> None:
    """With default filters, an exact duplicate of a better-ranked feature is dropped; an independent one is kept."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    signal = rng.normal(size=500)
    X = pd.DataFrame({"a": signal, "a_dup": signal, "other": rng.normal(size=500)})
    y = pd.Series(signal + 0.5 * X["other"] + rng.normal(scale=0.1, size=500))

    selector = RegressionSelector(Features(numericals=["a", "a_dup", "other"]), 3).fit(X, y)
    selected = selector.selected_features.versions

    assert "other" in selected
    assert ("a" in selected) != ("a_dup" in selected)
