"""Set of tests for ClassificationSelector module."""

import numpy as np
import pandas as pd
from pytest import fixture, raises, warns

from AutoCarver.features import Features
from AutoCarver.selectors import ClassificationSelector, SelectionConfig
from AutoCarver.selectors.filters import CramervFilter, SpearmanFilter
from AutoCarver.selectors.measures import (
    KruskalEtaSquaredMeasure,
    NanMeasure,
    SpearmanMeasure,
    TschuprowtMeasure,
)
from AutoCarver.selectors.utils.base_selector import remove_default_metrics

# NB: ``__name__`` is overridden per class, but reading it off the *class* hits the
# metaclass descriptor ("NanMeasure") instead of the override ("Nan") — only instances
# carry the real metric name, so these are spelled out as literals.
GATES = {"Nan", "Mode"}


def names(metrics: list) -> set[str]:
    """metric names of a per-type slot"""
    return {metric.__name__ for metric in metrics}


def test_classification_selector_default_measures(features_object: Features) -> None:
    """Task defaults land in the right per-type slot, alongside the mandatory gates."""
    selector = ClassificationSelector(features_object, 2)

    assert names(selector.measures["qualitatives"]) == GATES | {"TschuprowtMeasure"}
    assert names(selector.measures["quantitatives"]) == GATES | {"KruskalEtaSquaredMeasure"}
    # every type has exactly one ranking measure
    for kind in ("qualitatives", "quantitatives"):
        assert len(remove_default_metrics(selector.measures[kind])) == 1


def test_classification_selector_default_filters(features_object: Features) -> None:
    """Validity filters are always added; redundancy filters are routed per type."""
    selector = ClassificationSelector(features_object, 2)

    # NonDefaultValid is not `is_default`: it re-runs the validity gate on the ranking pass
    assert names(selector.filters["qualitatives"]) == {"Valid", "NonDefaultValid", "TschuprowtFilter"}
    assert names(selector.filters["quantitatives"]) == {"Valid", "NonDefaultValid", "SpearmanFilter"}


def test_classification_selector_custom_measures(features_object: Features) -> None:
    """A user-supplied gate keeps its threshold; missing gates are still added."""
    config = SelectionConfig(
        qualitative_measures=[NanMeasure(threshold=0.3), TschuprowtMeasure(threshold=0.1)],
        quantitative_measures=[KruskalEtaSquaredMeasure(threshold=0.05)],
    )
    selector = ClassificationSelector(features_object, 2, config=config)

    quali = selector.measures["qualitatives"]
    assert names(quali) == GATES | {"TschuprowtMeasure"}
    # the user's NanMeasure instance won: its threshold survived
    assert next(measure for measure in quali if measure.__name__ == "Nan").threshold == 0.3
    assert remove_default_metrics(quali)[0].threshold == 0.1
    assert remove_default_metrics(selector.measures["quantitatives"])[0].threshold == 0.05


def test_classification_selector_rejects_regression_measure(features_object: Features) -> None:
    """A non-reversible measure of the wrong target type is refused at construction."""
    with raises(ValueError, match="does not match the target type"):
        ClassificationSelector(features_object, 2, config=SelectionConfig(quantitative_measures=[SpearmanMeasure()]))


def test_classification_selector_rejects_wrong_type_slot(features_object: Features) -> None:
    """A measure/filter dropped into the wrong per-type slot is refused at construction."""
    with raises(ValueError, match="does not apply to quantitative features"):
        ClassificationSelector(features_object, 2, config=SelectionConfig(quantitative_measures=[TschuprowtMeasure()]))

    with raises(ValueError, match="does not apply to qualitative features"):
        ClassificationSelector(features_object, 2, config=SelectionConfig(qualitative_filters=[SpearmanFilter()]))


def test_classification_selector_falls_back_on_empty_measures(features_object: Features) -> None:
    """An explicitly empty per-type slot falls back to that type's default, with a warning."""
    with warns(UserWarning, match="no ranking measure applies to quantitative features"):
        selector = ClassificationSelector(features_object, 2, config=SelectionConfig(quantitative_measures=[]))

    assert names(remove_default_metrics(selector.measures["quantitatives"])) == {"KruskalEtaSquaredMeasure"}


@fixture
def mixed_sample() -> tuple[Features, pd.DataFrame, pd.Series]:
    """A mixed qualitative/quantitative dataset with a binary target"""
    rng = np.random.default_rng(0)
    size = 300
    X = pd.DataFrame(
        {
            "cat1": rng.choice(list("abc"), size),
            "cat2": rng.choice(list("xy"), size),
            "num1": rng.normal(size=size),
            "num2": rng.normal(size=size),
            "num3": rng.normal(size=size),
        }
    )
    y = pd.Series((X["num1"] + rng.normal(size=size) > 0).astype(int), name="target")
    features = Features(categoricals=["cat1", "cat2"], numericals=["num1", "num2", "num3"])
    return features, X, y


def test_quantitatives_kept_when_only_qualitative_measures_given(
    mixed_sample: tuple[Features, pd.DataFrame, pd.Series],
) -> None:
    """Regression: qualitative-only measures used to silently drop every quantitative feature."""
    features, X, y = mixed_sample
    config = SelectionConfig(
        qualitative_measures=[TschuprowtMeasure(threshold=0.001)],
        qualitative_filters=[CramervFilter(threshold=0.7)],
    )
    selector = ClassificationSelector(features, 4, config=config).fit(X, y)

    selected = selector.selected_features
    assert len(selected.quantitatives) > 0
    assert len(selected.qualitatives) > 0


def test_summary_shares_columns_across_types(
    mixed_sample: tuple[Features, pd.DataFrame, pd.Series],
) -> None:
    """Both types land in the same `measure`/`association` columns, not a ragged frame."""
    features, X, y = mixed_sample
    selector = ClassificationSelector(features, 3).fit(X, y)
    summary = selector.summary

    assert set(summary.columns) == {
        "feature",
        "Nan",
        "Mode",
        "measure",
        "association",
        "rank",
        "filter",
        "redundancy",
        "filtered_with",
        "selected",
    }
    # one row per feature, and both types share the association column
    assert len(summary) == len(features)
    assert set(summary["measure"]) == {"Tschuprowt", "KruskalEtaSquared"}
    assert summary["association"].notna().all()
    assert summary["selected"].sum() == len(selector.selected_features)
