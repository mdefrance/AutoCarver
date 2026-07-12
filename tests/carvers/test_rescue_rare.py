"""Tests for the rescue_rare fix.

Before this fix, a feature whose modality frequencies fail the qualitative
frequency check (a too-frequent mode -- including NaN as mode -- or no
frequent-enough modality) always raised a ``ValueError`` during discretization.
With ``ProcessingConfig(rescue_rare=True)``, such features are kept and, if
the normal carver combination search finds nothing viable, get one last
chance with the ``min_freq`` veto waived -- kept only if the target signal
holds on ``X_dev`` and/or every CV fold. See work/todos/PLAN-rescue-dominant.md.
"""

import numpy as np
import pandas as pd
import pytest
from pytest import raises

from AutoCarver import BinaryCarver, ContinuousCarver
from AutoCarver.combinations import CramervCombinations, KruskalCombinations
from AutoCarver.discretizers import ProcessingConfig
from AutoCarver.discretizers.qualitatives.nested_discretizer import check_frequencies
from AutoCarver.features import Features


def _dominant_column(
    n_dom: int, n_rare: int, dom_rate: float, rare_rate: float, dom_label: str = "A", rare_label: str = "B"
) -> tuple[pd.DataFrame, pd.Series]:
    """n_dom rows of dom_label + n_rare rows of rare_label; y built from exact
    positive counts so both groups' target rates are exact (not sampling noise)."""
    feature = np.array([dom_label] * n_dom + [rare_label] * n_rare, dtype=object)
    n_dom_pos = round(dom_rate * n_dom)
    n_rare_pos = round(rare_rate * n_rare)
    y = np.array([1] * n_dom_pos + [0] * (n_dom - n_dom_pos) + [1] * n_rare_pos + [0] * (n_rare - n_rare_pos))
    return pd.DataFrame({"feature": feature}), pd.Series(y)


def _binary_carver(rescue_rare: bool, **config_kwargs) -> BinaryCarver:
    return BinaryCarver(
        min_freq=0.05,
        max_n_mod=4,
        features=Features(categoricals=["feature"]),
        combination_evaluator=CramervCombinations(),
        config=ProcessingConfig(
            dropna=True, ordinal_encoding=True, copy=True, rescue_rare=rescue_rare, **config_kwargs
        ),
    )


def test_rescue_rare_default_unchanged_raises():
    """rescue_rare=False (default): dominant modality (99%) still raises, unchanged."""
    X, y = _dominant_column(3960, 40, 0.2, 0.9)
    carver = _binary_carver(rescue_rare=False)
    with raises(ValueError):
        carver.fit_transform(X, y)


def test_rescue_rare_informative_kept_with_x_dev():
    """Informative rare modality (dom rate=0.2, rare rate=0.9), rescue_rare=True,
    X_dev provided -> KEPT, 2 modalities, history mentions the rescue."""
    X, y = _dominant_column(3960, 40, 0.2, 0.9)
    X_dev, y_dev = _dominant_column(3960, 40, 0.2, 0.9)
    carver = _binary_carver(rescue_rare=True)
    carver.fit(X, y, X_dev=X_dev, y_dev=y_dev)
    X_transformed = carver.transform(X)

    assert "feature" not in [f.name for f in carver.dropped_features]
    assert X_transformed["feature"].nunique() == 2
    history = carver.features("feature").history
    assert history["viable"].fillna(False).any()
    viable_rows = history[history["viable"].fillna(False)]
    assert viable_rows["info"].astype(str).str.contains("rescue").any()


def test_rescue_rare_noninformative_dropped_with_x_dev():
    """Same shape, but rare modality has same rate as dominant -> DROPPED even with rescue."""
    X, y = _dominant_column(3960, 40, 0.2, 0.2)
    X_dev, y_dev = _dominant_column(3960, 40, 0.2, 0.2)
    carver = _binary_carver(rescue_rare=True)
    carver.fit(X, y, X_dev=X_dev, y_dev=y_dev)

    assert [f.name for f in carver.dropped_features] == ["feature"]
    history = carver.dropped_features[0].history
    assert len(history) > 0
    assert not history["viable"].fillna(False).any()


def test_rescue_rare_no_validation_view_skipped_with_warning():
    """rescue_rare=True but no X_dev/cv -> rescue skipped (warned), feature dropped."""
    X, y = _dominant_column(3960, 40, 0.2, 0.9)
    carver = _binary_carver(rescue_rare=True)
    with pytest.warns(UserWarning, match="rescue_rare"):
        carver.fit_transform(X, y)

    assert [f.name for f in carver.dropped_features] == ["feature"]


def test_rescue_rare_cv_only_kept():
    """rescue_rare=True with cv=3 (no X_dev) -> informative rare modality KEPT."""
    X, y = _dominant_column(3000, 180, 0.2, 0.9)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    X = X.iloc[idx].reset_index(drop=True)
    y = y.iloc[idx].reset_index(drop=True)

    carver = _binary_carver(rescue_rare=True)
    carver.fit(X, y, cv=3)
    X_transformed = carver.transform(X)

    assert "feature" not in [f.name for f in carver.dropped_features]
    assert X_transformed["feature"].nunique() == 2


def test_rescue_rare_nan_dominant_kept():
    """NaN as the dominant modality (99%) with an informative rare value -> KEPT
    under rescue_rare=True; raises under rescue_rare=False (NaN mode counts as
    too-common)."""
    feature = np.array([None] * 3960 + ["A"] * 40, dtype=object)
    y = np.array([1] * round(0.2 * 3960) + [0] * (3960 - round(0.2 * 3960)) + [1] * 36 + [0] * 4)
    X = pd.DataFrame({"feature": feature})
    y = pd.Series(y)

    X_dev, y_dev = X.copy(), y.copy()

    carver = _binary_carver(rescue_rare=True)
    carver.fit(X, y, X_dev=X_dev, y_dev=y_dev)
    X_transformed = carver.transform(X)

    assert "feature" not in [f.name for f in carver.dropped_features]
    assert X_transformed["feature"].nunique() == 2

    carver_default = _binary_carver(rescue_rare=False)
    with raises(ValueError):
        carver_default.fit_transform(X, y)


def test_rescue_rare_non_common_no_crash():
    """240 distinct labels x 16 rows each (every freq << min_freq=0.05), same target
    rate everywhere -> rescue_rare=True drops gracefully (no exception); rescue_rare=False
    raises."""
    labels = [f"v{i}" for i in range(240)]
    feature = np.repeat(labels, 16)
    X = pd.DataFrame({"feature": feature})
    y = pd.Series([0, 1] * (len(feature) // 2))

    carver = _binary_carver(rescue_rare=True)
    carver.fit_transform(X, y)
    assert [f.name for f in carver.dropped_features] == ["feature"]

    carver_default = _binary_carver(rescue_rare=False)
    with raises(ValueError):
        carver_default.fit_transform(X, y)


def test_rescue_rare_constant_feature_dropped_no_crash():
    """Constant feature (single value, no NaN) with rescue_rare=True -> warns at
    discretizer, dropped by the evaluator (single row), no exception."""
    X = pd.DataFrame({"feature": ["A"] * 2000})
    y = pd.Series([1] * 1000 + [0] * 1000)
    carver = _binary_carver(rescue_rare=True)
    carver.fit_transform(X, y)

    assert [f.name for f in carver.dropped_features] == ["feature"]


def test_rescue_rare_continuous_carver_kept():
    """ContinuousCarver: dominant modality (3960 rows) vs rare modality (40 rows)
    with a clearly shifted mean -> KEPT under rescue_rare=True with X_dev."""
    rng = np.random.default_rng(0)
    feature = np.array(["A"] * 3960 + ["B"] * 40, dtype=object)
    y = pd.Series(np.concatenate([rng.normal(0, 1, 3960), rng.normal(5, 1, 40)]))
    X = pd.DataFrame({"feature": feature})

    rng_dev = np.random.default_rng(0)
    y_dev = pd.Series(np.concatenate([rng_dev.normal(0, 1, 3960), rng_dev.normal(5, 1, 40)]))
    X_dev = X.copy()

    carver = ContinuousCarver(
        min_freq=0.05,
        max_n_mod=4,
        features=Features(categoricals=["feature"]),
        combination_evaluator=KruskalCombinations(),
        config=ProcessingConfig(dropna=True, ordinal_encoding=True, copy=True, rescue_rare=True),
    )
    carver.fit(X, y, X_dev=X_dev, y_dev=y_dev)
    X_transformed = carver.transform(X)

    assert "feature" not in [f.name for f in carver.dropped_features]
    assert X_transformed["feature"].nunique() == 2


def test_check_frequencies_rescue_rare_warns_instead_of_raising():
    """check_frequencies unit test: rescue_rare=True warns (mentioning both
    offending features) instead of raising; rescue_rare=False still raises."""
    too_common = np.array(["A"] * 995 + ["B"] * 5)
    non_common = np.repeat([f"v{i}" for i in range(200)], 5)
    X = pd.DataFrame({"too_common": too_common, "non_common": non_common})
    features = Features(categoricals=["too_common", "non_common"])
    features.fit(X, pd.Series([0, 1] * 500))

    with pytest.warns(UserWarning, match="too_common"):
        result = check_frequencies(features, X, min_freq=0.05, name="Test", rescue_rare=True)
    assert result is None

    with raises(ValueError, match="too_common"):
        check_frequencies(features, X, min_freq=0.05, name="Test", rescue_rare=False)
