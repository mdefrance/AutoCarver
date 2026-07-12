"""Tests for the NaN-vs-values rescue fix.

Before this fix, whenever the non-NaN combination search found nothing viable
(e.g. a feature with a single non-NaN modality), the NaN fan-out — which always
includes the ``[[all_values], [NaN]]`` partition — was silently skipped and the
feature was dropped untested, even when that partition was highly predictive
(informative missingness). See work/todos/PLAN-nan-vs-values.md.
"""

import numpy as np
import pandas as pd
from pytest import raises

from AutoCarver import BinaryCarver, ContinuousCarver, MulticlassCarver, OrdinalCarver
from AutoCarver.combinations import (
    CramervCombinations,
    KendallTauCCombinations,
    KruskalCombinations,
    TschuprowtMulticlassCombinations,
)
from AutoCarver.discretizers import ProcessingConfig
from AutoCarver.features import Features, OrdinalFeature


def _exact_binary_column(
    n_nan: int, n_val: int, nan_rate: float, val_rate: float, val_label: str = "A"
) -> tuple[pd.DataFrame, pd.Series]:
    """Deterministic categorical column: ``n_nan`` NaN rows + ``n_val`` ``val_label``
    rows, with ``y`` built from exact positive counts so the two groups' target
    rates are known exactly (not subject to sampling noise)."""
    feature = np.array([None] * n_nan + [val_label] * n_val, dtype=object)
    n_nan_pos = round(nan_rate * n_nan)
    n_val_pos = round(val_rate * n_val)
    y = np.array([1] * n_nan_pos + [0] * (n_nan - n_nan_pos) + [1] * n_val_pos + [0] * (n_val - n_val_pos))
    return pd.DataFrame({"feature": feature}), pd.Series(y)


def test_binary_carver_nan_vs_value_informative_categorical_kept():
    """50% NaN / 50% "A", informative missingness (P(y=1|NaN)=0.8, P(y=1|A)=0.2) -> KEPT."""
    X, y = _exact_binary_column(1000, 1000, nan_rate=0.8, val_rate=0.2)
    carver = BinaryCarver(
        min_freq=0.05,
        max_n_mod=4,
        features=Features(categoricals=["feature"]),
        combination_evaluator=CramervCombinations(),
        config=ProcessingConfig(dropna=True, ordinal_encoding=True, copy=True),
    )
    X_transformed = carver.fit_transform(X, y)

    assert "feature" not in [f.name for f in carver.dropped_features]
    assert [f.name for f in carver.features] == ["feature"]
    assert X_transformed["feature"].nunique() == 2


def test_binary_carver_nan_vs_value_informative_numeric_kept():
    """Same as above with a numeric constant (50% NaN / 50% ``1.0``) instead of a category."""
    feature = np.array([np.nan] * 1000 + [1.0] * 1000)
    n_nan_pos, n_val_pos = 800, 200
    y = pd.Series([1] * n_nan_pos + [0] * (1000 - n_nan_pos) + [1] * n_val_pos + [0] * (1000 - n_val_pos))
    X = pd.DataFrame({"feature": feature})

    carver = BinaryCarver(
        min_freq=0.05,
        max_n_mod=4,
        features=Features(numericals=["feature"]),
        combination_evaluator=CramervCombinations(),
        config=ProcessingConfig(dropna=True, ordinal_encoding=True, copy=True),
    )
    X_transformed = carver.fit_transform(X, y)

    assert "feature" not in [f.name for f in carver.dropped_features]
    assert [f.name for f in carver.features] == ["feature"]
    assert X_transformed["feature"].nunique() == 2


def test_binary_carver_nan_vs_value_noninformative_dropped():
    """50% NaN / 50% "A", y independent of missingness (same rate both sides) -> DROPPED,
    with a non-empty history recording the [[A], [NaN]] candidate as non-viable."""
    X, y = _exact_binary_column(1000, 1000, nan_rate=0.5, val_rate=0.5)
    carver = BinaryCarver(
        min_freq=0.05,
        max_n_mod=4,
        features=Features(categoricals=["feature"]),
        combination_evaluator=CramervCombinations(),
        config=ProcessingConfig(dropna=True, ordinal_encoding=True, copy=True),
    )
    carver.fit_transform(X, y)

    assert [f.name for f in carver.dropped_features] == ["feature"]
    history = carver.dropped_features[0].history
    assert len(history) > 0
    assert not history["viable"].fillna(False).any()


def test_binary_carver_nan_vs_value_rare_kept_with_low_min_freq():
    """94% NaN / 6% "A", informative missingness, min_freq=0.02 -> KEPT (2 modalities)."""
    X, y = _exact_binary_column(1880, 120, nan_rate=0.8, val_rate=0.1)
    carver = BinaryCarver(
        min_freq=0.02,
        max_n_mod=4,
        features=Features(categoricals=["feature"]),
        combination_evaluator=CramervCombinations(),
        config=ProcessingConfig(dropna=True, ordinal_encoding=True, copy=True),
    )
    X_transformed = carver.fit_transform(X, y)

    assert "feature" not in [f.name for f in carver.dropped_features]
    assert X_transformed["feature"].nunique() == 2


def test_binary_carver_nan_vs_value_rare_dropped_with_high_min_freq():
    """Same data as above, but min_freq=0.10 makes the 6% "A" modality non-representative
    -> DROPPED, and the drop reason mentions min_freq / "Non-representative"."""
    X, y = _exact_binary_column(1880, 120, nan_rate=0.8, val_rate=0.1)
    carver = BinaryCarver(
        min_freq=0.10,
        max_n_mod=4,
        features=Features(categoricals=["feature"]),
        combination_evaluator=CramervCombinations(),
        config=ProcessingConfig(dropna=True, ordinal_encoding=True, copy=True),
    )
    carver.fit_transform(X, y)

    assert [f.name for f in carver.dropped_features] == ["feature"]
    dropped_reasons = carver.summary["dropped_reason"].dropna().astype(str)
    assert dropped_reasons.str.contains("Non-representative").any()


def test_binary_carver_nan_vs_value_multi_modality_rescue():
    """3 non-NaN modalities (A, B, C) share the exact same target rate (0.3), so every
    non-NaN combination fails the distinct-rates test; NaN has a different rate (0.7).
    Before this fix the feature was dropped untested; now it's rescued via the
    all-values-vs-NaN partition."""
    feature = np.array([None] * 500 + ["A"] * 500 + ["B"] * 500 + ["C"] * 500, dtype=object)
    y = np.array([1] * 350 + [0] * 150 + ([1] * 150 + [0] * 350) * 3)
    X = pd.DataFrame({"feature": feature})
    y = pd.Series(y)

    carver = BinaryCarver(
        min_freq=0.05,
        max_n_mod=4,
        features=Features(categoricals=["feature"]),
        combination_evaluator=CramervCombinations(),
        config=ProcessingConfig(dropna=True, ordinal_encoding=True, copy=True),
    )
    X_transformed = carver.fit_transform(X, y)

    assert "feature" not in [f.name for f in carver.dropped_features]
    assert X_transformed["feature"].nunique() == 2


def test_binary_carver_nan_vs_value_dropna_false_single_value_kept_raw():
    """dropna=False: NaN can never be merged into another group (it must stay literal in
    the output, see Features.unfillna), but the all-values-vs-NaN split is still tested
    for viability when the non-NaN search finds nothing on its own. Informative
    missingness (single non-nan value) -> KEPT, with NaN left as raw float NaN (not
    ordinal-encoded) and the non-nan value(s) normally encoded."""
    X, y = _exact_binary_column(1000, 1000, nan_rate=0.8, val_rate=0.2)
    carver = BinaryCarver(
        min_freq=0.05,
        max_n_mod=4,
        features=Features(categoricals=["feature"]),
        combination_evaluator=CramervCombinations(),
        config=ProcessingConfig(dropna=False, ordinal_encoding=True, copy=True),
    )
    X_transformed = carver.fit_transform(X, y)

    assert "feature" not in [f.name for f in carver.dropped_features]
    assert X_transformed["feature"].isna().sum() == 1000
    assert X_transformed.loc[X_transformed["feature"].notna(), "feature"].nunique() == 1


def test_binary_carver_nan_vs_value_dropna_false_noninformative_dropped():
    """Same shape as above, but y independent of missingness -> still DROPPED: the
    all-values-vs-NaN split fails the distinct-rates viability test just like it does
    for dropna=True."""
    X, y = _exact_binary_column(1000, 1000, nan_rate=0.5, val_rate=0.5)
    carver = BinaryCarver(
        min_freq=0.05,
        max_n_mod=4,
        features=Features(categoricals=["feature"]),
        combination_evaluator=CramervCombinations(),
        config=ProcessingConfig(dropna=False, ordinal_encoding=True, copy=True),
    )
    carver.fit_transform(X, y)

    assert [f.name for f in carver.dropped_features] == ["feature"]


def test_binary_carver_nan_vs_value_dropna_false_multi_modality_merged():
    """3 non-nan modalities (A, B, C) share the exact same target rate; with dropna=False
    the only safe rescue candidate merges all of them into a single group (NaN can only
    ever be split off entirely, never folded into a partial group) -- KEPT, non-nan values
    merged into one code, NaN left raw."""
    feature = np.array([None] * 500 + ["A"] * 500 + ["B"] * 500 + ["C"] * 500, dtype=object)
    y = pd.Series([1] * 350 + [0] * 150 + ([1] * 150 + [0] * 350) * 3)
    X = pd.DataFrame({"feature": feature})

    carver = BinaryCarver(
        min_freq=0.05,
        max_n_mod=4,
        features=Features(categoricals=["feature"]),
        combination_evaluator=CramervCombinations(),
        config=ProcessingConfig(dropna=False, ordinal_encoding=True, copy=True),
    )
    X_transformed = carver.fit_transform(X, y)

    assert "feature" not in [f.name for f in carver.dropped_features]
    assert X_transformed["feature"].isna().sum() == 500
    assert X_transformed.loc[X_transformed["feature"].notna(), "feature"].nunique() == 1


def test_binary_evaluator_single_modality_no_nan_bails_out_with_history():
    """A constant feature with no NaN can never be rescued (nothing to split against):
    Gate 1 still bails out immediately, but (per this fix) it now historizes the raw
    distribution first, so the dropped feature keeps a legible history instead of an
    empty one."""
    feature = OrdinalFeature("feature", ["A"])
    feature.has_nan = False
    evaluator = CramervCombinations()
    xagg = pd.DataFrame({0: [500], 1: [500]}, index=["A"])

    result = evaluator.get_best_combination(feature, xagg, xagg, max_n_mod=4, min_freq=0.05, dropna=True)

    assert result is None
    assert len(feature.history) == 1
    assert feature.history.iloc[0]["info"] == "Raw distribution"
    assert not feature.history.iloc[0]["viable"]


def test_continuous_carver_nan_vs_value_informative_kept():
    """Continuous target: 50% NaN / 50% constant 1.0, shifted mean between the two
    groups -> KEPT."""
    feature = np.array([np.nan] * 1000 + [1.0] * 1000)
    rng = np.random.default_rng(0)
    y = pd.Series(np.concatenate([rng.normal(5, 1, 1000), rng.normal(0, 1, 1000)]))
    X = pd.DataFrame({"feature": feature})

    carver = ContinuousCarver(
        min_freq=0.05,
        max_n_mod=4,
        features=Features(numericals=["feature"]),
        combination_evaluator=KruskalCombinations(),
        config=ProcessingConfig(dropna=True, ordinal_encoding=True, copy=True),
    )
    X_transformed = carver.fit_transform(X, y)

    assert "feature" not in [f.name for f in carver.dropped_features]
    assert X_transformed["feature"].nunique() == 2


def test_ordinal_carver_nan_vs_value_informative_kept():
    """Ordinal target: 50% NaN / 50% constant 1.0, shifted class mix between the two
    groups -> KEPT."""
    feature = np.array([np.nan] * 1000 + [1.0] * 1000)
    y = pd.Series([3] * 700 + [2] * 200 + [1] * 100 + [1] * 700 + [2] * 200 + [3] * 100)
    X = pd.DataFrame({"feature": feature})

    carver = OrdinalCarver(
        min_freq=0.05,
        max_n_mod=4,
        features=Features(numericals=["feature"]),
        combination_evaluator=KendallTauCCombinations(),
        config=ProcessingConfig(dropna=True, ordinal_encoding=True, copy=True),
    )
    X_transformed = carver.fit_transform(X, y)

    assert "feature" not in [f.name for f in carver.dropped_features]
    assert X_transformed["feature"].nunique() == 2


def test_multiclass_carver_nan_vs_value_informative_kept():
    """Multiclass target: NaN / constant 1.0 (unequal group sizes -- the multiclass
    target rate falls back to a frequency-based CA axis for exactly 2 modalities, see
    AutoCarver/discretizers/utils/correspondence_analysis.py:fit_ca_axis, so equal-sized
    groups would tie regardless of class mix), shifted class mix between the two
    groups -> KEPT."""
    feature = np.array([np.nan] * 1100 + [1.0] * 900)
    y = pd.Series(["cat"] * 770 + ["dog"] * 220 + ["bird"] * 110 + ["bird"] * 630 + ["dog"] * 180 + ["cat"] * 90)
    X = pd.DataFrame({"feature": feature})

    carver = MulticlassCarver(
        min_freq=0.05,
        max_n_mod=4,
        features=Features(numericals=["feature"]),
        combination_evaluator=TschuprowtMulticlassCombinations(),
        config=ProcessingConfig(dropna=True, ordinal_encoding=True, copy=True),
    )
    X_transformed = carver.fit_transform(X, y)

    assert "feature" not in [f.name for f in carver.dropped_features]
    assert X_transformed["feature"].nunique() == 2


def test_binary_carver_constant_no_nan_raises():
    """A single-value feature with no NaN has nothing to rescue against and is
    rejected upfront by the discretizer's frequency check (pre-existing, unrelated
    to this fix -- a zero-variance column is never carvable)."""
    X = pd.DataFrame({"feature": ["A"] * 2000})
    y = pd.Series([1] * 1000 + [0] * 1000)
    carver = BinaryCarver(
        min_freq=0.05,
        max_n_mod=4,
        features=Features(categoricals=["feature"]),
        combination_evaluator=CramervCombinations(),
        config=ProcessingConfig(dropna=True, ordinal_encoding=True, copy=True, rescue_rare=False),
    )
    with raises(ValueError):
        carver.fit_transform(X, y)
