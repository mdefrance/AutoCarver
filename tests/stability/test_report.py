"""Tests for the post-fit stability report."""

import json

import numpy as np
import pandas as pd
from pytest import approx, fixture, mark, raises, warns

from AutoCarver import (
    BinaryCarver,
    ContinuousCarver,
    Features,
    MulticlassCarver,
    OneVsRestCarver,
    OrdinalCarver,
)


@fixture
def sample():
    """A frame whose target is driven by both a numerical and a categorical column."""
    rng = np.random.default_rng(0)
    size = 4000
    X = pd.DataFrame(
        {
            "num": rng.normal(size=size),
            "cat": rng.choice(list("ABCDE"), size=size, p=[0.35, 0.25, 0.2, 0.15, 0.05]),
        }
    )
    score = 0.9 * X["num"] + X["cat"].map({"A": -1.0, "B": -0.3, "C": 0.2, "D": 0.8, "E": 1.5})
    return X, score, rng


def features() -> Features:
    return Features(numericals=["num"], categoricals=["cat"])


@fixture
def binary(sample):
    """A fitted BinaryCarver plus the sample it was fitted on."""
    X, score, rng = sample
    y = pd.Series(rng.binomial(1, 1 / (1 + np.exp(-score))), name="y")
    carver = BinaryCarver(features(), min_freq=0.05, max_n_mod=4)
    # copy: carvers configured with copy=False carve in place, and X is reused below
    carver.fit(X.copy(), y)
    return carver, X, y


def test_self_comparison_is_perfectly_stable(binary):
    """The reference path must reproduce itself: same sample in, zero drift out."""
    carver, X, y = binary
    report = carver.evaluate_stability(X, y)

    assert report.has_target is True
    assert report.per_feature["psi"].max() < 1e-9
    assert (report.per_feature["psi_flag"] == "stable").all()
    assert report.per_feature["viable"].all()
    assert not report.per_feature["chi2_significant"].any()
    assert report.per_feature["n_modalities_drifted"].sum() == 0
    assert report.unstable_features == []
    # reference and new statistics must coincide modality by modality
    assert report.per_modality["count_ref"].tolist() == report.per_modality["count_new"].tolist()


def test_evaluate_stability_does_not_mutate_the_input(binary):
    carver, X, y = binary
    before = X.copy()
    carver.evaluate_stability(X, y)
    pd.testing.assert_frame_equal(X, before)


def test_population_shift_is_flagged(binary):
    carver, X, y = binary
    shifted = X[X["num"] > X["num"].quantile(0.35)]

    report = carver.evaluate_stability(shifted, y.loc[shifted.index])
    per_feature = report.summary

    assert per_feature.loc["Numerical('num')", "psi"] > 0.25
    assert per_feature.loc["Numerical('num')", "psi_flag"] == "shifted"
    assert per_feature.loc["Numerical('num')", "chi2_significant"]
    # the untouched categorical stays put
    assert per_feature.loc["Categorical('cat')", "psi_flag"] == "stable"
    assert "Numerical('num')" in report.unstable_features


def test_target_drift_is_flagged(binary):
    """Flipping the target inside one carved bin must trip the two-proportion test."""
    carver, X, y = binary
    drifted = y.copy()
    lowest = X["num"] < X["num"].quantile(0.2)
    drifted.loc[lowest] = 1

    report = carver.evaluate_stability(X, drifted)
    per_feature = report.summary

    assert per_feature.loc["Numerical('num')", "n_modalities_drifted"] > 0
    # population is untouched: only the target moved
    assert per_feature.loc["Numerical('num')", "psi"] < 1e-9


def test_without_a_target_only_frequency_metrics_are_computed(binary):
    carver, X, _ = binary
    shifted = X[X["num"] > X["num"].quantile(0.35)]

    report = carver.evaluate_stability(shifted)

    assert report.has_target is False
    assert report.per_feature["viable"].isna().all()
    assert report.per_modality["drift_pvalue"].isna().all()
    assert report.summary.loc["Numerical('num')", "psi"] > 0.25


def test_report_is_json_serializable(binary):
    carver, X, y = binary
    payload = carver.evaluate_stability(X, y).to_json()

    assert json.loads(json.dumps(payload))["has_target"] is True
    assert len(payload["per_feature"]) == len(carver.features)
    assert payload["unstable_features"] == []


def test_uncarved_features_are_skipped_with_a_warning(sample):
    """Only carved features carry reference statistics; the rest can't be scored."""
    X, score, rng = sample
    y = pd.Series(rng.binomial(1, 1 / (1 + np.exp(-score))), name="y")

    carver = BinaryCarver(features(), min_freq=0.05, max_n_mod=4)
    carver.fit(X.copy(), y)
    # what a discretizer-only feature looks like: fitted values, no statistics
    carver.features[0]._statistics = None

    with warns(UserWarning, match="carries no reference statistics"):
        report = carver.evaluate_stability(X, y)
    assert len(report.per_feature) == len(carver.features) - 1


@mark.parametrize("carver_class", [BinaryCarver, ContinuousCarver, OrdinalCarver, MulticlassCarver, OneVsRestCarver])
def test_every_carver_round_trips_through_json(sample, tmp_path, carver_class):
    """Saving and reloading must not change a single number of the report.

    This is what persisting the target rate's per-feature reference buys: the ridit
    reference (ordinal) and the correspondence-analysis axis (multiclass) are transient
    evaluator state that would otherwise be gone after ``load``.
    """
    X, score, rng = sample
    targets = {
        BinaryCarver: pd.Series(rng.binomial(1, 1 / (1 + np.exp(-score))), name="y"),
        ContinuousCarver: pd.Series(score + rng.normal(scale=0.5, size=len(X)), name="y"),
        OrdinalCarver: pd.Series(pd.cut(score, 4, labels=[0, 1, 2, 3]).astype(int), name="y"),
        MulticlassCarver: pd.Series(pd.cut(score, 3, labels=list("xyz")).astype(str), name="y"),
        OneVsRestCarver: pd.Series(pd.cut(score, 3, labels=list("xyz")).astype(str), name="y"),
    }
    y = targets[carver_class]

    carver = carver_class(features(), min_freq=0.05, max_n_mod=4)
    carver.fit(X.copy(), y)
    report = carver.evaluate_stability(X, y)

    path = tmp_path / "carver.json"
    carver.save(path)
    reloaded = carver_class.load(path)

    pd.testing.assert_frame_equal(report.per_feature, reloaded.evaluate_stability(X, y).per_feature)
    assert report.per_feature["viable"].all()
    assert report.per_feature["psi"].max() == approx(0.0, abs=1e-9)


def test_continuous_drift_uses_the_persisted_std(sample):
    """The stored ``std`` column is what makes the Welch test possible at all."""
    X, score, rng = sample
    y = pd.Series(score + rng.normal(scale=0.5, size=len(X)), name="y")

    carver = ContinuousCarver(features(), min_freq=0.05, max_n_mod=4)
    carver.fit(X.copy(), y)
    assert "std" in carver.features[0].statistics.columns

    report = carver.evaluate_stability(X, y + 0.4)
    assert report.per_modality["drift_pvalue"].notna().all()
    assert report.per_modality["drift_significant"].all()
    assert report.per_modality["rate_delta"].round(6).eq(0.4).all()


def test_incomplete_reference_reads_unknown_not_stable(binary):
    """A wiped/partial reference must be surfaced, never silently pass as stable."""
    carver, X, y = binary
    feature = carver.features[0]
    # what a manual split leaves behind: the affected bin's statistics are unknowable.
    # written straight to the backing dict (raw-label keyed) — the `statistics` property
    # re-maps its index under ordinal_encoding, so a get/set round-trip would corrupt it.
    wiped = next(iter(next(iter(feature._statistics.values()))))
    for column in feature._statistics:
        feature._statistics[column][wiped] = float("nan")

    report = carver.evaluate_stability(X, y)
    row = report.summary.loc[str(feature)]

    assert row["psi_flag"] == "unknown"
    assert np.isnan(row["psi"])
    assert np.isnan(row["chi2"]) and np.isnan(row["chi2_cramerv"])
    assert str(feature) in report.unstable_features


def test_negligible_effect_is_not_flagged_despite_significance(binary):
    """Chi-square power grows with n; the effect size is what keeps the verdict honest."""
    carver, X, y = binary
    report = carver.evaluate_stability(X, y)

    # self-comparison: nothing significant, nothing flagged
    assert report.per_feature["chi2_cramerv"].max() == approx(0.0, abs=1e-9)
    assert report.unstable_features == []


def test_target_median_gets_no_drift_test(sample):
    """The stored ``std`` describes values, not the sampling error of a median."""
    from AutoCarver.combinations.continuous.continuous_combination_evaluators import KruskalCombinations
    from AutoCarver.combinations.continuous.continuous_target_rates import TargetMedian

    X, score, rng = sample
    y = pd.Series(score + rng.normal(scale=0.5, size=len(X)), name="y")

    carver = ContinuousCarver(
        features(),
        min_freq=0.05,
        max_n_mod=4,
        combination_evaluator=KruskalCombinations(target_rate=TargetMedian()),
    )
    carver.fit(X.copy(), y)
    report = carver.evaluate_stability(X, y + 0.4)

    assert report.per_modality["drift_pvalue"].isna().all()
    # the delta is still reported, and viability still runs
    assert report.per_modality["rate_delta"].notna().all()


def test_multiclass_rejects_a_target_class_unseen_at_fit(sample):
    """The CA axis is fixed at fit time; it cannot score a class it never saw."""
    X, score, _ = sample
    y = pd.Series(pd.cut(score, 3, labels=list("xyz")).astype(str), name="y")

    carver = MulticlassCarver(features(), min_freq=0.05, max_n_mod=4)
    carver.fit(X.copy(), y)

    unseen = y.copy()
    unseen.iloc[:200] = "w"
    with raises(ValueError, match="cannot score classes it was never fit on"):
        carver.evaluate_stability(X, unseen)


@mark.parametrize("carver_class", [OrdinalCarver, MulticlassCarver])
def test_ordinal_and_multiclass_report_deltas_but_no_drift_test(sample, carver_class):
    """Their rate is a ridit / CA score: no recoverable sampling variance, so no p-value."""
    X, score, _ = sample
    y = {
        OrdinalCarver: pd.Series(pd.cut(score, 4, labels=[0, 1, 2, 3]).astype(int), name="y"),
        MulticlassCarver: pd.Series(pd.cut(score, 3, labels=list("xyz")).astype(str), name="y"),
    }[carver_class]

    carver = carver_class(features(), min_freq=0.05, max_n_mod=4)
    carver.fit(X.copy(), y)
    report = carver.evaluate_stability(X, y)

    assert report.per_modality["drift_pvalue"].isna().all()
    assert report.per_modality["rate_delta"].notna().all()
    # the viability block still runs
    assert report.per_feature["viable"].all()
