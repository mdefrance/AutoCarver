"""Tests for the stability metrics (PSI, chi-square GOF, drift tests)."""

from math import log, nan

import numpy as np
import pandas as pd
from pytest import approx, raises

from AutoCarver.stability.metrics import (
    chi2_homogeneity,
    population_stability_index,
    to_probability,
    two_proportion_test,
    welch_test,
)


def test_psi_of_identical_distributions_is_zero():
    frequency = pd.Series([0.5, 0.3, 0.2], index=["a", "b", "c"])
    psi, contributions = population_stability_index(frequency, frequency)
    assert psi == approx(0.0, abs=1e-12)
    assert contributions.abs().max() == approx(0.0, abs=1e-12)


def test_psi_is_positive_for_a_shift():
    reference = pd.Series([0.5, 0.3, 0.2], index=["a", "b", "c"])
    new = pd.Series([0.2, 0.3, 0.5], index=["a", "b", "c"])
    psi, contributions = population_stability_index(reference, new)
    assert psi > 0.25
    # the untouched modality contributes nothing; the two swapped ones carry it all
    assert contributions["b"] == approx(0.0, abs=1e-12)
    assert psi == approx(contributions.sum())


def test_psi_stays_finite_when_a_modality_vanishes():
    reference = pd.Series([0.5, 0.3, 0.2], index=["a", "b", "c"])
    new = pd.Series([0.6, 0.4, 0.0], index=["a", "b", "c"])
    psi, contributions = population_stability_index(reference, new)
    assert np.isfinite(psi)
    assert np.isfinite(contributions).all()
    assert contributions["c"] > 1


def test_psi_is_undefined_when_the_reference_is_incomplete():
    """A partial reference must not yield a plausible-looking index on a reduced support.

    Manual splits leave the affected bins' statistics unknowable (NaN); silently
    dropping them would renormalize the comparison onto a different bin set.
    """
    reference = pd.Series([0.5, nan, 0.2], index=["a", "b", "c"])
    new = pd.Series([0.1, 0.7, 0.2], index=["a", "b", "c"])
    psi, contributions = population_stability_index(reference, new)
    assert np.isnan(psi)
    assert contributions.isna().all()


def test_psi_of_an_all_nan_reference_is_not_zero():
    """The dangerous case: a wiped reference must never read as perfectly stable."""
    reference = pd.Series([nan, nan, nan], index=["a", "b", "c"])
    new = pd.Series([0.1, 0.7, 0.2], index=["a", "b", "c"])
    psi, _ = population_stability_index(reference, new)
    assert np.isnan(psi)


def test_psi_reindexes_a_new_sample_missing_modalities():
    reference = pd.Series([0.5, 0.3, 0.2], index=["a", "b", "c"])
    new = pd.Series([0.6, 0.4], index=["a", "b"])
    psi, contributions = population_stability_index(reference, new)
    assert list(contributions.index) == ["a", "b", "c"]
    assert np.isfinite(psi)


def test_chi2_homogeneity_matches_the_reference():
    reference = pd.Series([500, 300, 200], index=["a", "b", "c"])
    statistic, pvalue, dof, cramerv = chi2_homogeneity(reference, reference)
    assert statistic == approx(0.0)
    assert pvalue == approx(1.0)
    assert dof == 2
    assert cramerv == approx(0.0)


def test_chi2_homogeneity_detects_a_shift():
    reference = pd.Series([500, 300, 200], index=["a", "b", "c"])
    counts = pd.Series([200, 300, 500], index=["a", "b", "c"])
    _, pvalue, _, cramerv = chi2_homogeneity(reference, counts)
    assert pvalue < 0.01
    assert cramerv > 0.1


def test_chi2_homogeneity_uses_both_margins_not_fixed_expectations():
    """A two-sample test, so a *small* new sample matching the reference stays non-significant.

    A goodness-of-fit against fixed frequencies would ignore the reference's own
    sampling error and be anti-conservative here.
    """
    reference = pd.Series([500, 300, 200], index=["a", "b", "c"])
    counts = pd.Series([5, 3, 2], index=["a", "b", "c"])
    _, pvalue, _, _ = chi2_homogeneity(reference, counts)
    assert pvalue > 0.5


def test_chi2_homogeneity_cramerv_is_sample_size_neutral():
    """Same proportional shift, 100x the data: p collapses, the effect size does not."""
    small_ref = pd.Series([50, 30, 20], index=["a", "b", "c"])
    small_new = pd.Series([40, 30, 30], index=["a", "b", "c"])
    big_ref, big_new = small_ref * 100, small_new * 100

    _, small_p, _, small_v = chi2_homogeneity(small_ref, small_new)
    _, big_p, _, big_v = chi2_homogeneity(big_ref, big_new)

    assert small_p > 0.05 and big_p < 1e-10
    assert small_v == approx(big_v, rel=1e-6)


def test_chi2_homogeneity_needs_two_informative_modalities():
    reference = pd.Series([100, 0], index=["a", "b"])
    counts = pd.Series([100, 0], index=["a", "b"])
    statistic, pvalue, dof, cramerv = chi2_homogeneity(reference, counts)
    assert np.isnan(statistic) and np.isnan(pvalue) and np.isnan(cramerv)
    assert dof == 0


def test_chi2_homogeneity_refuses_an_incomplete_reference():
    reference = pd.Series([500, nan, 200], index=["a", "b", "c"])
    counts = pd.Series([500, 300, 200], index=["a", "b", "c"])
    statistic, pvalue, dof, cramerv = chi2_homogeneity(reference, counts)
    assert np.isnan(statistic) and np.isnan(pvalue) and np.isnan(cramerv)
    assert dof == 0


def test_two_proportion_test_flags_a_real_rate_move():
    index = ["a", "b"]
    pvalues = two_proportion_test(
        pd.Series([0.20, 0.50], index=index),
        pd.Series([1000, 1000], index=index),
        pd.Series([0.35, 0.50], index=index),
        pd.Series([1000, 1000], index=index),
    )
    assert pvalues["a"] < 0.001
    assert pvalues["b"] == approx(1.0)


def test_two_proportion_test_returns_nan_on_empty_modalities():
    index = ["a"]
    pvalues = two_proportion_test(
        pd.Series([0.2], index=index),
        pd.Series([0], index=index),
        pd.Series([0.2], index=index),
        pd.Series([0], index=index),
    )
    assert np.isnan(pvalues["a"])


def test_welch_test_flags_a_mean_shift_and_tolerates_missing_std():
    index = ["a", "b"]
    pvalues = welch_test(
        pd.Series([0.0, 0.0], index=index),
        pd.Series([1.0, nan], index=index),
        pd.Series([500, 500], index=index),
        pd.Series([0.5, 0.5], index=index),
        pd.Series([1.0, nan], index=index),
        pd.Series([500, 500], index=index),
    )
    assert pvalues["a"] < 0.001
    # a NaN reference std (singleton modality, or a carver fitted before std was stored)
    assert np.isnan(pvalues["b"])


def test_to_probability_inverts_every_binary_rate():
    probability = pd.Series([0.1, 0.4, 0.75], index=list("abc"))
    odds = probability / (1 - probability)
    woe = odds.map(log)

    assert np.allclose(to_probability("target_mean", probability), probability)
    assert np.allclose(to_probability("odds_ratio", odds), probability)
    assert np.allclose(to_probability("woe", woe), probability)


def test_to_probability_rejects_a_non_invertible_rate():
    with raises(ValueError, match="not an invertible binary target rate"):
        to_probability("target_mean_ridit", pd.Series([0.5]))
