"""set of tests for viability testing within carvers"""

import pandas as pd
from pytest import raises

from AutoCarver.combinations.utils.testing import (
    Keys,
    Messages,
    _test_distinct_target_rates_between_modalities,
    _test_minimum_frequency_per_modality,
    _test_modality_ordering,
    add_info,
    is_viable,
)
from AutoCarver.combinations.utils.testing import test_viability as _test_viability


def test_distinct_target_rates_between_modalities_distinct():
    """Test distinct target rates between modalities when all rates are distinct."""
    target_rates = pd.Series([0.1, 0.2, 0.3, 0.4])
    result = _test_distinct_target_rates_between_modalities(target_rates)
    assert result is True


def test_distinct_target_rates_between_modalities_non_distinct():
    """Test distinct target rates between modalities when some rates are non-distinct."""
    target_rates = pd.Series([0.1, 0.1, 0.3, 0.4])
    result = _test_distinct_target_rates_between_modalities(target_rates)
    assert result is False


def test_distinct_target_rates_between_modalities_all_same():
    """Test distinct target rates between modalities when all rates are the same."""
    target_rates = pd.Series([0.1, 0.1, 0.1, 0.1])
    result = _test_distinct_target_rates_between_modalities(target_rates)
    assert result is False


def test_distinct_target_rates_between_modalities_single_element():
    """Test distinct target rates between modalities with a single element."""
    target_rates = pd.Series([0.1])
    result = _test_distinct_target_rates_between_modalities(target_rates)
    assert result is True


def test_minimum_frequency_all_above():
    """Test minimum frequency per modality when all frequencies are above the threshold."""
    counts = pd.Series([2000, 3000, 4000, 5000])
    nobs = int(counts.sum())
    result = _test_minimum_frequency_per_modality(counts, nobs, 0.1, alpha=0.05)
    assert result is True


def test_minimum_frequency_some_below():
    """A modality clearly below min_freq at large n fails the test."""
    counts = pd.Series([2000, 3000, 500, 5000])  # 500/10500 ≈ 4.8% vs min_freq=10%
    nobs = int(counts.sum())
    result = _test_minimum_frequency_per_modality(counts, nobs, 0.1, alpha=0.05)
    assert result is False


def test_minimum_frequency_all_below():
    """All modalities significantly below min_freq → test fails.

    With 4 modalities of ~25% each on n=22000, all are significantly below a
    35% target — the Wilson upper bound is comfortably under 0.35 for all.
    """
    counts = pd.Series([4000, 6500, 7500, 4000])
    nobs = int(counts.sum())
    result = _test_minimum_frequency_per_modality(counts, nobs, 0.35, alpha=0.05)
    assert result is False


def test_minimum_frequency_exactly_at_threshold():
    """Modalities exactly at min_freq pass — the CI upper bound exceeds min_freq."""
    counts = pd.Series([1000, 1000, 1000, 1000])
    nobs = int(counts.sum())
    result = _test_minimum_frequency_per_modality(counts, nobs, 0.25, alpha=0.05)
    assert result is True


def test_minimum_frequency_borderline_passes_on_small_n():
    """On a small sample, a 5% modality is not significantly below 10% under Wilson CI."""
    counts = pd.Series([5, 30, 40, 50])  # 5/125=4% vs min_freq=10%; Wilson upper ≈ 9.0% → close
    nobs = int(counts.sum())
    # The 5/125 modality is not clearly below 10% with this small a sample.
    # Wilson upper(5, 125, 0.05) ≈ 0.090 — still below 0.10, so test fails.
    # Use a tinier example where upper > min_freq.
    counts = pd.Series([1, 30, 40, 50])  # n=121, count=1, upper(1, 121, 0.05) ≈ 0.045
    nobs = int(counts.sum())
    # 1/121 ≈ 0.83%, upper ≈ 4.5% < 10% → still flagged. Use larger borderline:
    counts = pd.Series([5, 30, 40])  # n=75, 5/75≈6.7%, upper(5,75,0.05)≈0.149>0.10 → passes
    nobs = int(counts.sum())
    result = _test_minimum_frequency_per_modality(counts, nobs, 0.1, alpha=0.05)
    assert result is True


def test_minimum_frequency_empty_series():
    """Test minimum frequency per modality with an empty series."""
    counts = pd.Series([], dtype=float)
    result = _test_minimum_frequency_per_modality(counts, 0, 0.1, alpha=0.05)
    assert result is True


def test_modality_ordering_same_order():
    """Test modality ordering when train and dev target rates have the same order."""
    train_target_rate = pd.Series([0.1, 0.2, 0.3, 0.4], index=["a", "b", "c", "d"])
    dev_target_rate = pd.Series([0.1, 0.2, 0.3, 0.4], index=["a", "b", "c", "d"])
    result = _test_modality_ordering(train_target_rate, dev_target_rate)
    assert result is True


def test_modality_ordering_different_order_same_values():
    """Test modality ordering when train and dev target rates have different orders but same
    values.
    """
    train_target_rate = pd.Series([0.1, 0.2, 0.3, 0.4], index=["a", "b", "c", "d"])
    dev_target_rate = pd.Series([0.4, 0.3, 0.2, 0.1], index=["d", "c", "b", "a"])
    result = _test_modality_ordering(train_target_rate, dev_target_rate)
    assert result is True


def test_modality_ordering_different_values_same_order():
    """Test modality ordering when train and dev target rates have the same order but different
    values.
    """
    train_target_rate = pd.Series([0.1, 0.2, 0.3, 0.4], index=["a", "b", "c", "d"])
    dev_target_rate = pd.Series([0.4, 0.3, 0.2, 0.1], index=["a", "b", "c", "d"])
    result = _test_modality_ordering(train_target_rate, dev_target_rate)
    assert result is False


def test_modality_ordering_partial_overlap():
    """Test modality ordering when train and dev target rates have partial overlap."""
    train_target_rate = pd.Series([0.1, 0.2, 0.3], index=["a", "b", "c"])
    dev_target_rate = pd.Series([0.1, 0.2, 0.4], index=["a", "b", "d"])
    result = _test_modality_ordering(train_target_rate, dev_target_rate)
    assert result is False


def test_modality_ordering_empty_series():
    """Test modality ordering with empty series."""
    train_target_rate = pd.Series([], dtype=float)
    dev_target_rate = pd.Series([], dtype=float)
    result = _test_modality_ordering(train_target_rate, dev_target_rate)
    assert result is True


def test_add_info_all_passed():
    """Test add_info when all tests are passed."""
    test_results = {
        Keys.RANKS_TRAIN_DEV.value: True,
        Keys.MIN_FREQ.value: True,
        Keys.DISTINCT_RATES.value: True,
        Keys.VIABLE.value: True,
    }
    min_freq = 0.05
    result = add_info(test_results, min_freq)
    expected = {
        Keys.VIABLE.value: True,
        Keys.INFO.value: Messages.PASSED_TESTS.value,
    }
    assert result == expected


def test_add_info_inversion_rates():
    """Test add_info when inversion rates test fails."""
    test_results = {
        Keys.RANKS_TRAIN_DEV.value: False,
        Keys.MIN_FREQ.value: True,
        Keys.DISTINCT_RATES.value: True,
        Keys.VIABLE.value: True,
    }
    min_freq = 0.05
    result = add_info(test_results, min_freq)
    expected = {
        Keys.VIABLE.value: True,
        Keys.INFO.value: Messages.INVERSION_RATES.value,
    }
    assert result == expected


def test_add_info_non_representative():
    """Test add_info when non-representative test fails."""
    test_results = {
        Keys.RANKS_TRAIN_DEV.value: True,
        Keys.MIN_FREQ.value: False,
        Keys.DISTINCT_RATES.value: True,
        Keys.VIABLE.value: True,
    }
    min_freq = 0.05
    result = add_info(test_results, min_freq)
    expected = {
        Keys.VIABLE.value: True,
        Keys.INFO.value: Messages.NON_REPRESENTATIVE.value.format(min_freq=min_freq),
    }
    assert result == expected


def test_add_info_non_distinct_rates():
    """Test add_info when non-distinct rates test fails."""
    test_results = {
        Keys.RANKS_TRAIN_DEV.value: True,
        Keys.MIN_FREQ.value: True,
        Keys.DISTINCT_RATES.value: False,
        Keys.VIABLE.value: True,
    }
    min_freq = 0.05
    result = add_info(test_results, min_freq)
    expected = {
        Keys.VIABLE.value: True,
        Keys.INFO.value: Messages.NON_DISTINCT_RATES.value,
    }
    assert result == expected


def test_add_info_multiple_failures():
    """Test add_info when multiple tests fail."""
    test_results = {
        Keys.RANKS_TRAIN_DEV.value: False,
        Keys.MIN_FREQ.value: False,
        Keys.DISTINCT_RATES.value: False,
        Keys.VIABLE.value: True,
    }
    min_freq = 0.05
    result = add_info(test_results, min_freq)
    expected = {
        Keys.VIABLE.value: True,
        Keys.INFO.value: "; ".join(
            [
                Messages.INVERSION_RATES.value,
                Messages.NON_REPRESENTATIVE.value.format(min_freq=min_freq),
                Messages.NON_DISTINCT_RATES.value,
            ]
        ),
    }
    assert result == expected


def _rates(frequencies: list[float], target: list[float], nobs: int = 10000) -> pd.DataFrame:
    """Builds a rates DataFrame with counts derived from frequencies × nobs."""
    return pd.DataFrame(
        {
            "frequency": frequencies,
            "target_mean": target,
            "count": [round(f * nobs) for f in frequencies],
        }
    )


def test_viability_min_freq_and_distinct_rates():
    """Test viability when minimum frequency and distinct rates are met."""
    rates = _rates([0.2, 0.3, 0.4], [0.1, 0.2, 0.3])
    result = _test_viability(rates, 0.1, "target_mean", alpha=0.05)
    assert result["train"][Keys.VIABLE.value] is True
    assert result["train_rates"].equals(rates)


def test_viability_min_freq_not_met():
    """At large n, a 5%-modality is significantly below 10% → not viable."""
    rates = _rates([0.05, 0.3, 0.4], [0.1, 0.2, 0.3])
    result = _test_viability(rates, 0.1, "target_mean", alpha=0.05)
    assert result["train"][Keys.VIABLE.value] is False
    assert result["train_rates"].equals(rates)


def test_viability_distinct_rates_not_met():
    """Test viability when distinct rates are not met."""
    rates = _rates([0.2, 0.3, 0.4], [0.1, 0.1, 0.3])
    result = _test_viability(rates, 0.1, "target_mean", alpha=0.05)
    assert result["train"][Keys.VIABLE.value] is False
    assert result["train_rates"].equals(rates)


def test_viability_with_train_target_rate():
    """Test viability with train target rate provided."""
    rates = _rates([0.2, 0.3, 0.4], [0.1, 0.2, 0.3])
    train_target_rate = pd.Series([0.1, 0.2, 0.3])
    result = _test_viability(rates, 0.1, "target_mean", alpha=0.05, train_target_rate=train_target_rate)
    assert result["dev"][Keys.VIABLE.value] is True


def test_viability_with_train_target_rate_ordering_not_met():
    """Test viability when train target rate ordering is not met."""
    rates = _rates([0.2, 0.3, 0.4], [0.1, 0.2, 0.3])
    train_target_rate = pd.Series([0.3, 0.2, 0.1])
    result = _test_viability(rates, 0.1, "target_mean", alpha=0.05, train_target_rate=train_target_rate)
    assert result["dev"][Keys.VIABLE.value] is False


def test_viability_empty_rates():
    """Test viability with empty rates."""
    rates = pd.DataFrame({"frequency": [], "target_mean": [], "count": []})
    result = _test_viability(rates, 0.1, "target_mean", alpha=0.05)
    expected = {
        "train": {"viable": True, Keys.INFO.value: Messages.PASSED_TESTS.value},
        "train_rates": rates,
    }
    assert result["train"] == expected["train"]
    assert result["train_rates"].equals(expected["train_rates"])


def test_is_viable_train_and_dev_viable():
    """Test is_viable when both train and dev are viable."""
    test_results = {"train": {"viable": True}, "dev": {"viable": True}}
    result = is_viable(test_results)
    assert result is True


def test_is_viable_train_viable_dev_not_viable():
    """Test is_viable when train is viable but dev is not viable."""
    test_results = {"train": {"viable": True}, "dev": {"viable": False}}
    result = is_viable(test_results)
    assert result is False


def test_is_viable_train_not_viable_dev_viable():
    """Test is_viable when train is not viable but dev is viable."""
    test_results = {"train": {"viable": False}, "dev": {"viable": True}}
    result = is_viable(test_results)
    assert result is False


def test_is_viable_train_viable_dev_none():
    """Test is_viable when train is viable and dev is None."""
    test_results = {"train": {"viable": True}, "dev": {"viable": None}}
    result = is_viable(test_results)
    assert result is True


def test_is_viable_train_not_viable_dev_none():
    """Test is_viable when train is not viable and dev is None."""
    test_results = {"train": {"viable": False}, "dev": {"viable": None}}
    result = is_viable(test_results)
    assert result is False


def test_is_viable_train_viable_no_dev():
    """Test is_viable when train is viable and dev key is missing."""
    test_results = {"train": {"viable": True}}
    with raises(KeyError):
        is_viable(test_results)


def test_is_viable_train_not_viable_no_dev():
    """Test is_viable when train is not viable and dev key is missing."""
    test_results = {"train": {"viable": False}}
    result = is_viable(test_results)
    assert result is False
