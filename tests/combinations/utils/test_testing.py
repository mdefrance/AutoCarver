""" set of tests for viability testing within carvers"""

from pandas import DataFrame, Series
from pytest import raises

from AutoCarver.combinations.utils.testing import (
    TestKeys,
    TestMessages,
    _test_distinct_target_rates_between_modalities,
    _test_minimum_frequency_per_modality,
    _test_modality_ordering,
    add_info,
    is_viable,
)
from AutoCarver.combinations.utils.testing import test_viability as _test_viability


def test_distinct_target_rates_between_modalities_distinct():
    """Test distinct target rates between modalities when all rates are distinct."""
    target_rates = Series([0.1, 0.2, 0.3, 0.4])
    result = _test_distinct_target_rates_between_modalities(target_rates)
    assert result is True


def test_distinct_target_rates_between_modalities_non_distinct():
    """Test distinct target rates between modalities when some rates are non-distinct."""
    target_rates = Series([0.1, 0.1, 0.3, 0.4])
    result = _test_distinct_target_rates_between_modalities(target_rates)
    assert result is False


def test_distinct_target_rates_between_modalities_all_same():
    """Test distinct target rates between modalities when all rates are the same."""
    target_rates = Series([0.1, 0.1, 0.1, 0.1])
    result = _test_distinct_target_rates_between_modalities(target_rates)
    assert result is False


def test_distinct_target_rates_between_modalities_single_element():
    """Test distinct target rates between modalities with a single element."""
    target_rates = Series([0.1])
    result = _test_distinct_target_rates_between_modalities(target_rates)
    assert result is True


def test_minimum_frequency_all_above():
    """Test minimum frequency per modality when all frequencies are above the threshold."""
    frequencies = Series([0.2, 0.3, 0.4, 0.5])
    min_freq = 0.1
    result = _test_minimum_frequency_per_modality(frequencies, min_freq)
    assert result is True


def test_minimum_frequency_some_below():
    """Test minimum frequency per modality when some frequencies are below the threshold."""
    frequencies = Series([0.2, 0.3, 0.05, 0.5])
    min_freq = 0.1
    result = _test_minimum_frequency_per_modality(frequencies, min_freq)
    assert result is False


def test_minimum_frequency_all_below():
    """Test minimum frequency per modality when all frequencies are below the threshold."""
    frequencies = Series([0.05, 0.08, 0.09, 0.05])
    min_freq = 0.1
    result = _test_minimum_frequency_per_modality(frequencies, min_freq)
    assert result is False


def test_minimum_frequency_exactly_at_threshold():
    """Test minimum frequency per modality when all frequencies are exactly at the threshold."""
    frequencies = Series([0.1, 0.1, 0.1, 0.1])
    min_freq = 0.1
    result = _test_minimum_frequency_per_modality(frequencies, min_freq)
    assert result is True


def test_minimum_frequency_empty_series():
    """Test minimum frequency per modality with an empty series."""
    frequencies = Series([])
    min_freq = 0.1
    result = _test_minimum_frequency_per_modality(frequencies, min_freq)
    assert result is True


def test_modality_ordering_same_order():
    """Test modality ordering when train and dev target rates have the same order."""
    train_target_rate = Series([0.1, 0.2, 0.3, 0.4], index=["a", "b", "c", "d"])
    dev_target_rate = Series([0.1, 0.2, 0.3, 0.4], index=["a", "b", "c", "d"])
    result = _test_modality_ordering(train_target_rate, dev_target_rate)
    assert result is True


def test_modality_ordering_different_order_same_values():
    """Test modality ordering when train and dev target rates have different orders but same
    values.
    """
    train_target_rate = Series([0.1, 0.2, 0.3, 0.4], index=["a", "b", "c", "d"])
    dev_target_rate = Series([0.4, 0.3, 0.2, 0.1], index=["d", "c", "b", "a"])
    result = _test_modality_ordering(train_target_rate, dev_target_rate)
    assert result is True


def test_modality_ordering_different_values_same_order():
    """Test modality ordering when train and dev target rates have the same order but different
    values.
    """
    train_target_rate = Series([0.1, 0.2, 0.3, 0.4], index=["a", "b", "c", "d"])
    dev_target_rate = Series([0.4, 0.3, 0.2, 0.1], index=["a", "b", "c", "d"])
    result = _test_modality_ordering(train_target_rate, dev_target_rate)
    assert result is False


def test_modality_ordering_partial_overlap():
    """Test modality ordering when train and dev target rates have partial overlap."""
    train_target_rate = Series([0.1, 0.2, 0.3], index=["a", "b", "c"])
    dev_target_rate = Series([0.1, 0.2, 0.4], index=["a", "b", "d"])
    result = _test_modality_ordering(train_target_rate, dev_target_rate)
    assert result is False


def test_modality_ordering_empty_series():
    """Test modality ordering with empty series."""
    train_target_rate = Series([], dtype=float)
    dev_target_rate = Series([], dtype=float)
    result = _test_modality_ordering(train_target_rate, dev_target_rate)
    assert result is True


def test_add_info_all_passed():
    """Test add_info when all tests are passed."""
    test_results = {
        TestKeys.RANKS_TRAIN_DEV.value: True,
        TestKeys.MIN_FREQ.value: True,
        TestKeys.DISTINCT_RATES.value: True,
        TestKeys.VIABLE.value: True,
    }
    min_freq = 0.05
    result = add_info(test_results, min_freq)
    expected = {
        TestKeys.VIABLE.value: True,
        TestKeys.INFO.value: TestMessages.PASSED_TESTS.value,
    }
    assert result == expected


def test_add_info_inversion_rates():
    """Test add_info when inversion rates test fails."""
    test_results = {
        TestKeys.RANKS_TRAIN_DEV.value: False,
        TestKeys.MIN_FREQ.value: True,
        TestKeys.DISTINCT_RATES.value: True,
        TestKeys.VIABLE.value: True,
    }
    min_freq = 0.05
    result = add_info(test_results, min_freq)
    expected = {
        TestKeys.VIABLE.value: True,
        TestKeys.INFO.value: TestMessages.INVERSION_RATES.value,
    }
    assert result == expected


def test_add_info_non_representative():
    """Test add_info when non-representative test fails."""
    test_results = {
        TestKeys.RANKS_TRAIN_DEV.value: True,
        TestKeys.MIN_FREQ.value: False,
        TestKeys.DISTINCT_RATES.value: True,
        TestKeys.VIABLE.value: True,
    }
    min_freq = 0.05
    result = add_info(test_results, min_freq)
    expected = {
        TestKeys.VIABLE.value: True,
        TestKeys.INFO.value: TestMessages.NON_REPRESENTATIVE.value.format(min_freq=min_freq),
    }
    assert result == expected


def test_add_info_non_distinct_rates():
    """Test add_info when non-distinct rates test fails."""
    test_results = {
        TestKeys.RANKS_TRAIN_DEV.value: True,
        TestKeys.MIN_FREQ.value: True,
        TestKeys.DISTINCT_RATES.value: False,
        TestKeys.VIABLE.value: True,
    }
    min_freq = 0.05
    result = add_info(test_results, min_freq)
    expected = {
        TestKeys.VIABLE.value: True,
        TestKeys.INFO.value: TestMessages.NON_DISTINCT_RATES.value,
    }
    assert result == expected


def test_add_info_multiple_failures():
    """Test add_info when multiple tests fail."""
    test_results = {
        TestKeys.RANKS_TRAIN_DEV.value: False,
        TestKeys.MIN_FREQ.value: False,
        TestKeys.DISTINCT_RATES.value: False,
        TestKeys.VIABLE.value: True,
    }
    min_freq = 0.05
    result = add_info(test_results, min_freq)
    expected = {
        TestKeys.VIABLE.value: True,
        TestKeys.INFO.value: "; ".join(
            [
                TestMessages.INVERSION_RATES.value,
                TestMessages.NON_REPRESENTATIVE.value.format(min_freq=min_freq),
                TestMessages.NON_DISTINCT_RATES.value,
            ]
        ),
    }
    assert result == expected


def test_viability_min_freq_and_distinct_rates():
    """Test viability when minimum frequency and distinct rates are met."""
    rates = DataFrame({"frequency": [0.2, 0.3, 0.4], "target_rate": [0.1, 0.2, 0.3]})
    min_freq = 0.1
    result = _test_viability(rates, min_freq)
    assert result["train"][TestKeys.VIABLE.value] is True
    assert result["train_rates"].equals(rates)


def test_viability_min_freq_not_met():
    """Test viability when minimum frequency is not met."""
    rates = DataFrame({"frequency": [0.05, 0.3, 0.4], "target_rate": [0.1, 0.2, 0.3]})
    min_freq = 0.1
    result = _test_viability(rates, min_freq)
    assert result["train"][TestKeys.VIABLE.value] is False
    assert result["train_rates"].equals(rates)


def test_viability_distinct_rates_not_met():
    """Test viability when distinct rates are not met."""
    rates = DataFrame({"frequency": [0.2, 0.3, 0.4], "target_rate": [0.1, 0.1, 0.3]})
    min_freq = 0.1
    result = _test_viability(rates, min_freq)
    assert result["train"][TestKeys.VIABLE.value] is False
    assert result["train_rates"].equals(rates)


def test_viability_with_train_target_rate():
    """Test viability with train target rate provided."""
    rates = DataFrame({"frequency": [0.2, 0.3, 0.4], "target_rate": [0.1, 0.2, 0.3]})
    train_target_rate = Series([0.1, 0.2, 0.3])
    min_freq = 0.1
    result = _test_viability(rates, min_freq, train_target_rate)
    assert result["dev"][TestKeys.VIABLE.value] is True


def test_viability_with_train_target_rate_ordering_not_met():
    """Test viability when train target rate ordering is not met."""
    rates = DataFrame({"frequency": [0.2, 0.3, 0.4], "target_rate": [0.1, 0.2, 0.3]})
    train_target_rate = Series([0.3, 0.2, 0.1])
    min_freq = 0.1
    result = _test_viability(rates, min_freq, train_target_rate)
    assert result["dev"][TestKeys.VIABLE.value] is False


def test_viability_empty_rates():
    """Test viability with empty rates."""
    rates = DataFrame({"frequency": [], "target_rate": []})
    min_freq = 0.1
    result = _test_viability(rates, min_freq)
    expected = {
        "train": {"viable": True, TestKeys.INFO.value: TestMessages.PASSED_TESTS.value},
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
