""" set of tests for viability testing within carvers"""

from pandas import Series, DataFrame
from AutoCarver.carvers.utils.testing import (
    _test_distinct_target_rates_between_modalities,
    _test_minimum_frequency_per_modality,
    _test_modality_ordering,
    _test_viability,
    is_viable,
)
from pytest import raises


def test_distinct_target_rates_between_modalities_distinct():
    target_rates = Series([0.1, 0.2, 0.3, 0.4])
    result = _test_distinct_target_rates_between_modalities(target_rates)
    assert result is True


def test_distinct_target_rates_between_modalities_non_distinct():
    target_rates = Series([0.1, 0.1, 0.3, 0.4])
    result = _test_distinct_target_rates_between_modalities(target_rates)
    assert result is False


def test_distinct_target_rates_between_modalities_all_same():
    target_rates = Series([0.1, 0.1, 0.1, 0.1])
    result = _test_distinct_target_rates_between_modalities(target_rates)
    assert result is False


def test_distinct_target_rates_between_modalities_single_element():
    target_rates = Series([0.1])
    result = _test_distinct_target_rates_between_modalities(target_rates)
    assert result is True


def test_minimum_frequency_all_above():
    frequencies = Series([0.2, 0.3, 0.4, 0.5])
    min_freq = 0.1
    result = _test_minimum_frequency_per_modality(frequencies, min_freq)
    assert result is True


def test_minimum_frequency_some_below():
    frequencies = Series([0.2, 0.3, 0.05, 0.5])
    min_freq = 0.1
    result = _test_minimum_frequency_per_modality(frequencies, min_freq)
    assert result is False


def test_minimum_frequency_all_below():
    frequencies = Series([0.05, 0.08, 0.09, 0.05])
    min_freq = 0.1
    result = _test_minimum_frequency_per_modality(frequencies, min_freq)
    assert result is False


def test_minimum_frequency_exactly_at_threshold():
    frequencies = Series([0.1, 0.1, 0.1, 0.1])
    min_freq = 0.1
    result = _test_minimum_frequency_per_modality(frequencies, min_freq)
    assert result is True


def test_minimum_frequency_empty_series():
    frequencies = Series([])
    min_freq = 0.1
    result = _test_minimum_frequency_per_modality(frequencies, min_freq)
    assert result is True


def test_modality_ordering_same_order():
    train_target_rate = Series([0.1, 0.2, 0.3, 0.4], index=["a", "b", "c", "d"])
    dev_target_rate = Series([0.1, 0.2, 0.3, 0.4], index=["a", "b", "c", "d"])
    result = _test_modality_ordering(train_target_rate, dev_target_rate)
    assert result is True


def test_modality_ordering_different_order_same_values():
    train_target_rate = Series([0.1, 0.2, 0.3, 0.4], index=["a", "b", "c", "d"])
    dev_target_rate = Series([0.4, 0.3, 0.2, 0.1], index=["d", "c", "b", "a"])
    result = _test_modality_ordering(train_target_rate, dev_target_rate)
    assert result is True


def test_modality_ordering_different_values_same_order():
    train_target_rate = Series([0.1, 0.2, 0.3, 0.4], index=["a", "b", "c", "d"])
    dev_target_rate = Series([0.4, 0.3, 0.2, 0.1], index=["a", "b", "c", "d"])
    result = _test_modality_ordering(train_target_rate, dev_target_rate)
    assert result is False


def test_modality_ordering_partial_overlap():
    train_target_rate = Series([0.1, 0.2, 0.3], index=["a", "b", "c"])
    dev_target_rate = Series([0.1, 0.2, 0.4], index=["a", "b", "d"])
    result = _test_modality_ordering(train_target_rate, dev_target_rate)
    assert result is False


def test_modality_ordering_empty_series():
    train_target_rate = Series([], dtype=float)
    dev_target_rate = Series([], dtype=float)
    result = _test_modality_ordering(train_target_rate, dev_target_rate)
    assert result is True


def test_viability_min_freq_and_distinct_rates():
    rates = DataFrame({"frequency": [0.2, 0.3, 0.4], "target_rate": [0.1, 0.2, 0.3]})
    min_freq = 0.1
    result = _test_viability(rates, min_freq)
    expected = {
        "train": {"viable": True, "min_freq": True, "distinct_rates": True},
        "train_rates": rates,
    }
    assert result == expected


def test_viability_min_freq_not_met():
    rates = DataFrame({"frequency": [0.05, 0.3, 0.4], "target_rate": [0.1, 0.2, 0.3]})
    min_freq = 0.1
    result = _test_viability(rates, min_freq)
    expected = {
        "train": {"viable": False, "min_freq": False, "distinct_rates": True},
        "train_rates": rates,
    }
    assert result == expected


def test_viability_distinct_rates_not_met():
    rates = DataFrame({"frequency": [0.2, 0.3, 0.4], "target_rate": [0.1, 0.1, 0.3]})
    min_freq = 0.1
    result = _test_viability(rates, min_freq)
    expected = {
        "train": {"viable": False, "min_freq": True, "distinct_rates": False},
        "train_rates": rates,
    }
    assert result == expected


def test_viability_with_train_target_rate():
    rates = DataFrame({"frequency": [0.2, 0.3, 0.4], "target_rate": [0.1, 0.2, 0.3]})
    train_target_rate = Series([0.1, 0.2, 0.3])
    min_freq = 0.1
    result = _test_viability(rates, min_freq, train_target_rate)
    expected = {
        "dev": {"viable": True, "min_freq": True, "distinct_rates": True, "ranks_train_dev": True}
    }
    assert result == expected


def test_viability_with_train_target_rate_ordering_not_met():
    rates = DataFrame({"frequency": [0.2, 0.3, 0.4], "target_rate": [0.1, 0.2, 0.3]})
    train_target_rate = Series([0.3, 0.2, 0.1])
    min_freq = 0.1
    result = _test_viability(rates, min_freq, train_target_rate)
    expected = {
        "dev": {"viable": False, "min_freq": True, "distinct_rates": True, "ranks_train_dev": False}
    }
    assert result == expected


def test_viability_empty_rates():
    rates = DataFrame({"frequency": [], "target_rate": []})
    min_freq = 0.1
    result = _test_viability(rates, min_freq)
    expected = {
        "train": {"viable": True, "min_freq": True, "distinct_rates": True},
        "train_rates": rates,
    }
    assert result == expected


def test_is_viable_train_and_dev_viable():
    test_results = {"train": {"viable": True}, "dev": {"viable": True}}
    result = is_viable(test_results)
    assert result is True


def test_is_viable_train_viable_dev_not_viable():
    test_results = {"train": {"viable": True}, "dev": {"viable": False}}
    result = is_viable(test_results)
    assert result is False


def test_is_viable_train_not_viable_dev_viable():
    test_results = {"train": {"viable": False}, "dev": {"viable": True}}
    result = is_viable(test_results)
    assert result is False


def test_is_viable_train_viable_dev_none():
    test_results = {"train": {"viable": True}, "dev": {"viable": None}}
    result = is_viable(test_results)
    assert result is True


def test_is_viable_train_not_viable_dev_none():
    test_results = {"train": {"viable": False}, "dev": {"viable": None}}
    result = is_viable(test_results)
    assert result is False


def test_is_viable_train_viable_no_dev():
    test_results = {"train": {"viable": True}}
    with raises(KeyError):
        is_viable(test_results)


def test_is_viable_train_not_viable_no_dev():
    test_results = {"train": {"viable": False}}
    result = is_viable(test_results)
    assert result is False
