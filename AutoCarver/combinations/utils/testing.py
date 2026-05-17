"""set of tests for combinations within carvers"""

from enum import StrEnum

import numpy as np
import pandas as pd


class Keys(StrEnum):
    """keys for test results"""

    RANKS_TRAIN_DEV = "ranks_train_dev"
    MIN_FREQ = "min_freq"
    DISTINCT_RATES = "distinct_rates"
    VIABLE = "viable"
    INFO = "info"


class Messages(StrEnum):
    """messages for test results"""

    INVERSION_RATES = "Inversion of target rates per modality"
    NON_REPRESENTATIVE = "Non-representative modality for min_freq={min_freq:2.2%}"
    NON_DISTINCT_RATES = "Non-distinct target rates per consecutive modalities"
    PASSED_TESTS = ""


def _test_distinct_target_rates_between_modalities(target_rates: pd.Series) -> bool:
    """tests for distinct target rates between consecutive modalities"""

    # - target rates are distinct for consecutive modalities
    return not any(np.isclose(target_rates[1:], target_rates.shift(1)[1:]))


def _test_minimum_frequency_per_modality(frequencies: pd.Series, min_freq: float) -> bool:
    """tests that minimum frequency has been reached for each modality"""
    # - minimum frequency is reached for all modalities
    return all(frequencies >= min_freq)


def _test_modality_ordering(train_target_rate: pd.Series, dev_target_rate: pd.Series) -> bool:
    """tests whether train and dev targets rates are similarly ranked between datasets"""
    # - grouped values have the same ranks in train/test
    return all(train_target_rate.sort_index().sort_values().index == dev_target_rate.sort_index().sort_values().index)


def is_viable(test_results: dict):
    """checks if combination is viable on train and dev (if provided)"""

    return test_results["train"][Keys.VIABLE.value] and (
        test_results["dev"][Keys.VIABLE.value] or test_results["dev"][Keys.VIABLE.value] is None
    )


def test_viability(
    rates: pd.DataFrame, min_freq: float, target_rate: str, train_target_rate: pd.Series | None = None
) -> dict:
    """tests viability of the rates"""

    # - minimum frequency is reached for all modalities
    min_freq_test = _test_minimum_frequency_per_modality(rates["frequency"], min_freq)

    # - target rates are distinct for all modalities
    distinct_rates = _test_distinct_target_rates_between_modalities(rates[target_rate])

    # gathering results
    test_results = {
        Keys.VIABLE.value: min_freq_test and distinct_rates,
        Keys.MIN_FREQ.value: min_freq_test,
        Keys.DISTINCT_RATES.value: distinct_rates,
    }

    # adding ranking test if train_rates where provided
    if train_target_rate is not None:
        ordering = _test_modality_ordering(train_target_rate, rates[target_rate])
        test_results.update(
            {
                Keys.RANKS_TRAIN_DEV.value: ordering,
                Keys.VIABLE.value: test_results[Keys.VIABLE.value] and ordering,
            }
        )

        # return tests on dev
        return {"dev": add_info(test_results, min_freq)}

    # return tests on train
    return {
        "train": add_info(test_results, min_freq),
        "train_rates": rates,
        Keys.VIABLE.value: test_results[Keys.VIABLE.value],
    }


def add_info(test_results: dict[str, bool], min_freq: float) -> dict[str, str | bool]:
    """Adds information to test results."""

    messages: list[str] = []
    if not test_results.get(Keys.RANKS_TRAIN_DEV.value, True):
        messages.append(Messages.INVERSION_RATES.value)
    if not test_results.get(Keys.MIN_FREQ.value, True):
        messages.append(Messages.NON_REPRESENTATIVE.value.format(min_freq=min_freq))
    if not test_results.get(Keys.DISTINCT_RATES.value, True):
        messages.append(Messages.NON_DISTINCT_RATES.value)

    if not messages:
        messages.append(Messages.PASSED_TESTS.value)

    return {
        Keys.VIABLE.value: test_results[Keys.VIABLE.value],
        Keys.INFO.value: "; ".join(messages),
    }
