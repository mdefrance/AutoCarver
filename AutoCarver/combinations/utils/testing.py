""" set of tests for combinations within carvers"""

from numpy import isclose
from pandas import DataFrame, Series


def _test_distinct_target_rates_between_modalities(target_rates: Series) -> bool:
    """tests for distinct target rates between consecutive modalities"""

    # - target rates are distinct for consecutive modalities
    return not any(isclose(target_rates[1:], target_rates.shift(1)[1:]))


def _test_minimum_frequency_per_modality(frequencies: Series, min_freq: float) -> bool:
    """tests that minimum frequency has been reached for each modality"""
    # - minimum frequency is reached for all modalities
    return all(frequencies >= min_freq)


def _test_modality_ordering(train_target_rate: Series, dev_target_rate: Series) -> bool:
    """tests whether train and dev targets rates are similarly ranked between datasets"""
    # - grouped values have the same ranks in train/test
    return all(
        train_target_rate.sort_index().sort_values().index
        == dev_target_rate.sort_index().sort_values().index
    )


def is_viable(test_results: dict):
    """checks if combination is viable on train and dev (if provided)"""

    return test_results["train"]["viable"] and (
        test_results["dev"]["viable"] or test_results["dev"]["viable"] is None
    )


def _test_viability(rates: DataFrame, min_freq: float, train_target_rate: Series = None) -> dict:
    """tests viability of the rates"""

    # - minimum frequency is reached for all modalities
    min_freq = _test_minimum_frequency_per_modality(rates["frequency"], min_freq)

    # - target rates are distinct for all modalities
    distinct_rates = _test_distinct_target_rates_between_modalities(rates["target_rate"])

    # gathering results
    test_results = {
        "viable": min_freq and distinct_rates,
        "min_freq": min_freq,
        "distinct_rates": distinct_rates,
    }

    # adding ranking test if train_rates where provided
    if train_target_rate is not None:
        ordering = _test_modality_ordering(train_target_rate, rates["target_rate"])
        test_results.update(
            {"ranks_train_dev": ordering, "viable": test_results["viable"] and ordering}
        )

        # return tests on dev
        return {"dev": test_results}

    # return tests on train
    return {"train": test_results, "train_rates": rates}
