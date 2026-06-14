"""set of tests for combinations within carvers"""

from enum import StrEnum

import numpy as np
import pandas as pd

from AutoCarver.discretizers.utils.frequency_ci import is_significantly_below


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


def _test_minimum_frequency_per_modality(
    counts: pd.Series, nobs: int, min_freq: int | float | None, alpha: float
) -> bool:
    """tests that no modality is significantly below ``min_freq`` (Wilson CI).

    A modality fails the test only when the Wilson upper bound of its observed
    proportion is strictly below ``min_freq`` — i.e. its frequency is
    significantly below the target at significance level ``alpha``.
    """
    if min_freq is None:
        return True
    return not bool(np.any(is_significantly_below(counts.values, nobs, float(min_freq), alpha)))


def _test_modality_ordering(train_target_rate: pd.Series, dev_target_rate: pd.Series) -> bool:
    """tests whether train and dev targets rates are similarly ranked between datasets"""
    # - grouped values have the same ranks in train/test
    # Tie-break by ordinal position (stable sort), not by label text: both series
    # already share the ordinal label order, so a stable sort preserves it. This
    # keeps carving independent of the cosmetic label strings.
    return all(train_target_rate.sort_values(kind="stable").index == dev_target_rate.sort_values(kind="stable").index)


def is_viable(test_results: dict):
    """checks if combination is viable on train and dev (if provided)"""

    return test_results["train"][Keys.VIABLE.value] and (
        test_results["dev"][Keys.VIABLE.value] or test_results["dev"][Keys.VIABLE.value] is None
    )


def test_viability(
    rates: pd.Series | pd.DataFrame,
    min_freq: int | float | None,
    target_rate: str,
    alpha: float,
    train_target_rate: pd.Series | None = None,
) -> dict:
    """tests viability of the rates.

    ``rates`` must carry per-modality ``count`` and ``frequency`` columns
    (added by the binary/continuous target-rate builders); CI tests use the
    counts and ``nobs = counts.sum()``.
    """

    # - no modality is significantly below min_freq (Wilson CI at level `alpha`)
    counts = rates["count"]
    nobs = int(counts.sum())
    min_freq_test = _test_minimum_frequency_per_modality(counts, nobs, min_freq, alpha)

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


def add_info(test_results: dict[str, bool], min_freq: int | float | None) -> dict[str, str | bool]:
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
