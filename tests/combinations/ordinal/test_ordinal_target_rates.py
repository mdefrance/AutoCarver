"""Set of tests for the ordinal target rates."""

import numpy as np
import pandas as pd
from pytest import raises

from AutoCarver.combinations.ordinal.ordinal_target_rates import TargetMeanLevel, TargetMeanRidit

# train crosstab: 3 feature groups x levels {1, 2, 3}; column totals (50, 30, 20)
TRAIN_XTAB = pd.DataFrame(
    {1: [30, 15, 5], 2: [10, 10, 10], 3: [2, 6, 12]},
    index=["a", "b", "c"],
)
# train ridits for f = (.5, .3, .2): (.25, .65, .9)
TRAIN_RIDITS = {1: 0.25, 2: 0.65, 3: 0.9}


def test_target_mean_ridit_requires_fit_reference():
    """compute raises until the train reference marginal is fit."""
    rate = TargetMeanRidit()
    with raises(RuntimeError):
        rate.compute(TRAIN_XTAB)

    rate.fit_reference(TRAIN_XTAB)
    assert list(rate.reference) == [50, 30, 20]


def test_target_mean_ridit_compute():
    """Per-group rate is the count-weighted mean of the train ridits."""
    rate = TargetMeanRidit()
    rate.fit_reference(TRAIN_XTAB)
    computed = rate.compute(TRAIN_XTAB)

    expected = {
        group: sum(TRAIN_XTAB.loc[group, level] * TRAIN_RIDITS[level] for level in [1, 2, 3])
        / TRAIN_XTAB.loc[group].sum()
        for group in ["a", "b", "c"]
    }
    assert np.allclose(computed["target_mean_ridit"], [expected["a"], expected["b"], expected["c"]])
    # bounded [0, 1] and monotone in the groups' construction
    assert computed["target_mean_ridit"].between(0, 1).all()
    assert computed["target_mean_ridit"].is_monotonic_increasing


def test_target_mean_ridit_dev_table_with_mismatched_levels():
    """A dev table missing a train level and carrying an extra one stays well-defined
    (train ridits for shared levels, CDF extension for unseen ones)."""
    rate = TargetMeanRidit()
    rate.fit_reference(TRAIN_XTAB)

    # dev has no level 2, but an extra level 4 (unseen in train -> ridit F(<4) = 1.0)
    dev_xtab = pd.DataFrame({1: [4, 1], 3: [1, 2], 4: [0, 2]}, index=["a", "b"])
    computed = rate.compute(dev_xtab)

    expected_a = (4 * 0.25 + 1 * 0.9 + 0 * 1.0) / 5
    expected_b = (1 * 0.25 + 2 * 0.9 + 2 * 1.0) / 5
    assert np.allclose(computed["target_mean_ridit"], [expected_a, expected_b])


def test_target_mean_level_unchanged_without_level_values():
    """level_values=None reads the levels from the crosstab columns (previous behaviour)."""
    rate = TargetMeanLevel()
    computed = rate.compute(TRAIN_XTAB)

    expected = [(30 + 20 + 6) / 42, (15 + 20 + 18) / 31, (5 + 20 + 36) / 27]
    assert np.allclose(computed["target_mean_level"], expected)


def test_target_mean_level_with_level_values():
    """level_values maps the crosstab columns onto the user scale."""
    rate = TargetMeanLevel(level_values={1: 0.0, 2: 0.1, 3: 10.0})
    computed = rate.compute(TRAIN_XTAB)

    expected = [(10 * 0.1 + 2 * 10.0) / 42, (10 * 0.1 + 6 * 10.0) / 31, (10 * 0.1 + 12 * 10.0) / 27]
    assert np.allclose(computed["target_mean_level"], expected)


def test_target_mean_level_missing_column_raises():
    """A crosstab column absent from level_values raises a clear error."""
    rate = TargetMeanLevel(level_values={1: 0.0, 2: 0.1})
    with raises(ValueError):
        rate.compute(TRAIN_XTAB)


def test_target_mean_level_validates_strictly_increasing():
    """level_values must be strictly increasing when levels are sorted ascending."""
    with raises(ValueError):
        TargetMeanLevel(level_values={1: 0.0, 2: 0.5, 3: 0.5})
    with raises(ValueError):
        TargetMeanLevel(level_values={1: 1.0, 2: 0.5})
