"""Parity tests for the closed-form continuous-robustness path (Step 3.5).

The legacy viability path runs ``_grouper`` (a ``Series.groupby(...).sum()``
over Python lists of y values) followed by ``target_rate.compute`` (a
``Series.apply(np.mean)`` over those lists). The fast path replaces both with
``np.bincount`` over per-modality ``(n, sum_y)`` arrays.

These tests assert that the fast path returns identical viability decisions
(and matching frequencies / target rates within ~1e-12) to the legacy path
across a wide variety of combinations and dataset shapes — both on the train
and on the dev sample, including the modality-ordering check.

Per §6 of the speedup plan we also verify that ``TargetMedian`` (which has no
``compute_from_stats``) silently falls back to the legacy path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from AutoCarver.combinations.continuous.continuous_combination_evaluators import KruskalCombinations
from AutoCarver.combinations.continuous.continuous_target_rates import TargetMean, TargetMedian
from AutoCarver.combinations.utils.combinations import combination_formatter, consecutive_combinations
from AutoCarver.combinations.utils.testing import Keys, is_viable
from AutoCarver.combinations.utils.testing import test_viability as _test_viability  # avoid pytest collection


def _legacy_viability_train(evaluator, combination):
    """Pure legacy path: _grouper + compute + test_viability."""
    xagg = evaluator._grouper(evaluator.samples.train, combination["index_to_groupby"])
    rates = evaluator.target_rate.compute(xagg)
    return _test_viability(rates, evaluator.min_freq, evaluator.target_rate.__name__)


def _legacy_viability_dev(evaluator, train_result, combination):
    """Pure legacy path for dev viability — mirrors the base class."""
    if not train_result[Keys.VIABLE.value] or evaluator.samples.dev.xagg is None:
        return {**train_result, "dev": {Keys.VIABLE.value: None}}
    train_target_rate = train_result["train_rates"][evaluator.target_rate.__name__]
    grouped_dev = evaluator._grouper(evaluator.samples.dev, combination["index_to_groupby"])
    dev_rates = evaluator.target_rate.compute(grouped_dev)
    dev_results = _test_viability(dev_rates, evaluator.min_freq, evaluator.target_rate.__name__, train_target_rate)
    merged = {**train_result, **dev_results}
    merged[Keys.VIABLE.value] = is_viable(merged)
    return merged


def _build_random_xagg(rng: np.random.Generator, modalities: list[str], size_range=(2, 20)):
    """Random Series-of-lists xagg matching what `get_target_values_by_modality` produces."""
    return pd.Series({m: rng.standard_normal(size=int(rng.integers(*size_range))).tolist() for m in modalities})


def _consume_setup(evaluator, combos_list):
    """Run _compute_associations once to populate the modality-stats cache."""
    # exhaust the streaming pipeline — first `next()` runs the upfront setup
    # that fills `self._train_modality_stats`.
    stream = evaluator._compute_associations(evaluator._group_xagg_by_combinations(iter(combos_list)))
    return list(stream)


# ---------------------------------------------------------------------------
# Train-only parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(6))
def test_train_viability_matches_legacy(seed: int):
    """Walk a few hundred combinations; fast viability decisions must match legacy."""
    rng = np.random.default_rng(seed)
    modalities = [chr(ord("A") + i) for i in range(7)]
    xagg_train = _build_random_xagg(rng, modalities, size_range=(3, 25))

    evaluator = KruskalCombinations(target_rate=TargetMean())
    evaluator.min_freq = 0.05
    evaluator.samples.set(train=xagg_train, dev=None)

    combos_list = list(consecutive_combinations(modalities, max_group_size=5))
    # Run setup pass so the modality-stats cache is populated for the fast path.
    associations = _consume_setup(evaluator, combos_list)

    for combo in associations:
        legacy = _legacy_viability_train(evaluator, combo)
        fast = evaluator._test_viability_train(combo)

        assert legacy[Keys.VIABLE.value] == fast[Keys.VIABLE.value]
        assert legacy["train"][Keys.VIABLE.value] == fast["train"][Keys.VIABLE.value]
        assert legacy["train"][Keys.INFO.value] == fast["train"][Keys.INFO.value]

        legacy_rates = legacy["train_rates"]
        fast_rates = fast["train_rates"]
        # same group labels in the same order
        assert list(legacy_rates.index) == list(fast_rates.index)
        np.testing.assert_allclose(
            fast_rates[evaluator.target_rate.__name__].values,
            legacy_rates[evaluator.target_rate.__name__].values,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            fast_rates["frequency"].values,
            legacy_rates["frequency"].values,
            rtol=1e-12,
            atol=1e-12,
        )


# ---------------------------------------------------------------------------
# Train + dev parity (exercises modality ordering check)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(6))
def test_train_and_dev_viability_match_legacy(seed: int):
    """Same as above with a dev sample; covers the ordering check."""
    rng = np.random.default_rng(seed + 100)
    modalities = [chr(ord("A") + i) for i in range(6)]
    xagg_train = _build_random_xagg(rng, modalities, size_range=(5, 30))
    xagg_dev = _build_random_xagg(rng, modalities, size_range=(3, 20))

    evaluator = KruskalCombinations(target_rate=TargetMean())
    evaluator.min_freq = 0.05
    evaluator.samples.set(train=xagg_train, dev=xagg_dev)

    combos_list = list(consecutive_combinations(modalities, max_group_size=4))
    associations = _consume_setup(evaluator, combos_list)

    for combo in associations:
        legacy_train = _legacy_viability_train(evaluator, combo)
        fast_train = evaluator._test_viability_train(combo)
        assert legacy_train[Keys.VIABLE.value] == fast_train[Keys.VIABLE.value]

        legacy_full = _legacy_viability_dev(evaluator, legacy_train, combo)
        fast_full = evaluator._test_viability_dev(fast_train, combo)

        assert legacy_full[Keys.VIABLE.value] == fast_full[Keys.VIABLE.value]
        assert legacy_full["dev"][Keys.VIABLE.value] == fast_full["dev"][Keys.VIABLE.value]
        if legacy_full["dev"][Keys.VIABLE.value] is not None:
            assert legacy_full["dev"].get(Keys.INFO.value) == fast_full["dev"].get(Keys.INFO.value)


# ---------------------------------------------------------------------------
# Empty groups in dev → frequency 0 / mean NaN, same as legacy
# ---------------------------------------------------------------------------


def test_dev_with_empty_modality_matches_legacy():
    """Modalities absent from dev get n=0 / sum_y=0 and the legacy
    apply(np.mean) returns NaN — both paths must agree on viability."""
    rng = np.random.default_rng(42)
    modalities = ["A", "B", "C", "D"]
    xagg_train = _build_random_xagg(rng, modalities, size_range=(5, 15))
    # Dev has empty list for "C" — mimics a modality unseen on dev.
    xagg_dev = pd.Series({"A": [0.1, 0.2, 0.3], "B": [1.0, 2.0], "C": [], "D": [3.0, 4.0]})

    evaluator = KruskalCombinations(target_rate=TargetMean())
    evaluator.min_freq = 0.05
    evaluator.samples.set(train=xagg_train, dev=xagg_dev)

    combos_list = list(consecutive_combinations(modalities, max_group_size=3))
    associations = _consume_setup(evaluator, combos_list)

    for combo in associations:
        legacy_train = _legacy_viability_train(evaluator, combo)
        fast_train = evaluator._test_viability_train(combo)
        legacy_full = _legacy_viability_dev(evaluator, legacy_train, combo)
        fast_full = evaluator._test_viability_dev(fast_train, combo)
        assert legacy_full[Keys.VIABLE.value] == fast_full[Keys.VIABLE.value]


# ---------------------------------------------------------------------------
# TargetMedian → must fall back to the legacy path (no compute_from_stats)
# ---------------------------------------------------------------------------


def test_target_median_falls_back_to_legacy():
    """TargetMedian deliberately has no `compute_from_stats`; the override
    must call `super()._test_viability_train` and produce the median-based
    result, not mean-based."""
    rng = np.random.default_rng(7)
    modalities = ["A", "B", "C"]
    xagg_train = _build_random_xagg(rng, modalities, size_range=(5, 15))

    evaluator = KruskalCombinations(target_rate=TargetMedian())
    evaluator.min_freq = 0.05
    evaluator.samples.set(train=xagg_train, dev=None)

    combos_list = list(consecutive_combinations(modalities, max_group_size=3))
    _consume_setup(evaluator, combos_list)

    combo = {"combination": [["A"], ["B"], ["C"]], "index_to_groupby": combination_formatter([["A"], ["B"], ["C"]])}
    legacy = _legacy_viability_train(evaluator, combo)
    fast = evaluator._test_viability_train(combo)

    # Fast path should fall back, producing identical median-based output.
    assert legacy[Keys.VIABLE.value] == fast[Keys.VIABLE.value]
    pd.testing.assert_frame_equal(fast["train_rates"], legacy["train_rates"])
