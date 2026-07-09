"""Viability tests for the multiclass evaluator: min-freq, distinct CA-score
rates, and the train/dev rank-preservation veto — all reused, unmodified, from
:mod:`AutoCarver.combinations.utils.testing` by routing a scalar CA score
through :class:`MulticlassTargetRate` (see
:mod:`AutoCarver.combinations.multiclass.multiclass_target_rates`).
"""

from __future__ import annotations

import pandas as pd
import pytest

from AutoCarver.combinations.multiclass.multiclass_combination_evaluators import (
    TschuprowtMulticlassCombinations,
)
from AutoCarver.features import OrdinalFeature

MIN_FREQ = 0.05


def _feature(labels: list[str]) -> OrdinalFeature:
    return OrdinalFeature("feature", labels)


def test_get_best_combination_fits_axis_and_returns_a_viable_combination():
    """End-to-end: a clear monotone (B, K) signal produces a viable grouping,
    and the CA axis ends up fit on the evaluator's target_rate."""
    feature = _feature(["m0", "m1", "m2", "m3"])
    xagg = pd.DataFrame(
        {1: [100, 40, 10, 5], 2: [10, 40, 40, 10], 3: [5, 10, 40, 100]},
        index=["m0", "m1", "m2", "m3"],
    )
    evaluator = TschuprowtMulticlassCombinations()
    result = evaluator.get_best_combination(feature, xagg, max_n_mod=3, min_freq=MIN_FREQ, dropna=False)

    assert result is not None
    assert evaluator.target_rate.axis is not None
    assert not evaluator.target_rate.axis.degenerate


def test_min_freq_veto_rejects_a_too_rare_modality():
    """A modality far below min_freq must not survive as its own group; the
    evaluator groups it away instead of returning a combination where it
    stands alone."""
    feature = _feature(["m0", "m1", "m2"])
    # m1 has 1 observation out of ~1000 — far below any reasonable min_freq
    xagg = pd.DataFrame(
        {1: [500, 1, 5], 2: [10, 0, 5], 3: [5, 0, 490]},
        index=["m0", "m1", "m2"],
    )
    evaluator = TschuprowtMulticlassCombinations()
    result = evaluator.get_best_combination(feature, xagg, max_n_mod=3, min_freq=0.05, dropna=False)

    assert result is not None
    # m1 must have been folded into a neighbouring group, not left standalone
    singleton_groups = [g for g in result["combination"] if g == ["m1"]]
    assert not singleton_groups


def test_dev_rank_inversion_vetoes_the_combination():
    """A dev sample whose group ordering (projected on the *train* CA axis)
    disagrees with train's own ordering must fail the robustness veto —
    ``get_best_combination`` then either falls back to a coarser/finer viable
    combination or returns ``None``."""
    feature = _feature(["m0", "m1", "m2", "m3"])
    train_xagg = pd.DataFrame(
        {1: [100, 40, 10, 5], 2: [10, 40, 40, 10], 3: [5, 10, 40, 100]},
        index=["m0", "m1", "m2", "m3"],
    )
    # dev exhibits the *reverse* class association per modality
    dev_xagg = pd.DataFrame(
        {1: [5, 10, 40, 100], 2: [10, 40, 40, 10], 3: [100, 40, 10, 5]},
        index=["m0", "m1", "m2", "m3"],
    )

    evaluator_with_dev = TschuprowtMulticlassCombinations()
    result_with_dev = evaluator_with_dev.get_best_combination(
        feature, train_xagg, dev_xagg, max_n_mod=4, min_freq=0.01, dropna=False
    )

    evaluator_without_dev = TschuprowtMulticlassCombinations()
    result_without_dev = evaluator_without_dev.get_best_combination(
        _feature(["m0", "m1", "m2", "m3"]), train_xagg, max_n_mod=4, min_freq=0.01, dropna=False
    )

    # Without a dev veto, a 2-group split survives. With a dev sample whose CA
    # projection is the exact reverse of train's on every grouping, every
    # candidate's rank order inverts on dev -> no combination is viable.
    assert result_without_dev is not None
    assert result_with_dev is None


def test_target_rate_compute_uses_fixed_train_axis_for_dev_groupings():
    """Direct unit check that `target_rate.compute` on a *different* table (a
    stand-in for a dev grouping) still projects through the axis fit on train,
    not a freshly-refit one."""
    feature = _feature(["m0", "m1", "m2", "m3"])
    train_xagg = pd.DataFrame(
        {1: [100, 40, 10, 5], 2: [10, 40, 40, 10], 3: [5, 10, 40, 100]},
        index=["m0", "m1", "m2", "m3"],
    )
    evaluator = TschuprowtMulticlassCombinations()
    evaluator.feature = feature
    evaluator.target_rate.fit_axis(train_xagg)
    axis_after_train = evaluator.target_rate.axis

    dev_like = pd.DataFrame({1: [7, 70], 2: [10, 10], 3: [70, 7]}, index=["g0", "g1"])
    rates = evaluator.target_rate.compute(dev_like)

    assert list(rates.columns) == ["ca_score", "frequency", "count"]
    # axis is unchanged by computing on a different table
    assert evaluator.target_rate.axis is axis_after_train


def test_target_rate_axis_not_set_raises():
    """Calling target_rate.compute before fit_axis raises (no silent default axis)."""
    evaluator = TschuprowtMulticlassCombinations()
    xagg = pd.DataFrame({1: [1, 2], 2: [3, 4], 3: [5, 6]}, index=["a", "b"])
    with pytest.raises(RuntimeError, match="CA axis is not fit"):
        evaluator.target_rate.compute(xagg)
