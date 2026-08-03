"""Set of tests for the ridits module."""

import numpy as np
import pandas as pd
from pytest import raises

from AutoCarver.stats.ridits import ridit_scores_for_levels, ridits_from_counts


def test_ridits_uniform_marginal():
    """Equal counts: ridits are the midpoints of K equal CDF slices."""
    counts = pd.Series({1: 10, 2: 10, 3: 10, 4: 10})
    scores = ridit_scores_for_levels([1, 2, 3, 4], counts)
    assert np.allclose(scores, [0.125, 0.375, 0.625, 0.875])


def test_ridits_skewed_marginal():
    """Hand-computed ridits F(j-1) + f_j/2 on a skewed marginal."""
    counts = pd.Series({1: 50, 2: 30, 3: 20})
    scores = ridit_scores_for_levels([1, 2, 3], counts)
    # f = (.5, .3, .2) -> ridits = (.25, .5 + .15, .8 + .1)
    assert np.allclose(scores, [0.25, 0.65, 0.9])


def test_ridits_reference_order_does_not_matter():
    """The reference marginal is sorted by level internally."""
    counts = pd.Series({2: 30, 3: 20, 1: 50})  # shuffled index
    scores = ridit_scores_for_levels([1, 2, 3], counts)
    assert np.allclose(scores, [0.25, 0.65, 0.9])


def test_ridits_unseen_level_cdf_extension():
    """Levels unseen in the reference get P(y < level): zero mass at the level itself."""
    counts = pd.Series({1: 50, 2: 30, 3: 20})
    scores = ridit_scores_for_levels([0, 2.5, 10], counts)
    # below everything -> 0; between 2 and 3 -> F(2) = .8; above everything -> 1
    assert np.allclose(scores, [0.0, 0.8, 1.0])


def test_ridits_single_level_degenerate():
    """A single-level reference: its own ridit is .5, unseen levels hit the CDF ends."""
    counts = pd.Series({3: 10})
    scores = ridit_scores_for_levels([2, 3, 4], counts)
    assert np.allclose(scores, [0.0, 0.5, 1.0])


def test_ridits_invariant_under_monotone_reencoding():
    """Re-encoding levels through any strictly increasing map leaves ridits unchanged."""
    counts = pd.Series({1: 5, 2: 25, 3: 40, 4: 30})
    reencoded = pd.Series({1: 5, 2: 25, 3: 40, 10: 30})  # 4 -> 10
    assert np.allclose(
        ridit_scores_for_levels([1, 2, 3, 4], counts),
        ridit_scores_for_levels([1, 2, 3, 10], reencoded),
    )


def test_ridits_zero_total_raises():
    """An empty reference marginal is rejected."""
    with raises(ValueError):
        ridit_scores_for_levels([1], pd.Series({1: 0, 2: 0}))


def test_ridits_from_counts_keeps_original_keys():
    """ridits_from_counts keys the dict by the marginal's own (non-cast) levels."""
    counts = pd.Series({1: 50, 2: 30, 3: 20})
    scores = ridits_from_counts(counts)
    assert set(scores) == {1, 2, 3}
    assert np.allclose([scores[1], scores[2], scores[3]], [0.25, 0.65, 0.9])
