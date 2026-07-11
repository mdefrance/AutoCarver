"""Ridit scoring of ordinal target levels against a fixed train marginal.

Shared by :mod:`AutoCarver.discretizers.qualitatives.categorical_discretizer`
(pre-carving modality ordering for an order-only ordinal target) and
:mod:`AutoCarver.combinations.ordinal` (the per-group scalar "rate" the
viability machinery tests) — both need the same fixed reference, and the ridit
of a level (its mean midrank rescaled to ``[0, 1]``, computed from the *train*
count-marginal) is exactly what lets levels from any table (raw modalities, a
carver's grouped candidate, or a dev-sample grouping) be scored against one
shared scale — mirroring :mod:`AutoCarver.discretizers.utils.correspondence_analysis`
for the multiclass path.

The ridit of reference level ``j`` is ``F(j-1) + f_j/2`` where ``f_j`` is the
level's train frequency and ``F(j-1)`` the cumulative frequency of all lower
levels. It is invariant under any strictly increasing re-encoding of the
levels, and a group's count-weighted mean ridit is the per-group quantity the
concordance statistics (Kendall's taus, Somers' D) respond to.
"""

import numpy as np
import pandas as pd


def ridit_scores_for_levels(levels, reference_counts: pd.Series) -> np.ndarray:
    """Ridits of arbitrary numeric ``levels`` against a fixed train count-marginal.

    Parameters
    ----------
    levels : iterable of numbers
        Levels to score (e.g. a crosstab's columns) — need not all appear in
        the reference.
    reference_counts : pd.Series
        Train count-marginal, indexed by level (``value_counts`` of the train
        target, or a train crosstab's column totals). Order does not matter.

    Returns
    -------
    np.ndarray
        One ridit per queried level: ``F(j-1) + f_j/2`` for reference levels;
        a level unseen in the reference gets ``P_train(y < level)`` (the
        natural CDF extension: zero mass at that level), so tables carrying
        extra levels stay well-defined.
    """
    reference_levels = np.asarray(reference_counts.index, dtype=float)
    counts = np.asarray(reference_counts.to_numpy(), dtype=float)
    total = counts.sum()
    if total <= 0:
        raise ValueError("reference_counts must carry a positive total count")

    order = np.argsort(reference_levels)
    reference_levels = reference_levels[order]
    frequencies = counts[order] / total

    query = np.asarray(list(levels), dtype=float)
    # mass strictly below each queried level (searchsorted 'left' counts reference
    # levels < query)
    position = np.searchsorted(reference_levels, query, side="left")
    below = np.concatenate([[0.0], np.cumsum(frequencies)])[position]

    # reference levels add half their own mass (mean midrank); unseen levels add none
    safe_position = np.minimum(position, len(reference_levels) - 1)
    is_reference = reference_levels[safe_position] == query
    return below + np.where(is_reference, frequencies[safe_position] / 2.0, 0.0)


def ridits_from_counts(counts: pd.Series) -> dict:
    """Scores a count-marginal's own levels, as a ``{level: ridit}`` dict.

    Convenience wrapper over :func:`ridit_scores_for_levels` keeping the
    original (non-float-cast) level keys, so the result is directly
    ``y.map``-able (the carver's pre-sort scale).
    """
    scores = ridit_scores_for_levels(counts.index, counts)
    return {level: float(score) for level, score in zip(counts.index, scores)}
