"""Kruskal-Wallis H and its tie-correction factor."""

import numpy as np


def tie_correction(values: np.ndarray) -> float:
    """Kruskal-Wallis tie factor ``1 - Sum(t^3 - t) / (N^3 - N)``; ``1.0`` when ``N < 2``.

    Matches ``scipy.stats.tiecorrect`` (which takes ranks; ties are the same either way).
    """
    n = values.size
    if n < 2:
        return 1.0
    _, counts = np.unique(values, return_counts=True)
    ties = float((counts**3 - counts).sum())
    return 1.0 - ties / (n**3 - n)


def h_from_rank_sums(rank_sums: np.ndarray, counts: np.ndarray, n_obs: float, tie_corr: float) -> float:
    """Tie-corrected H from per-group rank sums and counts. ``nan`` when ``tie_corr == 0``.

    ``nan`` also whenever any group is empty (``0/0`` propagates through the sum) —
    matches ``scipy.stats.kruskal``, which rejects an empty sample.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        ssbn = float((rank_sums**2 / counts).sum())
    if not np.isfinite(ssbn):
        return float("nan")

    h = (12.0 / (n_obs * (n_obs + 1))) * ssbn - 3.0 * (n_obs + 1)

    if tie_corr == 0:
        return float("nan")
    return h / tie_corr
