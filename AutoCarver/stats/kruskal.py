"""Kruskal-Wallis H and its tie-correction factor."""

import numpy as np


def tie_correction(values: np.ndarray) -> float:
    """Kruskal-Wallis tie factor; ``1.0`` when ``N < 2``.

    .. math::

        C_{tie} = 1 - \\frac{\\sum_t (t^3 - t)}{N^3 - N}

    where the sum runs over each group of :math:`t` tied values and :math:`N`
    is the total sample size. Matches ``scipy.stats.tiecorrect`` (which takes
    ranks; ties are the same either way).
    """
    n = values.size
    if n < 2:
        return 1.0
    _, counts = np.unique(values, return_counts=True)
    ties = float((counts**3 - counts).sum())
    return 1.0 - ties / (n**3 - n)


def h_from_rank_sums(rank_sums: np.ndarray, counts: np.ndarray, n_obs: float, tie_corr: float) -> float:
    """Tie-corrected H from per-group rank sums and counts. ``nan`` when ``tie_corr == 0``.

    .. math::

        H = \\frac{1}{C_{tie}} \\left[\\frac{12}{N(N+1)} \\sum_g \\frac{R_g^2}{n_g} - 3(N+1)\\right]

    where :math:`R_g` and :math:`n_g` are group ``g``'s rank sum and count,
    :math:`N` the total sample size, and :math:`C_{tie}` the
    :func:`tie_correction` factor. ``nan`` also whenever any group is empty
    (``0/0`` propagates through the sum) — matches ``scipy.stats.kruskal``,
    which rejects an empty sample.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        ssbn = float((rank_sums**2 / counts).sum())
    if not np.isfinite(ssbn):
        return float("nan")

    h = (12.0 / (n_obs * (n_obs + 1))) * ssbn - 3.0 * (n_obs + 1)

    if tie_corr == 0:
        return float("nan")
    return h / tie_corr
