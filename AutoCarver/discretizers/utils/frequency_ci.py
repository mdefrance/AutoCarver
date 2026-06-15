"""Wilson-score confidence interval helpers for modality-frequency tests.

A modality observed with ``count`` successes out of ``nobs`` trials is declared
*significantly below* a target frequency ``min_freq`` when the Wilson upper
bound of its proportion is below ``min_freq``. This replaces strict
``count / nobs < min_freq`` comparisons that are noisy on small samples.
"""

from functools import lru_cache

import numpy as np
from scipy.stats import norm


@lru_cache(maxsize=8)
def _z_score(alpha: float) -> float:
    """Two-sided z-score for the given alpha (cached per alpha)."""
    return float(norm.ppf(1.0 - alpha / 2.0))


def wilson_upper_bound(
    count: np.ndarray | int | float,
    nobs: int,
    alpha: float,
) -> np.ndarray | float:
    """Upper bound of the two-sided Wilson score interval for ``count / nobs``.

    Parameters
    ----------
    count : array-like or scalar
        Observed successes. Accepts integer counts or float counts (e.g.
        weighted/aggregated frequencies).
    nobs : int
        Number of trials. Must be ``>= 0``; returns ``1.0`` when ``nobs == 0``
        so callers treat empty samples as non-significant.
    alpha : float
        Two-sided significance level (e.g. ``0.05`` for a 95% interval).

    Returns
    -------
    array-like or scalar
        Wilson upper bound, same shape as ``count``.
    """
    if nobs <= 0:
        return 1.0 if np.isscalar(count) else np.ones_like(np.asarray(count, dtype=float))

    z = _z_score(alpha)
    n = float(nobs)
    phat = np.asarray(count, dtype=float) / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    half_width = (z / denom) * np.sqrt(phat * (1.0 - phat) / n + (z * z) / (4.0 * n * n))
    # A proportion CI bound is by definition in [0, 1]; clamp to absorb float
    # rounding (e.g. count == nobs cancels to exactly 1.0 in exact arithmetic but
    # lands at 1.0000000000000002 in floating point).
    upper = np.clip(center + half_width, 0.0, 1.0)

    if np.isscalar(count):
        return float(upper)
    return upper


def is_significantly_below(
    count: np.ndarray | int | float,
    nobs: int,
    min_freq: float,
    alpha: float,
) -> np.ndarray | bool:
    """Whether the observed proportion ``count / nobs`` is significantly below ``min_freq``.

    A modality is significantly below ``min_freq`` when the Wilson upper bound
    of its observed proportion is strictly below ``min_freq``.
    """
    upper = wilson_upper_bound(count, nobs, alpha)
    if isinstance(upper, np.ndarray):
        return upper < min_freq
    return bool(upper < min_freq)
