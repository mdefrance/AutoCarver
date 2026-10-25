"""Pearson chi² and its Cramér's V / Tschuprow's T derivatives.

Single home for arithmetic that used to be duplicated across the binary and
multiclass combination evaluators, and the qualitative selector kernels.
"""

import math

import numpy as np


def pearson_chi2(observed: np.ndarray, *, guard_zero_expected: bool = False) -> float:
    """Pearson :math:`\\chi^2` of a ``(B, C)`` observed contingency table.

    Replicates :func:`scipy.stats.chi2_contingency` with ``correction=False``:
    expected frequencies via the outer product of marginals divided by N.

    .. math::

        \\chi^2 = \\sum_{i, j} \\frac{(O_{ij} - E_{ij})^2}{E_{ij}}, \\qquad
        E_{ij} = \\frac{n_{i.}\\, n_{.j}}{n}

    where :math:`n_{i.}` and :math:`n_{.j}` are the row and column marginals
    and :math:`n` the grand total. No Yates continuity correction, not even on
    :math:`2 \\times 2` tables: this chi² feeds Cramér's V and Tschuprow's T,
    which are defined on the uncorrected statistic (a corrected one understates
    :math:`\\phi` on small samples). Yates only belongs where a p-value is read
    (:func:`AutoCarver.stability.metrics.chi2_homogeneity`).

    ``guard_zero_expected`` replaces the ``0/0`` of an all-zero row or column with
    ``0`` instead of ``nan``. The selector kernels need it (they build tables by
    ``bincount``, which can produce empty rows); the combination evaluators must
    **not** use it — they shift every cell by ``+tol`` beforehand, and changing
    that would break bit-exactness.
    """
    R = observed.sum(axis=1)
    C = observed.sum(axis=0)
    N = float(observed.sum())
    expected = np.outer(R, C) / N

    obs = observed
    if guard_zero_expected:
        with np.errstate(divide="ignore", invalid="ignore"):
            contrib = np.where(expected > 0, (obs - expected) ** 2 / expected, 0.0)
        return float(contrib.sum())
    return float(((obs - expected) ** 2 / expected).sum())


def bergsma_phi2(chi2: float, n_obs: float, n_rows: float, n_cols: float) -> tuple[float, float, float]:
    """Bergsma (2013) bias-corrected :math:`\\phi^2` and effective table dimensions.

    The plain :math:`\\phi^2 = \\chi^2 / N` is biased upward on finite samples:
    under independence :math:`\\chi^2` averages its degrees of freedom, so pure
    noise scores :math:`\\phi^2 \\approx (r-1)(c-1)/N > 0`, more so with more
    modalities and fewer rows. The correction subtracts that chance level and
    shrinks the dimensions to match:

    .. math::

        \\tilde\\phi^2_+ = \\max\\left(0,\\ \\frac{\\chi^2}{N} - \\frac{(r-1)(c-1)}{N-1}\\right),
        \\qquad \\tilde r = r - \\frac{(r-1)^2}{N-1}, \\qquad \\tilde c = c - \\frac{(c-1)^2}{N-1}

    Cramér's V and Tschuprow's T are then computed from :math:`\\tilde\\phi^2_+`,
    :math:`\\tilde r` and :math:`\\tilde c` (see :func:`cramerv_tschuprowt`). This
    keeps them comparable across tables of different sizes and sample sizes: a
    grouping with more modalities only scores higher if it adds more association
    than chance would, and a feature observed on few rows no longer outranks a
    well-covered one on noise alone. Its effect vanishes as :math:`N` grows.

    Returns ``(phi2, rows, cols)``; all ``NaN`` when ``chi2`` is ``NaN`` or
    ``n_obs <= 1``.

    References
    ----------
    W. Bergsma, "A bias-correction for Cramér's V and Tschuprow's T",
    Journal of the Korean Statistical Society 42 (2013), 323-328.
    """
    if n_obs <= 1 or math.isnan(chi2):
        return float("nan"), float("nan"), float("nan")
    phi2 = chi2 / n_obs - (n_rows - 1) * (n_cols - 1) / (n_obs - 1)
    # a difference at float-noise level is no association: without this, the square root in V/T
    # turns 1e-16 rounding gaps between two chi² computations into 1e-8 differences
    if phi2 <= 1e-12 * max(1.0, chi2 / n_obs):
        phi2 = 0.0
    rows = n_rows - (n_rows - 1) ** 2 / (n_obs - 1)
    cols = n_cols - (n_cols - 1) ** 2 / (n_obs - 1)
    return phi2, rows, cols


def _bias_corrected_v_t(chi2: float, n_obs: float, n_rows: float, n_cols: float) -> tuple[float, float]:
    """Unrounded Bergsma V and T; ``NaN`` wherever a corrected denominator is not positive.

    Clipped to 1: two identical columns give exactly 1 in exact arithmetic, and float
    rounding must not push them above it (a redundancy filter at threshold 1.0 drops
    only values strictly above).
    """
    phi2, rows, cols = bergsma_phi2(chi2, n_obs, n_rows, n_cols)
    v_denom = min(rows, cols) - 1
    t_denom_sq = (rows - 1) * (cols - 1)
    cramerv = min(1.0, math.sqrt(phi2 / v_denom)) if v_denom > 0 else float("nan")
    tschuprowt = min(1.0, math.sqrt(phi2 / math.sqrt(t_denom_sq))) if rows > 1 and cols > 1 else float("nan")
    return cramerv, tschuprowt


def cramerv_tschuprowt(chi2: float, n_obs: float, n_rows: int, n_cols: int, tol: float) -> tuple[float, float]:
    """Bias-corrected Cramér's V and Tschuprow's T from a chi² on an ``(n_rows, n_cols)`` table.

    .. math::

        \\tilde V = \\sqrt{\\frac{\\tilde\\phi^2_+}{\\min(\\tilde B, \\tilde K) - 1}}, \\qquad
        \\tilde T = \\sqrt{\\frac{\\tilde\\phi^2_+}{\\sqrt{(\\tilde B-1)(\\tilde K-1)}}}

    with :math:`B` = ``n_rows``, :math:`K` = ``n_cols``, :math:`N` = ``n_obs`` and
    the Bergsma (2013) correction of :func:`bergsma_phi2`. Both are ``NaN`` when
    their denominator vanishes (mirrors the binary/ordinal ``None``-on-degenerate
    convention), and both are quantised to ``tol``.

    Every combination evaluator (binary one-shot, binary DP, closed form,
    multiclass) converts chi² through this one function, so they agree
    bit-for-bit by construction. For a fixed number of groups and ``N`` the
    correction is monotone in chi², so the chi²-maximising DP search is
    unchanged; only candidates with different group counts are re-weighed.
    """
    cramerv, tschuprowt = _bias_corrected_v_t(chi2, n_obs, n_rows, n_cols)
    if not math.isnan(cramerv):
        cramerv = round(cramerv / tol) * tol
    if not math.isnan(tschuprowt):
        tschuprowt = round(tschuprowt / tol) * tol
    return cramerv, tschuprowt


def cramerv_tschuprowt_unrounded(chi2: float, n_obs: float, n_mod_x: float, n_mod_y: float) -> tuple[float, float]:
    """Selector-side bias-corrected V / T: no ``tol`` quantisation.

    Same Bergsma (2013) correction as :func:`cramerv_tschuprowt` (see
    :func:`bergsma_phi2`), with :math:`n_x`, :math:`n_y` the two columns'
    modality counts and ``n_obs`` the non-missing pair count. The correction
    matters most here: selectors compare features observed on very different
    numbers of rows, and the plain V/T of a sparsely observed feature is
    inflated by chance. Kept separate from :func:`cramerv_tschuprowt` for its
    degenerate-case conventions (a single-modality column returns ``chi2`` as V
    and ``0.0`` as T); do not "simplify" the two into one.
    """
    if min(n_mod_x, n_mod_y) <= 1:
        cramerv = float(chi2)
        tschuprowt = float("nan") if (n_mod_x - 1) * (n_mod_y - 1) < 0 else 0.0
        return cramerv, tschuprowt
    return _bias_corrected_v_t(chi2, n_obs, n_mod_x, n_mod_y)
