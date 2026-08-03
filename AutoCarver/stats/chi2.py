"""Pearson chi² and its Cramér's V / Tschuprow's T derivatives.

Single home for arithmetic that used to be duplicated across the binary and
multiclass combination evaluators, and the qualitative selector kernels.
"""

import math

import numpy as np
import pandas as pd


def pearson_chi2(observed: np.ndarray, *, guard_zero_expected: bool = False) -> float:
    """Pearson :math:`\\chi^2` of a ``(B, C)`` observed contingency table.

    Replicates :func:`scipy.stats.chi2_contingency` defaults: expected
    frequencies via the outer product of marginals divided by N, with Yates
    correction iff the table is exactly 2x2 (matches scipy's own threshold).

    .. math::

        \\chi^2 = \\sum_{i, j} \\frac{(O_{ij} - E_{ij})^2}{E_{ij}}, \\qquad
        E_{ij} = \\frac{n_{i.}\\, n_{.j}}{n}

    where :math:`n_{i.}` and :math:`n_{.j}` are the row and column marginals
    and :math:`n` the grand total. When the table is exactly :math:`2 \\times 2`,
    Yates' continuity correction shrinks :math:`|O_{ij} - E_{ij}|` by :math:`0.5`
    before squaring (matches scipy's own threshold for applying it).

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
    if obs.shape == (2, 2):
        diff = expected - obs
        direction = np.sign(diff)
        magnitude = np.minimum(0.5, np.abs(diff))
        obs = obs + magnitude * direction

    if guard_zero_expected:
        with np.errstate(divide="ignore", invalid="ignore"):
            contrib = np.where(expected > 0, (obs - expected) ** 2 / expected, 0.0)
        return float(contrib.sum())
    return float(((obs - expected) ** 2 / expected).sum())


def cramerv_tschuprowt(chi2: float, n_obs: float, n_rows: int, n_cols: int, tol: float) -> tuple[float, float]:
    """Cramér's V and Tschuprow's T from a chi² computed on an ``(n_rows, n_cols)`` table.

    .. math::

        V = \\sqrt{\\frac{\\chi^2}{N (\\min(B, K) - 1)}}, \\qquad
        T = \\sqrt{\\frac{\\chi^2}{N \\sqrt{(B-1)(K-1)}}}

    with :math:`B` = ``n_rows``, :math:`K` = ``n_cols``, :math:`N` = ``n_obs``.
    Both are ``NaN`` when their denominator vanishes (mirrors the
    binary/ordinal ``None``-on-degenerate convention).

    For ``n_cols == 2``, ``T`` is instead derived from the (already rounded)
    ``V`` via :math:`T = V / \\sqrt[4]{B - 1}` — the exact expression the binary
    combination evaluator's closed form uses. Both formulas are mathematically
    identical at ``K=2``, but only computing it this way guarantees the binary
    and multiclass evaluators agree bit-for-bit (independent ``sqrt``/``pow``
    sequences are not guaranteed to round identically) — pinned by the K=2
    parity test.
    """
    v_denom = min(n_rows, n_cols) - 1
    if v_denom > 0 and n_obs > 0:
        cramerv = math.sqrt(chi2 / (n_obs * v_denom))
        cramerv = round(cramerv / tol) * tol
    else:
        cramerv = float("nan")

    if n_cols == 2:
        if n_rows > 1:
            tschuprowt = cramerv / math.sqrt(math.sqrt(n_rows - 1))
            if pd.notna(tschuprowt):
                tschuprowt = round(tschuprowt / tol) * tol
        else:
            tschuprowt = cramerv
    else:
        t_denom = math.sqrt((n_rows - 1) * (n_cols - 1)) if n_rows > 1 else 0.0
        if t_denom > 0 and n_obs > 0:
            tschuprowt = math.sqrt(chi2 / (n_obs * t_denom))
            tschuprowt = round(tschuprowt / tol) * tol
        else:
            tschuprowt = float("nan")

    return cramerv, tschuprowt


def cramerv_tschuprowt_unrounded(chi2: float, n_obs: float, n_mod_x: float, n_mod_y: float) -> tuple[float, float]:
    """Selector-side V / T: no ``tol`` quantisation, T from the raw chi² (not from V).

    .. math::

        V = \\sqrt{\\frac{\\chi^2}{N (\\min(n_x, n_y) - 1)}}, \\qquad
        T = \\sqrt{\\frac{\\chi^2}{N \\sqrt{(n_x-1)(n_y-1)}}}

    with :math:`n_x`, :math:`n_y` the two features' modality counts. Different
    normalisation from :func:`cramerv_tschuprowt` — ``n_obs`` here is
    the non-missing pair count and there is no rounding — so this is kept as a
    separate function rather than merged into it; do not "simplify" the two
    into one.
    """
    min_n_mod = min(n_mod_x, n_mod_y)
    cramerv = math.sqrt(chi2 / n_obs / (min_n_mod - 1)) if min_n_mod > 1 else float(chi2)

    dof_prod = (n_mod_x - 1) * (n_mod_y - 1)
    if dof_prod < 0:
        tschuprowt = float("nan")
    else:
        dof_mods = math.sqrt(dof_prod)
        tschuprowt = math.sqrt(chi2 / n_obs / dof_mods) if dof_mods > 0 else 0.0

    return cramerv, tschuprowt
