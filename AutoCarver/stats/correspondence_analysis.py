"""Correspondence-analysis first-axis modality ordering and row scoring.

Shared by :mod:`AutoCarver.discretizers.qualitatives.categorical_discretizer`
(pre-carving modality ordering for an unordered multiclass target) and
:mod:`AutoCarver.combinations.multiclass` (the per-group scalar "rate" the
viability machinery tests) — both need the same fixed axis, and the CA
transition formula (project a row's own profile through the *training*
column masses and first right singular vector) is exactly what lets a
row from any table (raw modalities, a carver's own grouped candidate, or a
dev-sample grouping) be scored against one shared axis.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CAAxis:
    """A fixed correspondence-analysis first axis, reusable to score new rows.

    ``degenerate`` is ``True`` when the training table carried too little
    structure to define a meaningful axis (fewer than 3 rows, fewer than 2
    columns, or an ~zero first singular value); callers then fall back to a
    frequency-based order (see :func:`ca_row_scores`).
    """

    col_mass: np.ndarray
    v1: np.ndarray
    degenerate: bool = False


def fit_ca_axis(xtab: pd.DataFrame, tol: float = 1e-10) -> CAAxis:
    """Fits the correspondence-analysis first axis of a crosstab.

    ``xtab`` is an ``(n_rows, K)`` count crosstab (rows = modalities/groups,
    columns = target classes). Standardizes the table (row/column mass
    normalization), takes its SVD, and returns the (sign-fixed) first right
    singular vector plus the column mass vector needed to project any row's
    own profile onto it (see :func:`ca_row_scores`).
    """
    values = np.asarray(xtab.to_numpy(), dtype=float)
    n_rows, n_cols = values.shape
    total = values.sum()

    # 1 or 2 rows: order is trivial (nothing to disambiguate); skip CA.
    if n_rows <= 2 or n_cols < 2 or total <= 0:
        return CAAxis(col_mass=np.zeros(n_cols), v1=np.zeros(n_cols), degenerate=True)

    row_totals = values.sum(axis=1)
    col_totals = values.sum(axis=0)
    r = row_totals / total
    c = col_totals / total
    row_ok = r > tol
    col_ok = c > tol
    if row_ok.sum() < 2 or col_ok.sum() < 2:
        return CAAxis(col_mass=c, v1=np.zeros(n_cols), degenerate=True)

    P = values / total
    S = np.zeros_like(P)
    ix = np.ix_(row_ok, col_ok)
    r_ok = r[row_ok]
    c_ok = c[col_ok]
    S[ix] = (P[ix] - np.outer(r_ok, c_ok)) / np.sqrt(np.outer(r_ok, c_ok))

    _, sigma, vt = np.linalg.svd(S, full_matrices=False)
    if sigma.size == 0 or sigma[0] <= tol:
        return CAAxis(col_mass=c, v1=np.zeros(n_cols), degenerate=True)

    unsigned_axis = CAAxis(col_mass=c, v1=vt[0], degenerate=False)
    unsigned_scores = ca_row_scores(xtab, unsigned_axis)

    # Sign convention: flip so the largest-mass row's score is non-negative;
    # if that row's score is exactly zero, fall back to the next-largest-mass
    # row. Ties (equal mass, e.g. a perfectly symmetric table) are broken by
    # the row's own count vector, not its position: sorting by content — not
    # position — keeps the winner (and hence the sign) identical between the
    # original table and any row permutation of it.
    tie_break_order = sorted(
        range(n_rows),
        key=lambda i: (-row_totals[i], -abs(float(unsigned_scores.iloc[i])), tuple(values[i])),
    )
    sign = 1.0
    for row_pos in tie_break_order:
        score = float(unsigned_scores.iloc[row_pos])
        if score != 0.0:
            sign = 1.0 if score > 0 else -1.0
            break

    return CAAxis(col_mass=c, v1=vt[0] * sign, degenerate=False)


def ca_row_scores(xtab: pd.DataFrame, axis: CAAxis) -> pd.Series:
    """Projects each row of ``xtab`` onto a fixed :class:`CAAxis`.

    .. math::

        \\text{score}_i = \\sum_k \\frac{p_{ik} - c_k}{\\sqrt{c_k}}\\, v_{1k}

    where :math:`p_{ik}` is row :math:`i`'s own profile (proportions across
    columns), :math:`c_k` the fixed (training) column mass, and :math:`v_1`
    the fitted first right singular vector. Only the row's own profile and
    the fixed (training) column masses / axis are needed, so this is
    well-defined for any row set sharing ``xtab``'s columns — including a
    dev-sample grouping the axis was never fit on, or a carver's grouped
    candidate table.

    Falls back to (deterministic) descending-frequency scoring when
    ``axis.degenerate`` (encoded as ``-row_total`` so ascending sort still
    yields frequency-descending order).

    Raises
    ------
    ValueError
        When ``xtab`` doesn't carry exactly the classes the axis was fit on —
        typically a target class present in a later sample but unseen at fit
        time. The axis is fixed by construction, so such a table cannot be
        projected onto it (see :mod:`AutoCarver.stability`).
    """
    values = np.asarray(xtab.to_numpy(), dtype=float)
    row_totals = values.sum(axis=1)

    if axis.degenerate:
        return pd.Series(-row_totals, index=xtab.index)

    c = axis.col_mass
    if values.shape[1] != c.shape[0]:
        raise ValueError(
            f"[ca_row_scores] crosstab carries {values.shape[1]} target classes {list(xtab.columns)} but the "
            f"fitted CA axis was built on {c.shape[0]}; the axis cannot score classes it was never fit on."
        )
    col_ok = c > 1e-10
    safe_row_totals = np.where(row_totals > 0, row_totals, 1.0)
    profiles = values / safe_row_totals[:, None]
    profiles = np.where(row_totals[:, None] > 0, profiles, 0.0)
    centered = np.where(col_ok, (profiles - c) / np.sqrt(np.where(col_ok, c, 1.0)), 0.0)
    scores = centered @ axis.v1
    return pd.Series(scores, index=xtab.index)
