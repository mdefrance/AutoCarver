"""Rank-association statistics of an ordered contingency table: Kendall's tau-b,
Stuart's tau-c and Somers' D."""

import math

import numpy as np


def concordant_minus_discordant(values: np.ndarray) -> float:
    """Concordant minus discordant pairs :math:`C - D` of an ordered table.

    .. math::

        C - D = \\sum_{i,j} n_{ij}
        \\left(\\sum_{k>i,\\, l>j} n_{kl} - \\sum_{k>i,\\, l<j} n_{kl}\\right)

    ``values`` is the ``(r, c)`` cell-count array with rows / columns already
    ascending. Computed in closed form from the table's cumulative cell sums.
    """
    # concordant partners of each cell: counts strictly down-right (k>i, l>j)
    suffix = np.cumsum(np.cumsum(values[::-1, ::-1], axis=0), axis=1)[::-1, ::-1]
    down_right = np.zeros_like(values)
    down_right[:-1, :-1] = suffix[1:, 1:]

    # discordant partners of each cell: counts strictly down-left (k>i, l<j)
    suffix_rows_prefix_cols = np.cumsum(np.cumsum(values[::-1, :], axis=0)[::-1, :], axis=1)
    down_left = np.zeros_like(values)
    down_left[:-1, 1:] = suffix_rows_prefix_cols[1:, :-1]

    return float((values * down_right).sum()) - float((values * down_left).sum())


def rank_associations_from_counts(
    cd: float, n: float, untied_on_feature: float, untied_on_target: float, m: int
) -> dict[str, float | None]:
    """Assembles tau-b, tau-c and Somers' D from pre-computed pair counts.

    .. math::

        \\tau_b = \\frac{C - D}{\\sqrt{(P_0 - T_X)(P_0 - T_Y)}},
        \\qquad
        \\tau_c = \\frac{2 \\, m \\, (C - D)}{n^2 \\, (m - 1)},
        \\qquad
        D(Y \\mid X) = \\frac{C - D}{P_0 - T_X}

    with ``cd`` the :math:`C - D` count, ``untied_on_feature`` :math:`P_0 - T_X`,
    ``untied_on_target`` :math:`P_0 - T_Y`, and ``m`` the smaller of the number of
    non-empty rows and columns. ``tau_b`` matches :func:`scipy.stats.kendalltau`,
    ``tau_c`` applies Stuart's rectangular-table correction and ``somersd`` is the
    original asymmetric Somers' D ``D(Y|X)``.

    Shared by the closed form (:func:`rank_associations`) and the ordinal
    combination DP so both produce bit-identical values. Each measure is ``None``
    when its denominator vanishes.
    """
    denominator_b = math.sqrt(untied_on_feature * untied_on_target)
    return {
        "tau_b": cd / denominator_b if denominator_b > 0 else None,
        "tau_c": (2.0 * m * cd) / (n * n * (m - 1)) if m > 1 else None,
        "somersd": cd / untied_on_feature if untied_on_feature > 0 else None,
    }


def rank_associations(values: np.ndarray) -> dict[str, float | None]:
    """Kendall's tau-b, tau-c and Somers' D ``D(Y|X)`` for an ordered table.

    ``values`` is the ``(r, c)`` cell-count array with rows = ``X`` (feature
    groups) and columns = ``Y`` (target levels), both already in ascending order.
    Each measure is ``None`` when its denominator vanishes (degenerate table),
    mirroring the continuous evaluator's ``None`` convention.
    """
    n = float(values.sum())
    if n < 2:
        return {"tau_b": None, "tau_c": None, "somersd": None}

    cd = concordant_minus_discordant(values)
    row = values.sum(axis=1)
    col = values.sum(axis=0)
    all_pairs = n * (n - 1) / 2.0
    untied_on_feature = all_pairs - float((row * (row - 1) / 2.0).sum())
    untied_on_target = all_pairs - float((col * (col - 1) / 2.0).sum())
    m = min(int((row > 0).sum()), int((col > 0).sum()))
    return rank_associations_from_counts(cd, n, untied_on_feature, untied_on_target, m)
