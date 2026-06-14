"""Vectorized association kernels shared by the built-in measures.

These compute, in a handful of numpy/pandas operations, the same statistics the
scalar ``compute_association`` methods produce one feature at a time — but for
every column of a block at once. The scalar implementations remain the
reference: ``tests/selectors/test_vectorized_parity.py`` asserts these kernels
match them column-by-column.

Assumption: the target ``y`` carries no missing values (the normal case for a
selection target). The pooled Kruskal sample is then exactly each column's
non-NaN rows, matching ``scipy.stats.kruskal``.
"""

import numpy as np
import pandas as pd


def one_hot(groups: pd.Series) -> tuple[np.ndarray, int]:
    """One-hot ``(N, K)`` float matrix of a categorical series, plus ``K``."""
    codes, _ = pd.factorize(groups, use_na_sentinel=True)
    k = int(codes.max()) + 1 if codes.size and codes.max() >= 0 else 0
    matrix = np.zeros((codes.size, k), dtype=float)
    valid = codes >= 0
    matrix[np.arange(codes.size)[valid], codes[valid]] = 1.0
    return matrix, k


def _tie_factors(block: pd.DataFrame) -> np.ndarray:
    """``scipy.stats.tiecorrect`` factor per column over its non-NaN values."""
    factors = np.ones(block.shape[1])
    for j, col in enumerate(block.columns):
        values = block[col].to_numpy(dtype=float)
        values = values[~np.isnan(values)]
        n = values.size
        if n < 2:
            continue
        _, counts = np.unique(values, return_counts=True)
        ties = float((counts**3 - counts).sum())
        factors[j] = 1.0 - ties / (n**3 - n)
    return factors


def kruskal_h(block: pd.DataFrame, groups: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kruskal-Wallis ``H`` for every column of ``block`` grouped by ``groups``.

    Parameters
    ----------
    block : pd.DataFrame
        Quantitative feature block ``(N, P)``.
    groups : pd.Series
        Categorical grouping aligned to ``block`` rows (the target), shared by all columns.

    Returns
    -------
    H : np.ndarray ``(P,)``
        Tie-corrected H statistic, ``nan`` where undefined (a column with a
        single non-NaN value, or fewer than two populated groups).
    n_obs : np.ndarray ``(P,)``
        Pooled non-NaN observation count per column.
    n_groups : np.ndarray ``(P,)``
        Number of groups actually populated per column.
    """
    mask = block.notna().to_numpy()  # (N, P)
    ranks = np.where(mask, block.rank().to_numpy(), 0.0)  # (N, P), NaN ranks -> 0
    g_matrix, _ = one_hot(groups)  # (N, K)

    n = g_matrix.T @ mask.astype(float)  # (K, P) populated counts per group
    rank_sums = g_matrix.T @ ranks  # (K, P) summed ranks per group
    n_obs = mask.sum(0).astype(float)  # (P,)
    n_groups = (n > 0).sum(0)  # (P,)

    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.nansum(np.where(n > 0, rank_sums**2 / n, np.nan), axis=0)  # (P,)
        h = 12.0 / (n_obs * (n_obs + 1)) * term - 3.0 * (n_obs + 1)
        h = h / _tie_factors(block)

    # undefined where scipy.kruskal would raise / has_values would bail
    h = np.where((n_obs > 1) & (n_groups > 1), h, np.nan)
    return h, n_obs, n_groups.astype(float)


def kruskal_h_reversed(block: pd.DataFrame, y: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reversed Kruskal-Wallis ``H``: each qualitative column defines the groups,
    the continuous ``y`` is ranked (once) and split by those groups.

    Mirrors :func:`kruskal_h` but the grouping varies per column, so ``y`` is
    ranked a single time and each column's group sums are gathered with
    ``np.add.at``.
    """
    y_mask = y.notna().to_numpy()
    y_ranks_full = y.rank().to_numpy()
    p = block.shape[1]
    h = np.full(p, np.nan)
    n_obs = np.zeros(p)
    n_groups = np.zeros(p)

    for j, col in enumerate(block.columns):
        codes, _ = pd.factorize(block[col], use_na_sentinel=True)
        valid = (codes >= 0) & y_mask
        n = int(valid.sum())
        n_obs[j] = n
        if n < 2:
            continue
        col_codes = codes[valid]
        # re-rank y over the pooled valid subset (matches scipy on this column)
        ranks = pd.Series(y_ranks_full[valid]).rank().to_numpy()
        k = int(col_codes.max()) + 1
        rank_sums = np.zeros(k)
        counts = np.zeros(k)
        np.add.at(rank_sums, col_codes, ranks)
        np.add.at(counts, col_codes, 1.0)
        populated = counts > 0
        n_groups[j] = int(populated.sum())
        if n_groups[j] < 2:
            continue
        term = (rank_sums[populated] ** 2 / counts[populated]).sum()
        h_j = 12.0 / (n * (n + 1)) * term - 3.0 * (n + 1)
        # tie correction over the pooled y ranks
        _, tcounts = np.unique(ranks, return_counts=True)
        ties = float((tcounts**3 - tcounts).sum())
        if n**3 - n > 0:
            h_j /= 1.0 - ties / (n**3 - n)
        h[j] = h_j
    return h, n_obs, n_groups


def chi2_all(block: pd.DataFrame, y: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pearson chi² per qualitative column against categorical ``y``.

    Returns ``(chi2, n_obs, n_mod_x, n_mod_y)`` arrays of shape ``(P,)`` using
    ``bincount`` contingency tables instead of ``pd.crosstab`` +
    ``chi2_contingency``.
    """
    y_codes, y_cats = pd.factorize(y, use_na_sentinel=True)
    n_y = y_cats.size
    p = block.shape[1]
    chi2 = np.full(p, np.nan)
    n_obs = np.zeros(p)
    n_mod_x = np.zeros(p)

    for j, col in enumerate(block.columns):
        x_codes, x_cats = pd.factorize(block[col], use_na_sentinel=True)
        valid = (x_codes >= 0) & (y_codes >= 0)
        n = int(valid.sum())
        n_obs[j] = n
        m = x_cats.size
        n_mod_x[j] = m
        if n == 0 or m == 0 or n_y == 0:
            continue
        flat = x_codes[valid] * n_y + y_codes[valid]
        table = np.bincount(flat, minlength=m * n_y).reshape(m, n_y).astype(float)
        row = table.sum(1, keepdims=True)
        col_sum = table.sum(0, keepdims=True)
        expected = row @ col_sum / n
        with np.errstate(divide="ignore", invalid="ignore"):
            contrib = np.where(expected > 0, (table - expected) ** 2 / expected, 0.0)
        chi2[j] = contrib.sum()
    return chi2, n_obs, n_mod_x, np.full(p, float(n_y))
