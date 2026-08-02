"""Helpers shared by the four combination-evaluator families' interval DPs.

Everything here is a verbatim relocation from the binary/continuous/ordinal/multiclass
evaluator modules — no arithmetic was edited, only the duplication removed.
"""

import math
from collections.abc import Callable, Iterator

import numpy as np
import pandas as pd

from AutoCarver.combinations.utils.combinations import combination_formatter


def dp_inputs_from_xagg(raw_xagg: pd.DataFrame, raw_index: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aligns a raw crosstab to ``raw_index`` for the DP.

    Returns ``(M, n_per_mod, col_sums)`` where ``M`` is the ``(len(raw_index), c)``
    per-modality column-count matrix (rows absent from ``raw_xagg`` are zero),
    ``n_per_mod`` the row totals and ``col_sums`` the target marginal.
    """
    position = {label: i for i, label in enumerate(raw_xagg.index)}
    values = np.asarray(raw_xagg.values, dtype=float)
    M = np.zeros((len(raw_index), values.shape[1]))
    for row, label in enumerate(raw_index):
        source = position.get(label)
        if source is not None:
            M[row] = values[source]
    return M, M.sum(axis=1), M.sum(axis=0)


def sort_key(value: float | None) -> float:
    """Sort key putting ``None`` / ``NaN`` metrics last (descending sort)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("-inf")
    return float(value)


def chi2_pearson(obs: np.ndarray) -> float:
    """Pearson :math:`\\chi^2` for a ``(B, C)`` observed contingency table.

    Replicates :func:`scipy.stats.chi2_contingency` defaults: expected
    frequencies via the outer product of marginals divided by N, with Yates
    correction iff the table is exactly 2x2 (matches scipy's own threshold).
    Shared by the binary (``C=2``) and multiclass (``C=K``) chi² paths.
    """
    R = obs.sum(axis=1)
    C = obs.sum(axis=0)
    N = float(obs.sum())
    expected = np.outer(R, C) / N

    if obs.shape == (2, 2):
        diff = expected - obs
        direction = np.sign(diff)
        magnitude = np.minimum(0.5, np.abs(diff))
        obs = obs + magnitude * direction

    return float(((obs - expected) ** 2 / expected).sum())


def _dp_base_row(
    n_mod: int, seg_cost: Callable[[int, int], float], skip_cost: float | None
) -> list[list[tuple[float, tuple[int, ...]]]]:
    """dp[1][j]: the single-group partition ``[0, j)``, or empty when it hits ``skip_cost``."""
    row: list[list[tuple[float, tuple[int, ...]]]] = [[] for _ in range(n_mod + 1)]
    for j in range(1, n_mod + 1):
        c = seg_cost(0, j)
        if skip_cost is None or c != skip_cost:
            row[j] = [(c, (0, j))]
    return row


def _dp_next_row(
    prev_row: list[list[tuple[float, tuple[int, ...]]]],
    *,
    g: int,
    n_mod: int,
    seg_cost: Callable[[int, int], float],
    skip_cost: float | None,
    top_k: int,
    maximize: bool,
) -> list[list[tuple[float, tuple[int, ...]]]]:
    """dp[g][j]: best top-``top_k`` ``g``-group partitions of ``[0, j)``, built from ``dp[g-1]``."""
    row: list[list[tuple[float, tuple[int, ...]]]] = [[] for _ in range(n_mod + 1)]
    for j in range(g, n_mod + 1):
        candidates = _segment_candidates(prev_row, j=j, g=g, seg_cost=seg_cost, skip_cost=skip_cost)
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=maximize)
            row[j] = candidates[:top_k]
    return row


def _segment_candidates(
    prev_row: list[list[tuple[float, tuple[int, ...]]]],
    *,
    j: int,
    g: int,
    seg_cost: Callable[[int, int], float],
    skip_cost: float | None,
) -> list[tuple[float, tuple[int, ...]]]:
    """All ``dp[g-1][i] + seg_cost(i, j)`` extensions, skipping excluded segments."""
    candidates: list[tuple[float, tuple[int, ...]]] = []
    for i in range(g - 1, j):
        c = seg_cost(i, j)
        if skip_cost is not None and c == skip_cost:
            continue
        for prev_s, prev_splits in prev_row[i]:
            candidates.append((prev_s + c, prev_splits + (j,)))
    return candidates


def top_k_partitions(
    *,
    n_mod: int,
    cap: int,
    seg_cost: Callable[[int, int], float],
    top_k: int,
    maximize: bool = True,
    skip_cost: float | None = None,
) -> list[tuple[int, float, tuple[int, ...]]]:
    """Top-``top_k`` consecutive partitions of ``range(n_mod)`` into 2..``cap`` groups.

    ``seg_cost(i, j)`` is the additive cost of the segment ``[i, j)``. Returns
    ``(k, total_cost, splits)`` triples where ``splits = (0, s_1, ..., s_{k-1}, n_mod)``,
    one list per ``k`` merged and **not** globally sorted (callers translate cost to their
    metric before the final sort). ``skip_cost`` (the continuous path's ``-inf`` for empty
    segments) excludes a segment from the recurrence entirely.
    """
    rows: dict[int, list[list[tuple[float, tuple[int, ...]]]]] = {1: _dp_base_row(n_mod, seg_cost, skip_cost)}
    for g in range(2, cap + 1):
        rows[g] = _dp_next_row(
            rows[g - 1], g=g, n_mod=n_mod, seg_cost=seg_cost, skip_cost=skip_cost, top_k=top_k, maximize=maximize
        )

    out: list[tuple[int, float, tuple[int, ...]]] = []
    for k in range(2, cap + 1):
        for cost, splits in rows[k][n_mod]:
            out.append((k, cost, splits))
    return out


def compact_empty_modalities(M: np.ndarray, n_per_mod: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drops all-zero rows of ``M`` before running an interval DP.

    Empty modalities carry no observations and must never form their own group in a
    chi²/tau-c DP (an empty group shifts the effective group/row count that per-k
    denominators depend on). Returns ``(keep, kept_M, kept_n_per_mod)`` where ``keep``
    maps compacted positions back to raw positions.
    """
    keep = np.flatnonzero(n_per_mod > 0)
    return keep, M[keep], n_per_mod[keep]


def splits_to_combination(splits: tuple[int, ...], raw_index: list, keep: np.ndarray | None = None) -> list[list]:
    """Maps DP split points back to raw-label groups.

    With ``keep`` (compacted DP), each cut sits just before the first kept modality of
    the next group, so an empty modality attaches to the preceding group (leading
    empties join the first group, trailing empties the last). Without it, ``splits``
    already index directly into ``raw_index``.
    """
    bounds = list(splits) if keep is None else [0, *(int(keep[s]) for s in splits[1:-1]), len(raw_index)]
    return [list(raw_index[bounds[g] : bounds[g + 1]]) for g in range(len(bounds) - 1)]


def build_group_assignment(index_to_groupby: dict, mod_to_pos: dict, n_mod: int) -> tuple[np.ndarray, int]:
    """Maps each raw modality position to an integer group id, in first-appearance order.

    Modalities present in the xagg but absent from ``index_to_groupby`` become their own
    singleton groups — matching the legacy ``_grouper``'s ``groupby.get(iv, iv)`` fallback.
    Returns ``(assign, n_groups)`` where ``assign`` has shape ``(n_mod,)``.
    """
    leader_to_grp: dict = {}
    assign = np.empty(n_mod, dtype=np.intp)
    assigned = np.zeros(n_mod, dtype=bool)
    for mod, leader in index_to_groupby.items():
        gid = leader_to_grp.get(leader)
        if gid is None:
            gid = len(leader_to_grp)
            leader_to_grp[leader] = gid
        pos = mod_to_pos[mod]
        assign[pos] = gid
        assigned[pos] = True
    for pos in range(n_mod):
        if not assigned[pos]:
            assign[pos] = len(leader_to_grp)
            leader_to_grp[("__unmapped__", pos)] = len(leader_to_grp)
    return assign, len(leader_to_grp)


def nan_fanout_variants(
    base_partitions: list[dict],
    nan_label: str,
    raw_labels: list,
    max_n_mod: int,
) -> Iterator[list[list]]:
    """Yields NaN-augmented variants of each base consecutive partition.

    Mirrors :func:`AutoCarver.combinations.utils.combinations.nan_combinations`
    semantics: for every base combination, fold ``nan_label`` into each group;
    add it as its own group iff ``len(base) < max_n_mod``; finally yield the
    ``[list(raw_labels), [nan_label]]`` partition once. Subclass-specific
    scoring (Kruskal H / chi²) happens in the caller.
    """
    for base in base_partitions:
        base_combo = base["combination"]
        for j in range(len(base_combo)):
            variant = [g[:] for g in base_combo]
            variant[j] = variant[j] + [nan_label]
            yield variant
        if len(base_combo) < max_n_mod:
            yield [g[:] for g in base_combo] + [[nan_label]]
    yield [list(raw_labels), [nan_label]]


def score_nan_variants(
    *,
    base_partitions: list[dict],
    nan_label: str,
    raw_labels: list,
    max_n_mod: int,
    scorer: Callable[[dict], dict],
    sort_by: str,
) -> list[dict]:
    """Fan each base partition out across NaN placements, score, sort by ``sort_by`` desc.

    ``scorer(index_to_groupby) -> {metric: value, ...}`` computes the family-specific
    association measure(s) for one variant.
    """
    scored: list[dict] = []
    for variant in nan_fanout_variants(base_partitions, nan_label, raw_labels, max_n_mod):
        index_to_groupby = combination_formatter(variant)
        scored.append({"combination": variant, "index_to_groupby": index_to_groupby, **scorer(index_to_groupby)})
    scored.sort(key=lambda a: sort_key(a[sort_by]), reverse=True)
    return scored
