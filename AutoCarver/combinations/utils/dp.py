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
