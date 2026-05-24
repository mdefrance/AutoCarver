"""Module for binary combination evaluators."""

from abc import ABC
from collections.abc import Iterable, Iterator

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from tqdm import tqdm

from AutoCarver.combinations.binary.binary_target_rates import BinaryTargetRate, OddsRatio, TargetMean, Woe
from AutoCarver.combinations.utils.combination_evaluator import (
    AggregatedSample,
    CombinationEvaluator,
)
from AutoCarver.combinations.utils.combinations import combination_formatter
from AutoCarver.combinations.utils.target_rate import TargetRate

# Optional Rust kernel — see SPEEDUP_PLAN.md §7.
try:
    from AutoCarver._kernels import chi2_assoc_batch as _rust_chi2_assoc_batch  # ty: ignore[unresolved-import]

    _HAVE_RUST_CHI2 = True
except ImportError:  # pragma: no cover
    _HAVE_RUST_CHI2 = False


class BinaryCombinationEvaluator(CombinationEvaluator[pd.DataFrame], ABC):
    """Binary combination evaluator class."""

    is_y_binary = True
    _target_rate_classes: list[type[BinaryTargetRate]] = [TargetMean, OddsRatio, Woe]
    # narrow inherited attribute: binary evaluators always carry a BinaryTargetRate
    # (enforced by _init_target_rate).
    target_rate: BinaryTargetRate

    def _init_target_rate(self, target_rate: TargetRate[pd.DataFrame] | None) -> BinaryTargetRate:
        """Initializes target rate."""
        if target_rate is None:
            return TargetMean()
        elif not isinstance(target_rate, BinaryTargetRate):
            raise ValueError("target_rate must be a BinaryTargetRate")
        return target_rate

    def _association_measure(
        self,
        xagg: AggregatedSample | pd.Series | pd.DataFrame,
        n_obs: int | None = None,
        tol: float = 1e-10,
    ) -> dict[str, float | None]:
        """Computes measures of association between feature and target by crosstab.

        Used for the raw (one-shot) distribution. The hot per-combination loop
        goes through :meth:`_compute_associations`, which evaluates the same
        Pearson :math:`\\chi^2` in closed form (with Yates correction for 2×2
        tables, matching ``scipy.stats.chi2_contingency``) so the per-modality
        crosstab does not have to be rebuilt + handed to scipy on every
        combination.

        Parameters
        ----------
        xtab : pd.DataFrame
            Crosstab between feature and target.

        n_obs : int
            Sample total size.

        Returns
        -------
        dict[str, float]
            Cramér's V and Tschuprow's as a dict.
        """
        # number of values taken by the features
        n_mod_x = xagg.shape[0]

        # Chi2 statistic
        chi2 = chi2_contingency(xagg.values + tol)[0]

        # Cramér's V
        cramerv = np.sqrt(chi2 / n_obs)
        if pd.notna(cramerv):
            cramerv = round(cramerv / tol) * tol

        # Tschuprow's T
        tschuprowt = cramerv / np.sqrt(np.sqrt(n_mod_x - 1))
        if pd.notna(tschuprowt):
            tschuprowt = round(tschuprowt / tol) * tol

        return {"cramerv": cramerv, "tschuprowt": tschuprowt}

    def _grouper(self, xagg: AggregatedSample, groupby: dict) -> pd.DataFrame:
        """Groups a crosstab by groupby and sums column values by groups (vectorized)

        Parameters
        ----------
        xagg : pd.DataFrame
            crosstab between X and y
        groupby : list[str]
            indices to group by

        Returns
        -------
        DataFrame
            Crosstab grouped by indices
        """
        # all indices that may be duplicated
        index_values = np.array([groupby.get(index_value, index_value) for index_value in xagg.index])

        # all unique indices deduplicated
        unique_indices = np.unique(index_values)

        # initiating summed up array with zeros
        summed_values = np.zeros((len(unique_indices), len(xagg.columns)))

        # for each unique_index found in index_values sums xtab.Values at corresponding position
        # in summed_values
        np.add.at(summed_values, np.searchsorted(unique_indices, index_values), xagg.values)

        # converting back to dataframe
        return pd.DataFrame(summed_values, index=unique_indices, columns=xagg.columns)

    def _group_xagg_by_combinations(self, combinations: Iterable[list]) -> Iterator[dict]:
        """Streams combinations *without* building the per-combination crosstab.

        The closed-form chi² in :meth:`_compute_associations` only needs
        ``index_to_groupby`` plus precomputed per-raw-modality counts. The
        per-combination crosstab is rebuilt lazily, on demand, in
        :meth:`_test_viability_train` for the handful of top combinations
        actually checked for viability.
        """
        for combination in combinations:
            yield {
                "combination": combination,
                "index_to_groupby": combination_formatter(combination),
            }

    def _compute_associations(self, grouped_xaggs: Iterable[dict]) -> Iterator[dict]:
        """Closed-form, streaming chi² (Cramér's V + Tschuprow's T) across combinations.

        Statistically identical to ``scipy.stats.chi2_contingency`` with default
        parameters (Pearson :math:`\\chi^2`, Yates correction for 2×2 tables);
        the only difference is that per-raw-modality counts ``(n_0, n_1)`` are
        gathered **once** from :attr:`self.samples.train.xagg` and aggregated
        into per-combination groups in batches via **one BLAS matmul per batch**.

        Combinations are processed in batches of :data:`_CHI2_BATCH_SIZE`.
        For each batch:

        * the per-combination group assignment is encoded as a 0/1 matrix
          ``A_c`` of shape ``(max_g, n_mod)`` and the batch is stacked into a
          single ``(B, max_g, n_mod)`` tensor ``A``;
        * per-group counts ``(n_{0,g}, n_{1,g})`` are obtained in one BLAS
          call as ``A @ n0_per_mod`` and ``A @ n1_per_mod``;
        * ``tol`` is added to every in-range cell (matching the historical
          ``xagg.values + tol`` shift);
        * expected frequencies ``E_{gc} = R_g \\cdot C_c / N`` are computed
          via broadcast outer product;
        * Yates correction is applied iff the combination's table is 2×2
          (matching scipy's default);
        * :math:`\\chi^2 = \\sum_{g,c} (O_{gc} - E_{gc})^2 / E_{gc}`;
        * Cramér's V :math:`= \\sqrt{\\chi^2 / N_{\\text{obs}}}` and
          Tschuprow's T :math:`= V / \\sqrt[4]{k-1}`, both rounded to ``tol``
          decimals as before.

        Yields light ``{combination, index_to_groupby, cramerv, tschuprowt}``
        dicts in arrival order — sorting happens in
        :meth:`CombinationEvaluator._get_best_association`.
        """
        raw_xagg = self.samples.train.xagg
        # Per-raw-modality (n0, n1), in the order of raw_xagg.index
        n0_per_mod = raw_xagg.iloc[:, 0].to_numpy(dtype=float)
        n1_per_mod = raw_xagg.iloc[:, 1].to_numpy(dtype=float)
        n_obs = float(n0_per_mod.sum() + n1_per_mod.sum())

        mod_to_pos: dict = {m: i for i, m in enumerate(raw_xagg.index)}
        n_mod = len(mod_to_pos)

        tol = 1e-10

        batch_fn = _chi2_assoc_batch_rust if _HAVE_RUST_CHI2 else _chi2_assoc_batch

        batch: list[dict] = []
        for grouped_xagg in tqdm(grouped_xaggs, desc="Computing associations", disable=not self.verbose):
            batch.append(grouped_xagg)
            if len(batch) >= _CHI2_BATCH_SIZE:
                yield from batch_fn(
                    batch=batch,
                    n0_per_mod=n0_per_mod,
                    n1_per_mod=n1_per_mod,
                    n_obs=n_obs,
                    mod_to_pos=mod_to_pos,
                    n_mod=n_mod,
                    tol=tol,
                )
                batch = []
        if batch:
            yield from batch_fn(
                batch=batch,
                n0_per_mod=n0_per_mod,
                n1_per_mod=n1_per_mod,
                n_obs=n_obs,
                mod_to_pos=mod_to_pos,
                n_mod=n_mod,
                tol=tol,
            )


class TschuprowtCombinations(BinaryCombinationEvaluator):
    """Tschuprow's T based combination evaluation toolkit"""

    sort_by = "tschuprowt"


class CramervCombinations(BinaryCombinationEvaluator):
    """Cramér's V based combination evaluation toolkit"""

    sort_by = "cramerv"


# Number of combinations processed per batched matmul call. Trades peak RAM
# (``B * max_g * n_mod`` doubles for the assignment tensor) for amortized
# Python overhead — same trade-off as :data:`_KRUSKAL_BATCH_SIZE` for the
# continuous path.
_CHI2_BATCH_SIZE = 1024


# ---------------------------------------------------------------------------
# Closed-form chi^2 helpers (binary / 2-column contingency tables)
# ---------------------------------------------------------------------------


def _chi2_assoc_for_combination(
    *,
    n0_per_mod: np.ndarray,
    n1_per_mod: np.ndarray,
    n_obs: float,
    mod_to_pos: dict,
    n_mod: int,
    index_to_groupby: dict,
    tol: float,
) -> tuple[float, float]:
    """Closed-form Cramér's V & Tschuprow's T for one combination.

    Mirrors ``BinaryCombinationEvaluator._association_measure`` bit-for-bit
    (including the ``+ tol`` shift and the ``round(x / tol) * tol`` rounding)
    so the values produced here match the historical scipy-based ones to the
    last digit of the rounded representation.
    """
    # Build integer group assignment for this combination
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

    # Modalities present in raw_xagg but not in index_to_groupby become their
    # own singleton groups — matches the legacy `_grouper`'s
    # `groupby.get(iv, iv)` fallback (relevant in edge-case fixtures where the
    # crosstab carries a nan row but the feature claims has_nan=False).
    for pos in range(n_mod):
        if not assigned[pos]:
            leader_to_grp[("__unmapped__", pos)] = len(leader_to_grp)
            assign[pos] = leader_to_grp[("__unmapped__", pos)]

    n_groups = len(leader_to_grp)

    # Per-group (n0, n1) via bincount with weights (vectorised)
    n0_g = np.bincount(assign, weights=n0_per_mod, minlength=n_groups)
    n1_g = np.bincount(assign, weights=n1_per_mod, minlength=n_groups)

    # Build the (k, 2) observed table and add tol to every cell (matches existing scipy call)
    obs = np.stack((n0_g, n1_g), axis=1) + tol  # shape (n_groups, 2)

    chi2 = _chi2_pearson_2col(obs)

    cramerv = float(np.sqrt(chi2 / n_obs))
    if pd.notna(cramerv):
        cramerv = round(cramerv / tol) * tol

    if n_groups > 1:
        tschuprowt = cramerv / float(np.sqrt(np.sqrt(n_groups - 1)))
        if pd.notna(tschuprowt):
            tschuprowt = round(tschuprowt / tol) * tol
    else:
        tschuprowt = cramerv

    return cramerv, tschuprowt


def _chi2_pearson_2col(obs: np.ndarray) -> float:
    """Pearson :math:`\\chi^2` for a (k, 2) observed contingency table.

    Replicates :func:`scipy.stats.chi2_contingency` defaults:

    * expected frequencies via the outer product of marginals divided by N
      (same as :func:`scipy.stats.contingency.expected_freq`);
    * Yates correction iff the table is exactly 2×2 (same threshold scipy uses).
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


def _chi2_assoc_batch(
    *,
    batch: list[dict],
    n0_per_mod: np.ndarray,
    n1_per_mod: np.ndarray,
    n_obs: float,
    mod_to_pos: dict,
    n_mod: int,
    tol: float,
) -> Iterator[dict]:
    """Batched closed-form Cramér's V & Tschuprow's T via matmul.

    Encodes each combination's group assignment as a ``(max_g, n_mod)`` 0/1
    matrix stacked into ``A`` of shape ``(B, max_g, n_mod)``. Per-group
    ``(n0, n1)`` is obtained in one BLAS call as ``A @ n0_per_mod`` and
    ``A @ n1_per_mod``. Mirrors :func:`_chi2_assoc_for_combination` cell-for-cell
    (``+tol`` shift, Yates correction iff ``n_groups == 2``, ``round / tol``
    quantisation), so for ``B == 1`` the floats are bit-identical to the
    scalar path — preserving the pinned values in the parity tests.
    """
    B = len(batch)

    # Build integer assignment matrix `assign[b, pos] = group_id`, plus
    # n_groups[b]. Unmapped positions become their own singleton groups —
    # matches :func:`_chi2_assoc_for_combination` exactly.
    assign = np.empty((B, n_mod), dtype=np.intp)
    n_groups = np.empty(B, dtype=np.intp)
    for b, item in enumerate(batch):
        leader_to_grp: dict = {}
        assigned = np.zeros(n_mod, dtype=bool)
        for mod, leader in item["index_to_groupby"].items():
            gid = leader_to_grp.get(leader)
            if gid is None:
                gid = len(leader_to_grp)
                leader_to_grp[leader] = gid
            pos = mod_to_pos[mod]
            assign[b, pos] = gid
            assigned[pos] = True
        for pos in range(n_mod):
            if not assigned[pos]:
                assign[b, pos] = len(leader_to_grp)
                leader_to_grp[("__unmapped__", pos)] = len(leader_to_grp)
        n_groups[b] = len(leader_to_grp)

    max_g = int(n_groups.max())

    # 0/1 assignment tensor: A[b, g, m] = 1 iff modality m is in group g of combination b
    A = np.zeros((B, max_g, n_mod), dtype=np.float64)
    b_idx = np.repeat(np.arange(B), n_mod)
    mod_idx = np.tile(np.arange(n_mod), B)
    A[b_idx, assign.ravel(), mod_idx] = 1.0

    # Batched grouped (n0, n1) via BLAS matmul
    n0_g = A @ n0_per_mod  # (B, max_g)
    n1_g = A @ n1_per_mod  # (B, max_g)

    # In-range mask: cells beyond n_groups[b] are padding (kept at 0)
    in_range_mask = np.arange(max_g)[None, :] < n_groups[:, None]  # (B, max_g)

    # Observed table per combination with ``+tol`` shift on in-range cells only
    obs = np.stack((n0_g, n1_g), axis=-1)  # (B, max_g, 2)
    obs = np.where(in_range_mask[..., None], obs + tol, obs)

    # Marginals (padding rows/cols are 0)
    R = obs.sum(axis=2)  # (B, max_g)
    C = obs.sum(axis=1)  # (B, 2)
    N_per = obs.sum(axis=(1, 2))  # (B,)

    # Expected: E[b, g, c] = R[b, g] * C[b, c] / N[b]
    expected = R[:, :, None] * C[:, None, :] / N_per[:, None, None]  # (B, max_g, 2)

    # Yates correction for combinations whose table is exactly 2x2
    diff = expected - obs
    magnitude = np.minimum(0.5, np.abs(diff))
    yates_shift = magnitude * np.sign(diff)
    is_2group = n_groups == 2
    yates_apply = is_2group[:, None, None] & in_range_mask[..., None]
    obs_corrected = np.where(yates_apply, obs + yates_shift, obs)

    # Per-cell chi². Padding cells: expected==0 → 0/0 = NaN, zeroed via mask.
    with np.errstate(invalid="ignore", divide="ignore"):
        cells = (obs_corrected - expected) ** 2 / expected
    cells = np.where(in_range_mask[..., None], cells, 0.0)
    chi2 = cells.sum(axis=(1, 2))  # (B,)

    # Cramér's V & Tschuprow's T, with the same ``round(x / tol) * tol`` quantisation
    cramerv = np.sqrt(chi2 / n_obs)
    cramerv_q = np.where(np.isnan(cramerv), cramerv, np.round(cramerv / tol) * tol)

    # tschuprowt = cramerv_q / (n_groups - 1) ** 0.25 if n_groups > 1 else cramerv_q
    n_groups_f = n_groups.astype(np.float64)
    denom = np.where(n_groups_f > 1, np.sqrt(np.sqrt(np.maximum(n_groups_f - 1.0, 1.0))), 1.0)
    tt_raw = np.where(n_groups_f > 1, cramerv_q / denom, cramerv_q)
    tt_q = np.where(np.isnan(tt_raw), tt_raw, np.round(tt_raw / tol) * tol)

    for b, item in enumerate(batch):
        yield {
            "combination": item["combination"],
            "index_to_groupby": item["index_to_groupby"],
            "cramerv": float(cramerv_q[b]),
            "tschuprowt": float(tt_q[b]),
        }


def _chi2_assoc_batch_rust(
    *,
    batch: list[dict],
    n0_per_mod: np.ndarray,
    n1_per_mod: np.ndarray,
    n_obs: float,
    mod_to_pos: dict,
    n_mod: int,
    tol: float,
) -> Iterator[dict]:
    """Rust-backed equivalent of :func:`_chi2_assoc_batch`. Yields per-combination
    ``{combination, index_to_groupby, cramerv, tschuprowt}`` in arrival order with
    values quantised to ``tol`` precision (matches the NumPy path bit-for-bit on
    the parity fixtures)."""
    py_idx2gb = [item["index_to_groupby"] for item in batch]
    cramerv, tschuprowt, _ = _rust_chi2_assoc_batch(
        py_idx2gb,
        mod_to_pos,
        n_mod,
        n0_per_mod.astype(np.float64, copy=False),
        n1_per_mod.astype(np.float64, copy=False),
        float(n_obs),
        float(tol),
    )

    for b, item in enumerate(batch):
        yield {
            "combination": item["combination"],
            "index_to_groupby": item["index_to_groupby"],
            "cramerv": float(cramerv[b]),
            "tschuprowt": float(tschuprowt[b]),
        }
