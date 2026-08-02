"""Module for binary combination evaluators."""

from abc import ABC

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from AutoCarver.combinations.binary.binary_target_rates import BinaryTargetRate, OddsRatio, TargetMean, Woe
from AutoCarver.combinations.utils.combination_evaluator import AggregatedSample, CombinationEvaluator
from AutoCarver.combinations.utils.combinations import combination_formatter
from AutoCarver.combinations.utils.dp import (
    build_group_assignment,
    score_nan_variants,
    splits_to_combination,
    top_k_partitions,
)
from AutoCarver.combinations.utils.target_rate import TargetRate
from AutoCarver.features import GroupedList
from AutoCarver.stats.chi2 import cramerv_tschuprowt
from AutoCarver.stats.chi2 import pearson_chi2 as _chi2_pearson_2col


class BinaryCombinationEvaluator(CombinationEvaluator[pd.DataFrame], ABC):
    """Binary combination evaluator class."""

    is_y_binary = True
    _target_rate_classes: list[type[BinaryTargetRate]] = [TargetMean, OddsRatio, Woe]
    # narrow inherited attribute: binary evaluators always carry a BinaryTargetRate
    # (enforced by _init_target_rate).
    target_rate: BinaryTargetRate
    # narrow inherited `sort_by: str | None`: concrete binary subclasses
    # (TschuprowtCombinations, CramervCombinations) always set this to a str.
    sort_by: str

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

        # Tschuprow's T (undefined for a single modality: avoids a divide-by-zero)
        if n_mod_x <= 1:
            tschuprowt = np.nan
        else:
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
        # group leader per raw modality (raw modalities are in ordinal order)
        index_values = [groupby.get(index_value, index_value) for index_value in xagg.index]

        # unique leaders ordered by first appearance (ordinal order), not by label
        # text: keeps grouping independent of the cosmetic label strings so the
        # order-sensitive viability tests see the feature's natural ordering.
        group_of: dict = {}
        for leader in index_values:
            if leader not in group_of:
                group_of[leader] = len(group_of)
        unique_indices = list(group_of)

        # summing each raw modality's row into its group's position
        positions = np.fromiter((group_of[leader] for leader in index_values), dtype=np.intp, count=len(index_values))
        summed_values = np.zeros((len(unique_indices), len(xagg.columns)))
        np.add.at(summed_values, positions, xagg.values)

        # converting back to dataframe
        return pd.DataFrame(summed_values, index=unique_indices, columns=xagg.columns)

    def _get_best_combination_non_nan(self) -> dict | None:
        """DP-based override with progressive top-K.

        Replaces ``consecutive_combinations + _compute_associations`` with the
        interval-DP in :func:`_top_k_partitions_chi2_dp`, which returns the
        top-K consecutive partitions ranked by ``self.sort_by`` desc.

        **Progressive search.** Starts with ``top_k = self.dp_top_k_initial``.
        If the viability walk doesn't find a viable candidate within that
        top-K and escalation is enabled (``dp_escalate``), grows ``top_k`` ×4
        and re-runs DP — walking only the new entries from where we left off.
        Repeats until either a viable is found or DP exhausts every consecutive
        partition (signalled by ``len(result) < top_k``). Total work bounded by
        ~1.33× a single DP run at the final top_k.

        This makes the search **exhaustive in the worst case**, matching the
        legacy enumerate-and-score path's correctness while keeping the
        common case (viable found in top ~100) essentially free. Mirrors
        :meth:`ContinuousCombinationEvaluator._get_best_combination_non_nan`.

        The NaN-fan-out path (:meth:`_get_best_combination_with_nan`) still
        goes through the legacy enumerate-and-score loop.
        """
        feature_labels = self.feature.labels
        if feature_labels is None:
            raise RuntimeError(f"[{self.__name__}] feature labels are not populated")
        raw_labels = GroupedList(feature_labels[:])

        if self.feature.has_nan:
            if self.feature.dropna:
                raw_labels.remove(self.feature.nan)
            self.samples.dropna(self.feature.nan)

        self._historize_raw_combination()

        if self.samples.train.shape[0] <= 1:
            return None

        # Iterate over raw_labels (mirrors the parent's
        # ``consecutive_combinations(raw_labels, ...)`` enumeration). When
        # raw_labels and raw_xagg.index diverge (edge-case fixtures with
        # has_nan=False + nan row in xagg, or has_nan=True/dropna=False),
        # rows present in raw_xagg but not in raw_labels are excluded; labels
        # present in raw_labels but not in raw_xagg get zero counts — matching
        # the legacy ``_grouper``'s ``groupby.get(idx, idx)`` semantics where
        # an unmapped row produces no contribution.
        raw_xagg = self.samples.train.xagg
        all_n0 = raw_xagg.iloc[:, 0].to_numpy(dtype=float)
        all_n1 = raw_xagg.iloc[:, 1].to_numpy(dtype=float)
        xagg_pos = {m: i for i, m in enumerate(raw_xagg.index)}
        raw_index = list(raw_labels)
        n0_per_mod = np.fromiter(
            (all_n0[xagg_pos[m]] if m in xagg_pos else 0.0 for m in raw_index),
            dtype=float,
            count=len(raw_index),
        )
        n1_per_mod = np.fromiter(
            (all_n1[xagg_pos[m]] if m in xagg_pos else 0.0 for m in raw_index),
            dtype=float,
            count=len(raw_index),
        )

        # Progressive top-K with ×4 growth (shared driver). See docstring.
        viable = self._search_escalating(
            lambda top_k: _top_k_partitions_chi2_dp(
                n0_per_mod,
                n1_per_mod,
                max_n_mod=self.max_n_mod,
                raw_index=raw_index,
                sort_by=self.sort_by,
                top_k=top_k,
            )
        )
        self._apply_best_combination(viable)
        return viable

    def _get_best_combination_with_nan(self, best_combination: dict | None) -> dict | None:
        """DP-based override with NaN fan-out.

        Mirrors :meth:`ContinuousCombinationEvaluator._get_best_combination_with_nan`:

        1. DP top-K base consecutive partitions over the non-nan labels
           (:func:`_top_k_partitions_chi2_dp` on a restricted view of the
           per-modality ``(n0, n1)`` counts);
        2. fan each base out across NaN placements exactly like
           :func:`nan_combinations` (nan folded into each group, then nan
           as its own group when ``len(base) < max_n_mod``, plus the final
           ``[all_non_nan, [nan]]`` partition);
        3. re-score every variant in closed form with
           :func:`_chi2_assoc_for_combination` against the **full** per-modality
           counts (the nan row is in ``samples.train.xagg`` because
           :meth:`_get_best_combination_non_nan`'s ``_apply_best_combination``
           rebuilt it from raw);
        4. walk the sorted variants for the first viable, with progressive
           top-K ×4 growth on the base DP — dedup'd via a per-partition seen
           set so combinations carried over from a smaller ``top_k`` are not
           re-tested / re-historized.

        Falls back to the parent implementation when the guard condition
        (``self.dropna and feature.has_nan``) is not met — matches the legacy
        short-circuit behaviour. Runs even when the non-nan search failed
        (``best_combination is None``): the nan row is restored from raw
        before reading per-modality counts below.
        """
        if not (self.dropna and self.feature.has_nan):
            return super()._get_best_combination_with_nan(best_combination)

        # non-nan search failed -> xaggs are still nan-filtered; restore the nan
        # row before reading per-modality counts below
        if best_combination is None:
            self.samples.restore_raw()

        if self.verbose:
            print(f"[{self.__name__}] Grouping NaNs")

        feature_labels = self.feature.labels
        if feature_labels is None:
            raise RuntimeError(f"[{self.__name__}] feature labels are not populated")
        raw_labels = GroupedList(feature_labels[:])
        raw_labels.remove(self.feature.nan)
        nan_label = self.feature.nan

        # Full per-modality (n0, n1) — nan row is in xagg because
        # _apply_best_combination on the non-nan winner rebuilt it from raw.
        raw_xagg = self.samples.train.xagg
        n0_per_mod = raw_xagg.iloc[:, 0].to_numpy(dtype=float)
        n1_per_mod = raw_xagg.iloc[:, 1].to_numpy(dtype=float)
        n_obs = float(n0_per_mod.sum() + n1_per_mod.sum())
        mod_to_pos: dict = {m: i for i, m in enumerate(raw_xagg.index)}
        n_mod = len(mod_to_pos)
        tol = 1e-10

        # Non-nan subset, aligned to raw_labels order, for the base DP.
        non_nan_index = list(raw_labels)
        n0_non_nan = np.fromiter(
            (n0_per_mod[mod_to_pos[m]] for m in non_nan_index),
            dtype=float,
            count=len(non_nan_index),
        )
        n1_non_nan = np.fromiter(
            (n1_per_mod[mod_to_pos[m]] for m in non_nan_index),
            dtype=float,
            count=len(non_nan_index),
        )

        def _run_round(top_k: int) -> tuple[list[dict], int]:
            base_partitions = _top_k_partitions_chi2_dp(
                n0_non_nan,
                n1_non_nan,
                max_n_mod=self.max_n_mod,
                raw_index=non_nan_index,
                sort_by=self.sort_by,
                top_k=top_k,
                tol=tol,
            )

            def _scorer(index_to_groupby: dict) -> dict:
                cv, tt = _chi2_assoc_for_combination(
                    n0_per_mod=n0_per_mod,
                    n1_per_mod=n1_per_mod,
                    n_obs=n_obs,
                    mod_to_pos=mod_to_pos,
                    n_mod=n_mod,
                    index_to_groupby=index_to_groupby,
                    tol=tol,
                )
                return {"cramerv": cv, "tschuprowt": tt}

            scored = score_nan_variants(
                base_partitions=base_partitions,
                nan_label=nan_label,
                raw_labels=non_nan_index,
                max_n_mod=self.max_n_mod,
                scorer=_scorer,
                sort_by=self.sort_by,
            )
            return scored, len(base_partitions)

        viable = self._search_escalating_nan(_run_round)
        self._apply_best_combination(viable)
        return viable


class TschuprowtCombinations(BinaryCombinationEvaluator):
    """Tschuprow's T based combination evaluation toolkit.

    Search uses :ref:`progressive top-K interval DP <DPChi2>` over the
    closed-form Pearson :math:`\\chi^2` decomposition (per-k DP with constant
    column marginals, Yates correction iff ``k == 2``). Statistically equivalent
    to :func:`scipy.stats.chi2_contingency` — bit-exact agreement pinned by
    parity tests.
    """

    sort_by = "tschuprowt"


class CramervCombinations(BinaryCombinationEvaluator):
    """Cramér's V based combination evaluation toolkit.

    Same DP search as :class:`TschuprowtCombinations` (see :ref:`DPChi2`); only
    the ``sort_by`` key differs. :math:`V = \\sqrt{\\chi^2 / N_{obs}}` is a
    monotone transform of :math:`\\chi^2` at fixed :math:`k`.
    """

    sort_by = "cramerv"


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
    # Build integer group assignment for this combination. Modalities present in
    # raw_xagg but not in index_to_groupby become their own singleton groups —
    # matches the legacy `_grouper`'s `groupby.get(iv, iv)` fallback (relevant in
    # edge-case fixtures where the crosstab carries a nan row but the feature
    # claims has_nan=False).
    assign, n_groups = build_group_assignment(index_to_groupby, mod_to_pos, n_mod)

    # Per-group (n0, n1) via bincount with weights (vectorised)
    n0_g = np.bincount(assign, weights=n0_per_mod, minlength=n_groups)
    n1_g = np.bincount(assign, weights=n1_per_mod, minlength=n_groups)

    # Build the (k, 2) observed table and add tol to every cell (matches existing scipy call)
    obs = np.stack((n0_g, n1_g), axis=1) + tol  # shape (n_groups, 2)

    chi2 = _chi2_pearson_2col(obs)

    return cramerv_tschuprowt(chi2, n_obs, n_groups, 2, tol=tol)


def _top_k_partitions_chi2_dp(  # noqa: C901
    n0_per_mod: np.ndarray,
    n1_per_mod: np.ndarray,
    *,
    max_n_mod: int,
    raw_index: list,
    sort_by: str = "tschuprowt",
    top_k: int = 1000,
    tol: float = 1e-10,
) -> list[dict]:
    """Top-K consecutive-segmentation partitions ranked by a chi²-derived metric.

    Binary analogue of
    :func:`AutoCarver.combinations.continuous.continuous_combination_evaluators._top_k_partitions_kruskal_dp`.

    The per-segment chi² cell contribution

    .. math::

        c_g = (n_{0,g} + \\tau - E_{0,g})^2 / E_{0,g}
              + (n_{1,g} + \\tau - E_{1,g})^2 / E_{1,g}

    (with Yates correction iff the combination has exactly 2 groups) is
    additive across groups **given a fixed number of groups k**: the column
    marginals ``C[c] = N_c + k·τ`` and total ``N = N₀ + N₁ + 2k·τ`` depend
    only on ``k``, not on the split positions. So we run a separate interval-DP
    per ``k ∈ [2, K]`` and merge.

    Returns a list of ``{combination, index_to_groupby, cramerv, tschuprowt}``
    sorted by ``sort_by`` desc — mirrors the yield shape of
    :meth:`_compute_associations` so it drops into the streaming pipeline.

    Complexity: O(K² · n_mod² · top_k · log top_k). Independent of the
    combination count (which can reach ~8 M at ``n_mod=40, max_n_mod=7``).

    Edge cases (mirror :func:`_chi2_assoc_for_combination`):

    * ``max_n_mod < 2`` or ``n_mod < 2``: returns ``[]``.
    * ``sort_by`` must be ``"cramerv"`` or ``"tschuprowt"``.
    """
    if sort_by not in ("cramerv", "tschuprowt"):
        raise ValueError(f"sort_by must be 'cramerv' or 'tschuprowt', got {sort_by!r}")

    n_mod = len(raw_index)
    K = min(max_n_mod, n_mod)
    if K < 2:
        return []

    n0_prefix = np.concatenate([[0.0], np.cumsum(n0_per_mod.astype(np.float64))])
    n1_prefix = np.concatenate([[0.0], np.cumsum(n1_per_mod.astype(np.float64))])
    N0_total = float(n0_prefix[-1])
    N1_total = float(n1_prefix[-1])
    n_obs = N0_total + N1_total

    # Collected across all k: (sort_key, cramerv_q, tschuprowt_q, splits)
    all_entries: list[tuple[float, float, float, tuple[int, ...]]] = []

    for k_groups in range(2, K + 1):
        C0 = N0_total + k_groups * tol
        C1 = N1_total + k_groups * tol
        N_with_tol = N0_total + N1_total + 2.0 * k_groups * tol
        yates = k_groups == 2

        def seg_cost(
            i: int, j: int, _C0: float = C0, _C1: float = C1, _N: float = N_with_tol, _yates: bool = yates
        ) -> float:
            obs0 = (n0_prefix[j] - n0_prefix[i]) + tol
            obs1 = (n1_prefix[j] - n1_prefix[i]) + tol
            R = obs0 + obs1
            E0 = R * _C0 / _N
            E1 = R * _C1 / _N
            if _yates:
                d0 = E0 - obs0
                d1 = E1 - obs1
                obs0 = obs0 + (1.0 if d0 > 0 else (-1.0 if d0 < 0 else 0.0)) * min(0.5, abs(d0))
                obs1 = obs1 + (1.0 if d1 > 0 else (-1.0 if d1 < 0 else 0.0)) * min(0.5, abs(d1))
            return (obs0 - E0) ** 2 / E0 + (obs1 - E1) ** 2 / E1

        # Only the final row (k == k_groups) is used: rebuilding the DP per k_groups
        # is required because C0/C1/N/yates depend on k_groups (see docstring).
        entries = top_k_partitions(n_mod=n_mod, cap=k_groups, seg_cost=seg_cost, top_k=top_k, maximize=True)

        # Translate chi² → cramerv (quantised) → tschuprowt (quantised). Matches
        # :func:`_chi2_assoc_for_combination` cell-for-cell.
        denom = (k_groups - 1) ** 0.25  # k_groups ≥ 2 here, so denom > 0
        for k, chi2, splits in entries:
            if k != k_groups:
                continue
            cramerv_raw = (chi2 / n_obs) ** 0.5
            cramerv_q = round(cramerv_raw / tol) * tol
            tt_raw = cramerv_q / denom
            tt_q = round(tt_raw / tol) * tol
            sort_key = tt_q if sort_by == "tschuprowt" else cramerv_q
            all_entries.append((sort_key, cramerv_q, tt_q, splits))

    all_entries.sort(key=lambda x: x[0], reverse=True)
    all_entries = all_entries[:top_k]

    out: list[dict] = []
    for _, cv, tt, splits in all_entries:
        combination = splits_to_combination(splits, raw_index)
        out.append(
            {
                "combination": combination,
                "index_to_groupby": combination_formatter(combination),
                "cramerv": float(cv),
                "tschuprowt": float(tt),
            }
        )
    return out
