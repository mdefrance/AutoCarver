"""Module for continuous combination evaluators."""

from abc import ABC

import numpy as np
import pandas as pd
from scipy.stats import kruskal, rankdata, tiecorrect
from tqdm import tqdm

from AutoCarver.combinations.continuous.continuous_target_rates import ContinuousTargetRate, TargetMean, TargetMedian
from AutoCarver.combinations.utils.combination_evaluator import AggregatedSample, CombinationEvaluator


class ContinuousCombinationEvaluator(CombinationEvaluator, ABC):
    """Continuous combination evaluator class."""

    is_y_continuous = True
    _target_rate_classes: list[type[ContinuousTargetRate]] = [TargetMean, TargetMedian]

    def _init_target_rate(self, target_rate: ContinuousTargetRate | None) -> ContinuousTargetRate:
        """Initializes target rate."""
        if target_rate is None:
            return TargetMean()
        elif not isinstance(target_rate, ContinuousTargetRate):
            raise ValueError("target_rate must be a ContinuousTargetRate")
        return target_rate

    def _association_measure(
        self, xagg: AggregatedSample, n_obs: int | None = None, tol: float = 1e-10
    ) -> dict[str, float | None]:
        """Computes measures of association between feature and quantitative target.

        Used for the raw (one-shot) distribution. The hot per-combination loop
        goes through :meth:`_compute_associations` which evaluates the same
        Kruskal–Wallis H statistic in closed form without re-ranking — see
        :func:`_modality_rank_stats` and :func:`_kruskal_h_from_modality_stats`.

        Parameters
        ----------
        xagg : pd.DataFrame
            Values taken by y for each of x's modalities.

        Returns
        -------
        dict[str, float]
            Kruskal-Wallis' H as a dict.
        """
        _, _ = n_obs, tol  # unused attribute

        # Kruskal-Wallis' H
        try:
            return {"kruskal": kruskal(*tuple(xagg.values))[0]}
        except (ValueError, IndexError):
            return {"kruskal": None}

    def _grouper(self, xagg: AggregatedSample, groupby: dict[str, str]) -> pd.Series:
        """Groups values of y

        Parameters
        ----------
        yval : pd.Series
            _description_
        groupby : _type_
            _description_

        Returns
        -------
        Series
            _description_
        """
        # NOTE: kept as list-concatenating groupby.sum() for compatibility with
        # downstream consumers (target rates, viability tests, public API tests
        # that pin the Series-of-lists shape). The Kruskal-Wallis hot loop no
        # longer goes through this path — see _compute_associations below.
        return xagg.groupby(groupby).sum()

    def _compute_associations(self, grouped_xaggs: list[dict]) -> list[dict]:
        """Closed-form Kruskal–Wallis evaluation across all combinations.

        Statistically identical to ``scipy.stats.kruskal`` (including the
        tie correction factor); the only difference is that y is ranked **once**
        from :attr:`self.samples.train.xagg` instead of being re-ranked from
        scratch for every combination.

        For each combination:

        * group rank sums ``R_j`` and counts ``n_j`` are obtained by summing
          per-raw-modality rank sums via :func:`numpy.bincount` over the integer
          assignment derived from ``index_to_groupby``;
        * the Kruskal–Wallis H statistic is computed in closed form

          .. math::

              H = \\frac{12}{N(N+1)} \\sum_j \\frac{R_j^2}{n_j} - 3(N+1),
              \\quad H_{\\text{corrected}} = H \\,/\\, \\left(1 -
              \\frac{\\sum_i (t_i^3 - t_i)}{N^3 - N}\\right)

          where the tie correction factor depends only on the y values and is
          computed once per feature.

        Edge cases follow ``scipy.stats.kruskal``:

        * any group with ``n_j == 0`` → ``H = NaN``;
        * fewer than 2 groups, or fewer than 2 observations total → ``None``
          (matches the existing ``ValueError`` swallowing in
          :meth:`_association_measure`).
        """
        raw_xagg = self.samples.train.xagg
        # Pre-rank y once for the whole feature
        R_per_mod, n_per_mod, N, tie_corr = _modality_rank_stats(raw_xagg)

        # Map modality label -> position in R_per_mod / n_per_mod
        mod_to_pos: dict = {m: i for i, m in enumerate(raw_xagg.index)}
        n_mod = len(mod_to_pos)

        associations: list[dict] = []
        for grouped_xagg in tqdm(grouped_xaggs, desc="Computing associations", disable=not self.verbose):
            h = _kruskal_h_for_combination(
                R_per_mod=R_per_mod,
                n_per_mod=n_per_mod,
                N=N,
                tie_corr=tie_corr,
                mod_to_pos=mod_to_pos,
                n_mod=n_mod,
                index_to_groupby=grouped_xagg["index_to_groupby"],
            )
            associations.append({**grouped_xagg, "kruskal": h})

        # sorting associations according to specified metric
        return pd.DataFrame(associations).sort_values(self.sort_by, ascending=False).to_dict(orient="records")


class KruskalCombinations(ContinuousCombinationEvaluator):
    """Kruskal-Wallis' H based combination evaluation toolkit"""

    sort_by = "kruskal"


# ---------------------------------------------------------------------------
# Closed-form Kruskal–Wallis helpers
# ---------------------------------------------------------------------------


def _modality_rank_stats(
    raw_xagg: pd.Series,
) -> tuple[np.ndarray | None, np.ndarray, int, float | None]:
    """Rank ``raw_xagg``'s pooled y once and return per-modality stats.

    Returns ``(R_per_mod, n_per_mod, N, tie_corr)`` where:

    * ``R_per_mod[i]`` is the sum of average ranks of the y values in the i-th
      raw modality of ``raw_xagg`` (``rank_sum`` in Kruskal–Wallis notation);
    * ``n_per_mod[i]`` is the count of observations in the i-th raw modality;
    * ``N`` is the total number of observations;
    * ``tie_corr`` is the Kruskal–Wallis tie correction factor
      ``1 - Σ(t_i³ - t_i) / (N³ - N)`` — depends only on the y multiset.

    When ``N < 2``, ``R_per_mod`` and ``tie_corr`` are returned as ``None``
    so the per-combination caller can short-circuit.
    """
    raw_lists = [np.asarray(v, dtype=float) for v in raw_xagg.values]
    n_per_mod = np.fromiter((len(v) for v in raw_lists), dtype=np.int64, count=len(raw_lists))
    N = int(n_per_mod.sum())

    if N < 2 or len(raw_lists) == 0:
        return None, n_per_mod, N, None

    all_y = np.concatenate(raw_lists)
    ranks = rankdata(all_y, method="average")
    tie_corr = tiecorrect(ranks)

    offsets = np.concatenate([[0], np.cumsum(n_per_mod)])
    R_per_mod = np.empty(len(raw_lists), dtype=float)
    for i in range(len(raw_lists)):
        R_per_mod[i] = ranks[offsets[i] : offsets[i + 1]].sum()
    return R_per_mod, n_per_mod, N, tie_corr


def _kruskal_h_for_combination(
    *,
    R_per_mod: np.ndarray | None,
    n_per_mod: np.ndarray,
    N: int,
    tie_corr: float | None,
    mod_to_pos: dict,
    n_mod: int,
    index_to_groupby: dict,
) -> float | None:
    """Closed-form Kruskal–Wallis H for one combination.

    ``mod_to_pos`` is a precomputed ``{modality_label: position_in_R_per_mod}``
    map; ``index_to_groupby`` is the per-combination ``{modality: group_leader}``
    dict produced by :func:`combination_formatter`.
    """
    if R_per_mod is None or N < 2:
        return None

    # Build integer group assignment for this combination
    leader_to_grp: dict = {}
    assign = np.empty(n_mod, dtype=np.intp)
    for mod, leader in index_to_groupby.items():
        gid = leader_to_grp.get(leader)
        if gid is None:
            gid = len(leader_to_grp)
            leader_to_grp[leader] = gid
        assign[mod_to_pos[mod]] = gid

    n_groups = len(leader_to_grp)
    # scipy.stats.kruskal requires at least 2 groups; mirror that here.
    if n_groups < 2:
        return None

    # Per-group rank sums and counts (vectorized).
    R_g = np.bincount(assign, weights=R_per_mod, minlength=n_groups)
    n_g = np.bincount(assign, weights=n_per_mod.astype(float), minlength=n_groups)

    # Empty groups -> NaN, matching scipy (0**2 / 0 → nan).
    if (n_g == 0).any():
        with np.errstate(invalid="ignore", divide="ignore"):
            ssbn = float((R_g**2 / n_g).sum())
        if not np.isfinite(ssbn):
            return float("nan")
    else:
        ssbn = float((R_g**2 / n_g).sum())

    h = (12.0 / (N * (N + 1))) * ssbn - 3.0 * (N + 1)

    # All values identical → tie_corr == 0; scipy returns nan from H/0.
    if tie_corr is None or tie_corr == 0:
        return float("nan")

    return h / tie_corr
