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


class BinaryCombinationEvaluator(CombinationEvaluator, ABC):
    """Binary combination evaluator class."""

    is_y_binary = True
    _target_rate_classes: list[type[BinaryTargetRate]] = [TargetMean, OddsRatio, Woe]

    def _init_target_rate(self, target_rate: BinaryTargetRate | None) -> BinaryTargetRate:
        """Initializes target rate."""
        if target_rate is None:
            return TargetMean()
        elif not isinstance(target_rate, BinaryTargetRate):
            raise ValueError("target_rate must be a BinaryTargetRate")
        return target_rate

    def _association_measure(
        self, xagg: AggregatedSample, n_obs: int | None = None, tol: float = 1e-10
    ) -> dict[str, float]:
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
        into per-combination groups via :func:`numpy.bincount`, skipping the
        scipy call and the per-combination DataFrame construction.

        For each combination:

        * group counts ``(n_{0,g}, n_{1,g})`` are obtained by summing per-raw-modality
          counts via bincount on the integer assignment derived from
          ``index_to_groupby``;
        * ``tol`` is added to every cell (matching the existing scipy call,
          which received ``xagg.values + tol``);
        * the expected frequencies ``E_{gc} = R_g \\cdot C_c / N`` are computed
          via outer product;
        * Yates correction is applied iff the table is 2×2 (matching scipy's
          default);
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

        for grouped_xagg in tqdm(grouped_xaggs, desc="Computing associations", disable=not self.verbose):
            cv, tt = _chi2_assoc_for_combination(
                n0_per_mod=n0_per_mod,
                n1_per_mod=n1_per_mod,
                n_obs=n_obs,
                mod_to_pos=mod_to_pos,
                n_mod=n_mod,
                index_to_groupby=grouped_xagg["index_to_groupby"],
                tol=tol,
            )
            yield {
                "combination": grouped_xagg["combination"],
                "index_to_groupby": grouped_xagg["index_to_groupby"],
                "cramerv": cv,
                "tschuprowt": tt,
            }


class TschuprowtCombinations(BinaryCombinationEvaluator):
    """Tschuprow's T based combination evaluation toolkit"""

    sort_by = "tschuprowt"


class CramervCombinations(BinaryCombinationEvaluator):
    """Cramér's V based combination evaluation toolkit"""

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
