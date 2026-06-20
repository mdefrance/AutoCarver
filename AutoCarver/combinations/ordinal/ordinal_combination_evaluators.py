"""Module for ordinal combination evaluators."""

import math
from abc import ABC

import numpy as np
import pandas as pd

from AutoCarver.combinations.ordinal.ordinal_target_rates import OrdinalTargetRate, TargetMeanLevel
from AutoCarver.combinations.utils.combination_evaluator import AggregatedSample, CombinationEvaluator
from AutoCarver.combinations.utils.combinations import group_crosstab
from AutoCarver.combinations.utils.target_rate import TargetRate


class OrdinalCombinationEvaluator(CombinationEvaluator[pd.DataFrame], ABC):
    """Ordinal combination evaluator class.

    The aggregation is an ordered contingency table
    ``feature-groups (rows, in target-rate order) × ordinal-target-levels
    (cols, ascending)`` — the binary crosstab generalised from 2 columns to as
    many columns as the target has levels. Three rank-association statistics are
    computed per combination; concrete subclasses pick which one ranks
    combinations via :attr:`sort_by`:

    * :ref:`tau_c` — Kendall/Stuart's tau-c (**default**, rectangular-table
      correction; self-balances to a robust, meaningful number of modalities);
    * :ref:`tau_b` — Kendall's tau-b (matches :func:`scipy.stats.kendalltau`);
    * :ref:`somersd` — the original asymmetric Somers' D ``D(Y|X)`` (target given
      feature).

    The symmetric Kendall taus reward a split only when it is genuinely
    discriminative and otherwise favour fewer, more robust modalities — like
    :class:`TschuprowtCombinations` and the Kruskal effect sizes. Somers' D is
    asymmetric and leans strongly toward the coarsest split.

    Search uses the inherited enumerate-and-score path.
    """

    is_y_ordinal = True
    _target_rate_classes: list[type[OrdinalTargetRate]] = [TargetMeanLevel]
    # narrow inherited attribute: ordinal evaluators always carry an OrdinalTargetRate
    # (enforced by _init_target_rate).
    target_rate: OrdinalTargetRate
    # narrow inherited `sort_by: str | None`: concrete subclasses always set a str.
    sort_by: str

    def _init_target_rate(self, target_rate: TargetRate[pd.DataFrame] | None) -> OrdinalTargetRate:
        """Initializes target rate."""
        if target_rate is None:
            return TargetMeanLevel()
        if not isinstance(target_rate, OrdinalTargetRate):
            raise ValueError("target_rate must be an OrdinalTargetRate")
        return target_rate

    def _grouper(self, xagg: AggregatedSample, groupby: dict) -> pd.DataFrame:
        """Groups a crosstab by ``groupby`` and sums column values by group.

        Shares :func:`group_crosstab` with the binary path: leaders are ordered
        by first appearance so grouping stays independent of label text.
        """
        return group_crosstab(xagg, groupby)

    def _association_measure(
        self,
        xagg: AggregatedSample | pd.Series | pd.DataFrame,
        n_obs: int | None = None,
        tol: float = 1e-10,
    ) -> dict[str, float | None]:
        """Computes Kendall's tau-b, tau-c and Somers' D between feature and target.

        Parameters
        ----------
        xagg : pd.DataFrame
            Ordered contingency table (rows = feature groups, cols = ordinal
            target levels). ``n_obs`` / ``tol`` are unused (the rank statistics
            only depend on the table's cell counts).

        Returns
        -------
        dict[str, float | None]
            ``{"tau_b": ..., "tau_c": ..., "somersd": ...}``; any may be
            ``None`` for a degenerate table.
        """
        _, _ = n_obs, tol  # unused
        return _ordinal_associations(np.asarray(xagg.values, dtype=float))


class KendallTauCCombinations(OrdinalCombinationEvaluator):
    """Kendall's tau-c based combination evaluation toolkit (ordinal default).

    Stuart's tau-c applies a ``min(r, c)`` correction tailored to **rectangular**
    tables — exactly our shape (few feature groups × many target levels) — so its
    magnitude stays comparable across combinations with different group counts and
    it leans toward fewer, robust modalities, only adding one when a split is
    genuinely meaningful.
    """

    sort_by = "tau_c"


class KendallTauBCombinations(OrdinalCombinationEvaluator):
    """Kendall's tau-b based combination evaluation toolkit.

    Bit-exact with :func:`scipy.stats.kendalltau` (the ``tau-b`` variant) on the
    grouped contingency table — pinned by parity tests. Normalised by the
    geometric mean of both margins' untied pairs; tends to retain more modalities
    on smoothly monotone signals than :class:`KendallTauCCombinations`.
    """

    sort_by = "tau_b"


class SomersDCombinations(OrdinalCombinationEvaluator):
    """Somers' D based combination evaluation toolkit.

    The original asymmetric Somers' D ``D(Y|X)`` — concordant minus discordant
    pairs over pairs untied on the feature ``X`` — matching
    ``scipy.stats.somersd(table).statistic``. Being asymmetric it leans strongly
    toward the coarsest split (its maximum over groupings is typically two
    modalities); offered for users who specifically want raw Somers' D rather
    than the self-balancing Kendall taus.
    """

    sort_by = "somersd"


def _ordinal_associations(values: np.ndarray) -> dict[str, float | None]:
    """Kendall's tau-b, tau-c and Somers' D ``D(Y|X)`` for an ordered table.

    ``values`` is the ``(r, c)`` cell-count array with rows = ``X`` (feature
    groups) and columns = ``Y`` (target levels), both already in ascending order.
    Concordant ``C`` and discordant ``D`` pairs and the margin tie counts are
    computed once and shared across:

    * ``tau_b = (C − D) / sqrt((P0 − T_X)(P0 − T_Y))`` — matches
      ``scipy.stats.kendalltau``;
    * ``tau_c = 2·m·(C − D) / (n²·(m − 1))`` with ``m = min(r, c)`` over the
      non-empty rows/columns (Stuart's rectangular-table correction);
    * ``somersd = (C − D) / (P0 − T_X)`` — the original Somers' D ``D(Y|X)``;

    where ``P0 = n(n−1)/2`` and ``T_X`` / ``T_Y`` are pairs tied on the feature /
    target. Each measure is ``None`` when its denominator vanishes (degenerate
    table), mirroring the continuous evaluator's ``None`` convention.
    """
    n = float(values.sum())
    if n < 2:
        return {"tau_b": None, "tau_c": None, "somersd": None}

    # concordant partners of each cell: counts strictly down-right (k>i, l>j)
    suffix = np.cumsum(np.cumsum(values[::-1, ::-1], axis=0), axis=1)[::-1, ::-1]
    down_right = np.zeros_like(values)
    down_right[:-1, :-1] = suffix[1:, 1:]

    # discordant partners of each cell: counts strictly down-left (k>i, l<j)
    suffix_rows_prefix_cols = np.cumsum(np.cumsum(values[::-1, :], axis=0)[::-1, :], axis=1)
    down_left = np.zeros_like(values)
    down_left[:-1, 1:] = suffix_rows_prefix_cols[1:, :-1]

    concordant_minus_discordant = float((values * down_right).sum()) - float((values * down_left).sum())

    row = values.sum(axis=1)
    col = values.sum(axis=0)
    all_pairs = n * (n - 1) / 2.0
    untied_on_feature = all_pairs - float((row * (row - 1) / 2.0).sum())
    untied_on_target = all_pairs - float((col * (col - 1) / 2.0).sum())

    # tau-b: geometric mean of untied pairs on each margin
    denominator_b = math.sqrt(untied_on_feature * untied_on_target)
    tau_b = concordant_minus_discordant / denominator_b if denominator_b > 0 else None

    # tau-c: Stuart's rectangular-table correction
    m = min(int((row > 0).sum()), int((col > 0).sum()))
    tau_c = (2.0 * m * concordant_minus_discordant) / (n * n * (m - 1)) if m > 1 else None

    # somersd: original asymmetric D(Y|X), untied on the feature
    somersd = concordant_minus_discordant / untied_on_feature if untied_on_feature > 0 else None

    return {"tau_b": tau_b, "tau_c": tau_c, "somersd": somersd}
