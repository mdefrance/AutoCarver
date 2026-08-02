"""Tool to build optimized buckets out of Quantitative and Qualitative features
for ordinal targets (ordered, integer-encoded modalities).
"""

from typing import Literal

import numpy as np
import pandas as pd

from AutoCarver.carvers.binary_carver import get_crosstab
from AutoCarver.carvers.utils.base_carver import BaseCarver, Samples, parallel_aggregate
from AutoCarver.combinations import CombinationEvaluator, KendallTauCCombinations
from AutoCarver.combinations.ordinal.ordinal_target_rates import (
    OrdinalTargetRate,
    TargetMeanLevel,
    TargetMeanRidit,
)
from AutoCarver.discretizers.utils.base_discretizer import ProcessingConfig
from AutoCarver.discretizers.utils.ridits import ridits_from_counts
from AutoCarver.features import Features
from AutoCarver.utils import extend_docstring


class OrdinalCarver(BaseCarver):
    """Automatic carving of continuous, discrete, categorical and ordinal
    features that maximizes association with an **ordinal** target.

    The target must be **integer-encoded** with ordered levels (e.g. ``1..K``,
    ``K > 2``); the level order is taken from the ascending integer values.

    For ordinal targets, Kendall's :ref:`tau_c` is the default association
    measure to sort combinations — it rewards groupings whose order matches the
    target's while favouring robust, parsimonious cardinality. :ref:`tau_b` and
    the original Somers' D (:ref:`somersd`) are also available via
    ``combination_evaluator``.

    ``target_scale`` declares how the integer encoding of the levels should be
    read (it drives the modality pre-sort and the viability rate; the rank-based
    tau statistics are encoding-invariant either way):

    * ``"ridit"`` (**default**) — order-only levels (*Poor* / *Fair* / *Good*):
      levels are scored by their train ridits, invariant under any strictly
      increasing re-encoding.
    * ``"level"`` — count targets (e.g. 0–5 claims), where the encoding *is*
      the scale and the mean level (expected count) is the right summary.
    * ``{level: value}`` — known representative values per level (e.g. a
      calibrated default probability per rating grade), strictly increasing.

    When individual continuous target values are available, use
    :class:`ContinuousCarver` instead.
    """

    __name__ = "OrdinalCarver"
    is_y_ordinal = True
    _default_evaluator = KendallTauCCombinations
    _evaluator_trait = "is_y_ordinal"
    _target_kind = "ordinal targets"
    _evaluator_choices = "KendallTauCCombinations, KendallTauBCombinations, SomersDCombinations"

    @extend_docstring(BaseCarver.__init__, exclude=["combination_evaluator"])
    def __init__(
        self,
        features: Features,
        min_freq: float = 0.02,
        max_n_mod: int = 5,
        *,
        combination_evaluator: CombinationEvaluator | None = None,
        target_scale: Literal["ridit", "level"] | dict = "ridit",
        config: ProcessingConfig | None = None,
    ) -> None:
        """
        Parameters
        ----------
        combination_evaluator : CombinationEvaluator, optional
            Pre-built evaluator instance measuring association between
            :class:`Features` and an ordinal target. Defaults to
            :class:`KendallTauCCombinations`.

            Choose from: :class:`KendallTauCCombinations` (default),
            :class:`KendallTauBCombinations`, :class:`SomersDCombinations`.

        target_scale : "ridit", "level" or dict, optional
            How the integer encoding of the target levels is read, by default
            ``"ridit"``. A dict maps each level to its (strictly increasing) representative value. Conflicts with a
            ``combination_evaluator`` carrying an explicit non-ridit ``target_rate``.
        """
        combination_evaluator = self._resolve_evaluator(combination_evaluator)

        # resolving target_scale into the evaluator's rate — the single source of truth
        # both the viability tests and the modality pre-sort derive from.
        if isinstance(combination_evaluator.target_rate, TargetMeanRidit):
            # the evaluator carries the default rate (no explicit user choice)
            combination_evaluator.target_rate = _target_rate_from_scale(target_scale, self.__name__)
        elif target_scale != "ridit":
            raise ValueError(
                f"[{self.__name__}] both target_scale={target_scale!r} and an explicit "
                f"target_rate ({type(combination_evaluator.target_rate).__name__}) were "
                "provided; declare the scale through only one of them."
            )

        super().__init__(
            features=features,
            min_freq=min_freq,
            max_n_mod=max_n_mod,
            combination_evaluator=combination_evaluator,
            config=config,
        )

    def _prepare_samples(self, samples: Samples) -> Samples:
        """Validates format and content of X and y."""
        if samples.train.y is None:
            raise ValueError(f"[{self.__name__}] y must be provided")
        if not pd.api.types.is_numeric_dtype(samples.train.y):
            raise ValueError(
                f"[{self.__name__}] y must be an integer-encoded ordinal Series; "
                "integer-encode your ordered target (e.g. 1..K) before carving."
            )

        y_values = np.unique(samples.train.y)
        if len(y_values) <= 2:
            raise ValueError(f"[{self.__name__}] provided y has <=2 levels, consider using BinaryCarver instead.")
        if not np.all(np.equal(np.mod(y_values, 1), 0)):
            raise ValueError(
                f"[{self.__name__}] y must be integer-encoded ordered levels (e.g. 1..K); got non-integer values."
            )

        # deriving the modality pre-sort scale from the resolved target rate (single source
        # of truth: pre-sort and viability can never disagree) before super() discretizes
        target_rate = self.combination_evaluator.target_rate
        if isinstance(target_rate, TargetMeanRidit):
            self.config.y_level_scores = ridits_from_counts(samples.train.y.value_counts())
        elif isinstance(target_rate, TargetMeanLevel) and target_rate.level_values is not None:
            uncovered = [level for level in y_values if level not in target_rate.level_values]
            if len(uncovered) > 0:
                raise ValueError(
                    f"[{self.__name__}] observed y levels {uncovered} are missing from "
                    "target_scale; provide a value for every train level."
                )
            self.config.y_level_scores = dict(target_rate.level_values)

        return super()._prepare_samples(samples)

    def _aggregator(self, X: pd.DataFrame, y: pd.Series) -> dict[str, pd.Series | pd.DataFrame | None]:
        """Computes ordered contingency tables (feature modalities × ordinal target
        levels) for specified features, ordered according to the known labels.

        Threaded across features when ``n_jobs > 1`` (pd.crosstab emits one column per ordinal
        level, sorted ascending — correct ordinal column order)."""
        return parallel_aggregate(get_crosstab, self.features, X, y, self.config.n_jobs)

    def _fit_rate_reference(self, xagg: pd.Series | pd.DataFrame) -> None:
        """Fits the ridit reference from the raw train crosstab before the pre-combination
        "Raw distribution" print (mirrors
        :meth:`~AutoCarver.carvers.multiclass_carver.MulticlassCarver._fit_rate_reference`).

        Without this, a verbose fit would call ``target_rate.compute`` (via
        :meth:`BaseCarver._pretty_print`) before :meth:`get_best_combination`
        has had a chance to fit the reference (see
        :meth:`~AutoCarver.combinations.ordinal.ordinal_target_rates.TargetMeanRidit.fit_reference`).
        This fit only serves the print: ``get_best_combination`` refits the
        reference before scoring any candidate — from the same crosstab, minus
        the NaN row when the feature has NaNs — so carving never depends on it.
        """
        target_rate = self.combination_evaluator.target_rate
        if isinstance(target_rate, OrdinalTargetRate):
            target_rate.fit_reference(xagg)  # type: ignore


def _target_rate_from_scale(target_scale: Literal["ridit", "level"] | dict, name: str) -> OrdinalTargetRate:
    """Resolves a ``target_scale`` mode into its :class:`OrdinalTargetRate`."""
    if isinstance(target_scale, dict):
        return TargetMeanLevel(level_values=target_scale)
    if target_scale == "ridit":
        return TargetMeanRidit()
    if target_scale == "level":
        return TargetMeanLevel()
    raise ValueError(f"[{name}] target_scale must be 'ridit', 'level' or a {{level: value}} dict, got {target_scale!r}")
