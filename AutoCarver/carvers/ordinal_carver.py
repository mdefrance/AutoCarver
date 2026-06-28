"""Tool to build optimized buckets out of Quantitative and Qualitative features
for ordinal targets (ordered, integer-encoded modalities).
"""

import numpy as np
import pandas as pd

from AutoCarver.carvers.binary_carver import get_crosstab
from AutoCarver.carvers.utils.base_carver import BaseCarver, Samples, parallel_aggregate
from AutoCarver.combinations import CombinationEvaluator, KendallTauCCombinations
from AutoCarver.discretizers.utils.base_discretizer import ProcessingConfig
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
    """

    __name__ = "OrdinalCarver"
    is_y_ordinal = True

    @extend_docstring(BaseCarver.__init__, exclude=["combination_evaluator"])
    def __init__(
        self,
        features: Features,
        min_freq: float,
        max_n_mod: int,
        *,
        combination_evaluator: CombinationEvaluator | None = None,
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
        """
        if combination_evaluator is None:
            combination_evaluator = KendallTauCCombinations()
        if not combination_evaluator.is_y_ordinal:
            raise ValueError(
                f"[{self.__name__}] {type(combination_evaluator).__name__} is not suited for ordinal targets. "
                f"Choose from: KendallTauCCombinations, KendallTauBCombinations, SomersDCombinations."
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

        return super()._prepare_samples(samples)

    def _aggregator(self, X: pd.DataFrame, y: pd.Series) -> dict[str, pd.Series | pd.DataFrame | None]:
        """Computes ordered contingency tables (feature modalities × ordinal target
        levels) for specified features, ordered according to the known labels.

        Threaded across features when ``n_jobs > 1`` (pd.crosstab emits one column per ordinal
        level, sorted ascending — correct ordinal column order)."""
        return parallel_aggregate(get_crosstab, self.features, X, y, self.config.n_jobs)
