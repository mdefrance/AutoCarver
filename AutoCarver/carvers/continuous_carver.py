"""Tool to build optimized buckets out of Quantitative and Qualitative features
for continuous regression tasks.
"""

import numpy as np
import pandas as pd

from AutoCarver.carvers.utils.base_carver import BaseCarver, Samples
from AutoCarver.combinations import CombinationEvaluator, KruskalCombinations
from AutoCarver.discretizers.utils.base_discretizer import ProcessingConfig
from AutoCarver.features import BaseFeature, Features
from AutoCarver.utils import extend_docstring


class ContinuousCarver(BaseCarver):
    """Automatic carving of continuous, discrete, categorical and ordinal
    features that maximizes association with a continuous target.

    For continuous targets, :ref:`kruskal` is used as association measure to sort combinations.

    Examples
    --------
    `Continuous Regression Example <https://autocarver.readthedocs.io/en/latest/examples/Carvers/
    ContinuousRegression/continuous_regression_example.html>`_
    """

    __name__ = "ContinuousCarver"
    is_y_continuous = True

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
            :class:`Features` and a continuous target. Defaults to
            :class:`KruskalCombinations`.

            Currently, only :ref:`KruskalCombinations` is implemented for
            continuous targets.
        """
        if combination_evaluator is None:
            combination_evaluator = KruskalCombinations()
        if not combination_evaluator.is_y_continuous:
            raise ValueError(
                f"[{self.__name__}] {type(combination_evaluator).__name__} is not suited for continuous targets. "
                f"Choose from: KruskalCombinations."
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
        if str in samples.train.y.apply(type).unique():
            raise ValueError(f"[{self.__name__}] y must be a continuous Series (int or float, not object)")

        y_values = np.unique(samples.train.y)
        if len(y_values) <= 2:
            raise ValueError(f"[{self.__name__}] provided y is binary, consider using BinaryCarver instead.")

        return super()._prepare_samples(samples)

    def _aggregator(self, X: pd.DataFrame, y: pd.Series) -> dict[str, pd.Series | pd.DataFrame | None]:
        """Computes y values for modalities of specified features and ensures the ordering
        according to the known labels"""
        # checking for empty datasets
        yvals = {feature.version: None for feature in self.features}
        if X is not None:
            # list of y values for each modality of X
            for feature in self.features:
                yvals.update({feature.version: get_target_values_by_modality(X, y, feature)})

        return yvals


def get_target_values_by_modality(X: pd.DataFrame, y: pd.Series, feature: BaseFeature) -> pd.Series:
    """Computes y values for modalities of specified features and ensures the ordering
    according to the known labels"""

    # list of y values for each modality of X
    yval = y.groupby(X[feature.version]).apply(lambda u: list(u))

    # reindexing to ensure the right order (labels may be None pre-fit; pandas
    # treats None as "no reindex" so the original ordering is kept)
    return yval.reindex(feature.labels, fill_value=[])  # type: ignore
