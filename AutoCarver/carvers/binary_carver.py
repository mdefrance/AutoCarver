"""Tool to build optimized buckets out of Quantitative and Qualitative features
for binary classification tasks.
"""

import numpy as np
import pandas as pd

from AutoCarver.carvers.utils.base_carver import BaseCarver, Samples
from AutoCarver.combinations import CombinationEvaluator, TschuprowtCombinations
from AutoCarver.discretizers.utils.base_discretizer import DiscretizerConfig
from AutoCarver.features import BaseFeature, Features
from AutoCarver.utils import extend_docstring


class BinaryCarver(BaseCarver):
    """Automatic carving of continuous, discrete, categorical and ordinal
    features that maximizes association with a binary target.


    Examples
    --------
    `Binary Classification Example <https://autocarver.readthedocs.io/en/latest/examples/Carvers/
    BinaryClassification/binary_classification_example.html>`_
    """

    __name__ = "BinaryCarver"
    is_y_binary = True

    @extend_docstring(BaseCarver.__init__, exclude=["ordinal_encoding"])
    def __init__(
        self,
        features: Features,
        min_freq: float,
        *,
        dropna: bool = True,
        ordinal_encoding: bool = True,
        max_n_mod: int = 5,
        combinations: CombinationEvaluator | None = None,
        discretizer_min_freq: float | None = None,
        config: DiscretizerConfig | None = None,
    ) -> None:
        """
        Keyword Arguments
        -----------------

        combinations : BinaryCombinationEvaluator, optional
            Metric to perform association measure between :class:`Features` and target.

            .. tip::
                * Use :ref:`TschuprowtCombinations` for less, more robust, modalities
                * Use :ref:`CramervCombinations` for more, less robust, modalities
        """

        if combinations is None:
            combinations = TschuprowtCombinations(max_n_mod=max_n_mod)

        if not combinations.is_y_binary:
            raise ValueError(
                f"[{self.__name__}] {combinations} is not suited for binary targets. "
                f"Choose from: TschuprowtCombinations, CramervCombinations."
            )

        super().__init__(
            features=features,
            min_freq=min_freq,
            combinations=combinations,
            dropna=dropna,
            ordinal_encoding=ordinal_encoding,
            discretizer_min_freq=discretizer_min_freq,
            config=config,
        )

    def _prepare_data(self, samples: Samples) -> Samples:
        """Validates format and content of X and y."""
        if samples.train.y is None:
            raise ValueError(f"[{self.__name__}] y must be provided")
        y_values = np.unique(samples.train.y)
        if not ((0 in y_values) and (1 in y_values)) or len(y_values) != 2:
            raise ValueError(f"[{self.__name__}] y must be a binary Series of 0 and 1 (int or float, not object)")

        return super()._prepare_data(samples)

    def _aggregator(self, X: pd.DataFrame, y: pd.Series) -> dict[str, pd.DataFrame | None]:
        """Computes crosstabs for specified features and ensures that the crosstab is ordered
        according to the known labels"""
        # checking for empty datasets (dev)
        xtabs = {feature.version: None for feature in self.features}
        if X is not None:
            # crosstab for each feature
            for feature in self.features:
                xtabs.update({feature.version: get_crosstab(X, y, feature)})

        return xtabs


def get_crosstab(X: pd.DataFrame, y: pd.Series, feature: BaseFeature) -> pd.DataFrame:
    """Computes crosstabs for specified features and ensures that the crosstab is ordered
    according to the known labels"""

    # computing crosstab between binary y and categorical X
    xtab = pd.crosstab(X[feature.version], y)

    # reindexing to ensure the right order
    return xtab.reindex(feature.labels, fill_value=0)
