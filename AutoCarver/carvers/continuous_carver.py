"""Tool to build optimized buckets out of Quantitative and Qualitative features
for continuous regression tasks.
"""

from numpy import unique
from pandas import DataFrame, Series

from ..combinations import KruskalCombinations
from ..features import BaseFeature, Features
from ..utils import extend_docstring
from .utils.base_carver import BaseCarver, Samples


class ContinuousCarver(BaseCarver):
    """Automatic carving of continuous, discrete, categorical and ordinal
    features that maximizes association with a continuous target.

    For continuous targets, :ref:`Kruskal` is used as association measure to sort combinations.

    Examples
    --------
    `Continuous Regression Example <https://autocarver.readthedocs.io/en/latest/examples/Carvers/
    ContinuousRegression/continuous_regression_example.html>`_
    """

    __name__ = "ContinuousCarver"
    is_y_continuous = True

    @extend_docstring(BaseCarver.__init__)
    def __init__(
        self,
        features: Features,
        min_freq: float,
        dropna: bool = True,
        max_n_mod: int = 5,
        **kwargs,
    ) -> None:
        """
        Keyword Arguments
        -----------------
        combinations : ContinuousCombinationEvaluator, optional
            Metric to perform association measure between :class:`Features` and target.

            Currently, only :ref:`KruskalCombinations` are implemented.
        """
        # default binary combinations
        combinations = kwargs.pop("combinations", None)
        if combinations is None:
            combinations = KruskalCombinations(max_n_mod=max_n_mod)

        # association measure used to find the best groups for binary targets
        if not combinations.is_y_continuous:
            raise ValueError(
                f"[{self.__name__}] {combinations} is not suited for continuous targets. "
                f"Choose from: KruskalCombinations."
            )

        # Initiating BaseCarver
        super().__init__(
            features=features,
            min_freq=min_freq,
            combinations=combinations,
            dropna=dropna,
            **kwargs,
        )

    def _prepare_data(self, samples: Samples) -> Samples:
        """Validates format and content of X and y.

        Parameters
        ----------
        X : DataFrame
            Dataset used to discretize. Needs to have columns has specified in
            ``AutoCarver.features``.

        y : Series
            Binary target feature with wich the association is maximized.

        X_dev : DataFrame, optional
            Dataset to evalute the robustness of discretization, by default ``None``
            It should have the same distribution as X.

        y_dev : Series, optional
            Binary target feature with wich the robustness of discretization is evaluated,
            by default ``None``

        Returns
        -------
        tuple[DataFrame, DataFrame]
            Copies of (X, X_dev) to be used according to target type
        """

        # continuous target, checking values
        if str in samples.train.y.apply(type).unique():
            raise ValueError(
                f"[{self.__name__}] y must be a continuous Series (int or float, not object)"
            )

        y_values = unique(samples.train.y)
        if len(y_values) <= 2:
            raise ValueError(
                f"[{self.__name__}] provided y is binary, consider using BinaryCarver instead."
            )
        # Checking for binary target and copying X
        return super()._prepare_data(samples)

    def _aggregator(self, X: DataFrame, y: Series) -> dict[str, DataFrame]:
        """Computes y values for modalities of specified features and ensures the ordering
        according to the known labels

        Parameters
        ----------
        features : list[str]
            _description_
        X : DataFrame
            _description_
        y : Series
            _description_
        labels_orders : dict[str, GroupedList]
            _description_

        Returns
        -------
        dict[str, DataFrame]
            _description_
        """
        # checking for empty datasets
        yvals = {feature.version: None for feature in self.features}
        if X is not None:
            # list of y values for each modality of X
            for feature in self.features:
                yvals.update({feature.version: get_target_values_by_modality(X, y, feature)})

        return yvals


def get_target_values_by_modality(X: DataFrame, y: Series, feature: BaseFeature) -> dict:
    """Computes y values for modalities of specified features and ensures the ordering
    according to the known labels"""

    # list of y values for each modality of X
    yval = y.groupby(X[feature.version]).apply(lambda u: list(u))

    # reindexing to ensure the right order
    return yval.reindex(feature.labels, fill_value=[])
