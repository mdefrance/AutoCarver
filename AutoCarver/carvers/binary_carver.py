"""Tool to build optimized buckets out of Quantitative and Qualitative features
for binary classification tasks.
"""

from numpy import unique
from pandas import DataFrame, Series, crosstab

from ..combinations import TschuprowtCombinations
from ..features import BaseFeature, Features
from ..utils import extend_docstring
from .utils.base_carver import BaseCarver, Samples


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
        dropna: bool = True,
        max_n_mod: int = 5,
        **kwargs,
    ) -> None:
        """
        Keyword Arguments
        -----------------

        ordinal_encoding : bool, optional
            Whether or not to ordinal encode :class:`Features`, by default ``True``

        combinations : BinaryCombinationEvaluator, optional
            Metric to perform association measure between :class:`Features` and target.

            .. tip::
                * Use :ref:`TschuprowtCombinations` for less, more robust, modalities
                * Use :ref:`CramervCombinations` for more, less robust, modalities
        """

        # default binary combinations
        combinations = kwargs.pop("combinations", None)
        if combinations is None:
            combinations = TschuprowtCombinations(max_n_mod=max_n_mod)

        # association measure used to find the best groups for binary targets
        if not combinations.is_y_binary:
            raise ValueError(
                f"[{self.__name__}] {combinations} is not suited for binary targets. "
                f"Choose from: TschuprowtCombinations, CramervCombinations."
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
        tuple[DataFrame, DataFrame, dict[str, Callable]]
            Copies of (X, X_dev) and helpers to be used according to target type
        """
        # binary target, checking values
        y_values = unique(samples.train.y)
        if not ((0 in y_values) and (1 in y_values)) or len(y_values) != 2:
            raise ValueError(
                f"[{self.__name__}] y must be a binary Series of 0 and 1 (int or float, not object)"
            )

        # Checking for binary target and discretizing X
        return super()._prepare_data(samples)

    def _aggregator(self, X: DataFrame, y: Series) -> dict[str, DataFrame]:
        """Computes crosstabs for specified features and ensures that the crosstab is ordered
        according to the known labels

        Parameters
        ----------
        X : DataFrame
            _description_
        y : Series
            _description_

        Returns
        -------
        dict[str, DataFrame]
            dict of crosstab(X, y) by feature name
        """
        # checking for empty datasets (dev)
        xtabs = {feature.version: None for feature in self.features}
        if X is not None:
            # crosstab for each feature
            for feature in self.features:
                xtabs.update({feature.version: get_crosstab(X, y, feature)})

        return xtabs


def get_crosstab(X: DataFrame, y: Series, feature: BaseFeature) -> dict:
    """Computes crosstabs for specified features and ensures that the crosstab is ordered
    according to the known labels"""

    # computing crosstab between binary y and categorical X
    xtab = crosstab(X[feature.version], y)

    # reindexing to ensure the right order
    return xtab.reindex(feature.labels, fill_value=0)
