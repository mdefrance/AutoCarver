"""Tool to build optimized buckets out of Quantitative and Qualitative features
for binary classification tasks.
"""

from typing import Callable

from numpy import unique
from pandas import DataFrame, Series, crosstab

from ..features import BaseFeature, Features
from ..utils import extend_docstring
from .utils.base_carver import BaseCarver


class BinaryCarver(BaseCarver):
    """Automatic carving of continuous, discrete, categorical and ordinal
    features that maximizes association with a binary target.

    Examples
    --------
    `Binary Classification Example <https://autocarver.readthedocs.io/en/latest/examples/
    BinaryClassification/binary_classification_example.html>`_
    """

    __name__ = "BinaryCarver"

    @extend_docstring(BaseCarver.__init__)
    def __init__(
        self,
        sort_by: str,
        min_freq: float,
        features: Features,
        *,
        max_n_mod: int = 5,
        dropna: bool = True,
        **kwargs: dict,
    ) -> None:
        """
        Parameters
        ----------
        sort_by : str
            Metric to be used to perform association measure between features and target.

            * ``"tschuprowt"``, for :ref:`Tschuprowt`.
            * ``"cramerv"``, for :ref:`Cramerv`.

            **Tip**: use ``"tschuprowt"`` for more robust, or less output modalities,
            use ``"cramerv"`` for more output modalities.
        """

        # association measure used to find the best groups for binary targets
        implemented_measures = ["tschuprowt", "cramerv"]
        if sort_by not in implemented_measures:
            raise ValueError(
                f"[{self.__name__}] Measure '{sort_by}' not implemented for binary targets. "
                f"Choose from: {str(implemented_measures)}."
            )

        # Initiating BaseCarver
        super().__init__(
            min_freq=min_freq,
            sort_by=sort_by,
            features=features,
            max_n_mod=max_n_mod,
            dropna=dropna,
            **kwargs,
        )

    def _prepare_data(
        self,
        X: DataFrame,
        y: Series,
        X_dev: DataFrame = None,
        y_dev: Series = None,
    ) -> tuple[DataFrame, DataFrame, dict[str, Callable]]:
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
        y_values = unique(y)
        if not ((0 in y_values) and (1 in y_values)) or len(y_values) != 2:
            raise ValueError(
                f"[{self.__name__}] y must be a binary Series (int or float, not object)"
            )

        # Checking for binary target and discretizing X
        x_copy, x_dev_copy = super()._prepare_data(X, y, X_dev=X_dev, y_dev=y_dev)

        return x_copy, x_dev_copy

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

    def _target_rate(self, xagg: DataFrame) -> DataFrame:
        """Computes target rate per row for a binary target (column) in a crosstab

        Parameters
        ----------
        xagg : DataFrame
            crosstab of X by y

        Returns
        -------
        DataFrame
            y mean by xagg index
        """
        return xagg[1].divide(xagg.sum(axis=1)).sort_values()


def get_crosstab(X: DataFrame, y: Series, feature: BaseFeature) -> dict:
    """Computes crosstabs for specified features and ensures that the crosstab is ordered
    according to the known labels"""

    # computing crosstab between binary y and categorical X
    xtab = crosstab(X[feature.version], y)

    # reindexing to ensure the right order
    return xtab.reindex(feature.labels, fill_value=0)
