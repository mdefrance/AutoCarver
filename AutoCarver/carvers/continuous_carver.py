"""Tool to build optimized buckets out of Quantitative and Qualitative features
for continuous regression tasks.
"""

from numpy import mean, unique
from pandas import DataFrame, Series

from ..features import Features, BaseFeature
from ..utils import extend_docstring
from .utils.base_carver import BaseCarver


class ContinuousCarver(BaseCarver):
    """Automatic carving of continuous, discrete, categorical and ordinal
    features that maximizes association with a continuous target.

    For continuous targets, :ref:`Kruskal` is used as association measure to sort combinations.

    Examples
    --------
    `Continuous Regression Example <https://autocarver.readthedocs.io/en/latest/examples/Multiclass
    Classification/multiclass_classification_example.html>`_
    """

    __name__ = "ContinuousCarver"

    @extend_docstring(BaseCarver.__init__)
    def __init__(
        self,
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
        """
        # association measure used to find the best groups for continuous targets
        if "sort_by" in kwargs and kwargs.get("sort_by") != "kruskal":
            raise ValueError(
                f"[{self.__name__}] Measure '{kwargs.get('sort_by')}' not implemented for "
                "continuous targets. Use 'kruskal' instead."
            )

        # Initiating BaseCarver
        super().__init__(
            min_freq=min_freq,
            sort_by="kruskal",
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
    ) -> tuple[DataFrame, DataFrame]:
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
        # Checking for binary target and copying X
        x_copy, x_dev_copy = super()._prepare_data(X, y, X_dev=X_dev, y_dev=y_dev)

        # continuous target, checking values
        y_values = unique(y)
        if len(y_values) <= 2:
            raise ValueError(
                f"[{self.__name__}] provided y is binary, consider using BinaryCarver instead."
            )
        if str in y.apply(type).unique():
            raise ValueError(
                f"[{self.__name__}] y must be a continuous Series (int or float, not object)"
            )

        return x_copy, x_dev_copy

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

    def _target_rate(self, xagg: Series) -> Series:
        """Computes target average per group for a continuous target

        Parameters
        ----------
        xagg : Series
            _description_

        Returns
        -------
        Series
            _description_
        """
        return xagg.apply(mean).sort_values()


def get_target_values_by_modality(X: DataFrame, y: Series, feature: BaseFeature) -> dict:
    """Computes y values for modalities of specified features and ensures the ordering
    according to the known labels"""

    # list of y values for each modality of X
    yval = y.groupby(X[feature.version]).apply(lambda u: list(u))

    # reindexing to ensure the right order
    return yval.reindex(feature.labels, fill_value=[])
