"""Tool to build optimized buckets out of Quantitative and Qualitative features
for continuous regression tasks.
"""

from typing import Callable

from numpy import mean, unique
from pandas import DataFrame, Series
from scipy.stats import kruskal

from ..discretizers import GroupedList
from ..discretizers.utils.base_discretizers import extend_docstring
from .base_carver import BaseCarver


class ContinuousCarver(BaseCarver):
    """Automatic carving of continuous, discrete, categorical and ordinal
    features that maximizes association with a continuous target.

    For continuous targets, :ref:`Kruskal` is used as association measure to sort combinations.

    Examples
    --------
    `Continuous Regression Example <https://autocarver.readthedocs.io/en/latest/examples/MulticlassClassification/multiclass_classification_example.html>`_
    """

    @extend_docstring(BaseCarver.__init__)
    def __init__(
        self,
        min_freq: float,
        *,
        quantitative_features: list[str] = None,
        qualitative_features: list[str] = None,
        ordinal_features: list[str] = None,
        values_orders: dict[str, GroupedList] = None,
        max_n_mod: int = 5,
        min_freq_mod: float = None,
        output_dtype: str = "float",
        dropna: bool = True,
        copy: bool = False,
        verbose: bool = False,
        **kwargs: dict,
    ) -> None:
        """
        Parameters
        ----------
        """
        # association measure used to find the best groups for continuous targets
        if "sort_by" in kwargs:
            assert kwargs.get("sort_by") == "kruskal", (
                f" - [ContinuousCarver] Measure '{kwargs.get('sort_by')}' not yet implemented for "
                "continuous targets. Only 'kruskal' measure is implemented for continuous targets."
            )
            kwargs.pop("sort_by")

        # Initiating BaseCarver
        super().__init__(
            min_freq=min_freq,
            sort_by="kruskal",
            quantitative_features=quantitative_features,
            qualitative_features=qualitative_features,
            ordinal_features=ordinal_features,
            values_orders=values_orders,
            max_n_mod=max_n_mod,
            min_freq_mod=min_freq_mod,
            output_dtype=output_dtype,
            dropna=dropna,
            copy=copy,
            verbose=verbose,
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
            Dataset used to discretize. Needs to have columns has specified in ``AutoCarver.features``.

        y : Series
            Binary target feature with wich the association is maximized.

        X_dev : DataFrame, optional
            Dataset to evalute the robustness of discretization, by default ``None``
            It should have the same distribution as X.

        y_dev : Series, optional
            Binary target feature with wich the robustness of discretization is evaluated, by default ``None``

        Returns
        -------
        tuple[DataFrame, DataFrame]
            Copies of (X, X_dev) to be used according to target type
        """
        # Checking for binary target and copying X
        x_copy, x_dev_copy = super()._prepare_data(X, y, X_dev=X_dev, y_dev=y_dev)

        # continuous target, checking values
        y_values = unique(y)
        assert (
            len(y_values) > 2
        ), " - [ContinuousCarver] provided y is binary, consider using BinaryCarver instead."
        not_numeric = str in y.apply(type).unique()
        assert (
            not not_numeric
        ), " - [ContinuousCarver] y must be a continuous Series (int or float, not object)"

        return x_copy, x_dev_copy

    def _aggregator(
        self, features: list[str], X: DataFrame, y: Series, labels_orders: dict[str, GroupedList]
    ) -> dict[str, DataFrame]:
        """Computes y values for modalities of specified features and ensures the ordering according to the known labels

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
        yvals = {feature: None for feature in features}
        if X is not None:
            # crosstab for each feature
            for feature in features:
                # computing crosstab with str_nan
                yval = y.groupby(X[feature]).apply(lambda u: list(u))

                # reordering according to known_order
                yval = yval.reindex(labels_orders[feature])

                # storing results
                yvals.update({feature: yval})

        return yvals

    def _grouper(self, yval: Series, groupby: dict[str:str]) -> Series:
        """Groups values of y

        Parameters
        ----------
        yval : Series
            _description_
        groupby : _type_
            _description_

        Returns
        -------
        Series
            _description_
        """
        # TODO: convert this to the vectorial version like BinaryCarver
        return yval.groupby(groupby).sum()

    def _association_measure(self, yval: Series, **kwargs) -> dict[str, float]:
        """Computes measures of association between feature and quantitative target.

        Parameters
        ----------
        yval : DataFrame
            Values taken by y for each of x's modalities.

        Returns
        -------
        dict[str, float]
            Kruskal-Wallis' H as a dict.
        """
        # Kruskal-Wallis' H
        return {"kruskal": kruskal(*tuple(yval.values))[0]}

    def _target_rate(self, yval: Series) -> Series:
        """Computes target average per group for a continuous target

        Parameters
        ----------
        yval : Series
            _description_

        Returns
        -------
        Series
            _description_
        """
        return yval.apply(mean).sort_values()

    def _printer(self, yval: Series = None) -> DataFrame:
        """Prints a continuous yval's statistics

        Parameters
        ----------
        yval : Series
            A series of values of y by modalities of x, by default None

        Returns
        -------
        DataFrame
            Target rate and frequency per modality
        """
        # checking for an xtab
        stats = None
        if yval is not None:
            # target rate and frequency statistics per modality
            stats = DataFrame(
                {
                    # target rate per modality
                    "target_rate": yval.apply(mean),
                    # frequency per modality
                    "frequency": yval.apply(len) / yval.apply(len).sum(),
                }
            )

        return stats

    @extend_docstring(BaseCarver.fit)
    def fit(
        self,
        X: DataFrame,
        y: Series,
        *,
        X_dev: DataFrame = None,
        y_dev: Series = None,
    ) -> None:
        # preparing datasets and checking for wrong values
        x_copy, x_dev_copy = self._prepare_data(X, y, X_dev, y_dev)

        # Fitting BaseCarver
        super().fit(x_copy, y, X_dev=x_dev_copy, y_dev=y_dev)

        return self
