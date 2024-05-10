"""Tool to build optimized buckets out of Quantitative and Qualitative features
for binary classification tasks.
"""

from typing import Callable

from numpy import add, array, searchsorted, sqrt, unique, zeros
from pandas import DataFrame, Series, crosstab
from scipy.stats import chi2_contingency

from ..discretizers.utils.base_discretizers import extend_docstring
from ..features import GroupedList
from .base_carver import BaseCarver


class BinaryCarver(BaseCarver):
    """Automatic carving of continuous, discrete, categorical and ordinal
    features that maximizes association with a binary target.

    Examples
    --------
    `Binary Classification Example <https://autocarver.readthedocs.io/en/latest/examples/
    BinaryClassification/binary_classification_example.html>`_
    """

    @extend_docstring(BaseCarver.__init__)
    def __init__(
        self,
        sort_by: str,
        min_freq: float,
        *,
        quantitatives: list[str] = None,
        categoricals: list[str] = None,
        ordinals: list[str] = None,
        ordinal_values: dict[str, GroupedList] = None,
        max_n_mod: int = 5,
        min_freq_mod: float = None,
        output_dtype: str = "float",
        dropna: bool = True,
        # copy: bool = False,
        # verbose: bool = False,
        # n_jobs: int = 1,
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
                f" - [BinaryCarver] Measure '{sort_by}' not implemented for binary targets. "
                f"Choose from: {str(implemented_measures)}."
            )

        # Initiating BaseCarver
        super().__init__(
            min_freq=min_freq,
            sort_by=sort_by,
            quantitatives=quantitatives,
            categoricals=categoricals,
            ordinals=ordinals,
            ordinal_values=ordinal_values,
            max_n_mod=max_n_mod,
            min_freq_mod=min_freq_mod,
            output_dtype=output_dtype,
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
                " - [BinaryCarver] y must be a binary Series (int or float, not object)"
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
        # checking for empty datasets
        xtabs = {feature.name: None for feature in self.features}
        if X is not None:
            # crosstab for each feature
            for feature in self.features:
                # computing crosstab with str_nan
                xtab = crosstab(X[feature.name], y)

                # reordering according to known_order
                xtab = xtab.reindex(feature.labels)

                # storing results
                xtabs.update({feature: xtab})

        return xtabs

    def _grouper(self, xagg: DataFrame, groupby: list[str]) -> DataFrame:
        """Groups a crosstab by groupby and sums column values by groups (vectorized)

        Parameters
        ----------
        xagg : DataFrame
            crosstab between X and y
        groupby : list[str]
            indices to group by

        Returns
        -------
        DataFrame
            Crosstab grouped by indices
        """
        # all indices that may be duplicated
        index_values = array([groupby.get(index_value, index_value) for index_value in xagg.index])

        # all unique indices deduplicated
        unique_indices = unique(index_values)

        # initiating summed up array with zeros
        summed_values = zeros((len(unique_indices), len(xagg.columns)))

        # for each unique_index found in index_values sums xtab.Values at corresponding position
        # in summed_values
        add.at(summed_values, searchsorted(unique_indices, index_values), xagg.values)

        # converting back to dataframe
        return DataFrame(summed_values, index=unique_indices, columns=xagg.columns)

    def _association_measure(self, xagg: DataFrame, n_obs: int) -> dict[str, float]:
        """Computes measures of association between feature and target by crosstab.

        Parameters
        ----------
        xtab : DataFrame
            Crosstab between feature and target.

        n_obs : int
            Sample total size.

        Returns
        -------
        dict[str, float]
            Cramér's V and Tschuprow's as a dict.
        """
        # number of values taken by the features
        n_mod_x = xagg.shape[0]

        # Chi2 statistic
        chi2 = chi2_contingency(xagg)[0]

        # Cramér's V
        cramerv = sqrt(chi2 / n_obs)

        # Tschuprow's T
        tschuprowt = cramerv / sqrt(sqrt(n_mod_x - 1))

        return {"cramerv": cramerv, "tschuprowt": tschuprowt}

    # def _target_rate(self, xtab: DataFrame) -> DataFrame:
    #     """Computes target rate per row for a binary target (column) in a crosstab

    #     Parameters
    #     ----------
    #     xagg : DataFrame
    #         crosstab of X by y

    #     Returns
    #     -------
    #     DataFrame
    #         y mean by xagg index
    #     """
    #     return xtab[1].divide(xtab.sum(axis=1)).sort_values()

    def _printer(self, xagg: DataFrame = None) -> DataFrame:
        """Prints a binary xtab's statistics

        Parameters
        ----------
        xagg : Dataframe
            A crosstab, by default None

        Returns
        -------
        DataFrame
            Target rate and frequency per modality
        """
        # checking for an xtab
        stats = None
        if xagg is not None:
            # target rate and frequency statistics per modality
            stats = DataFrame(
                {
                    # target rate per modality
                    "target_rate": xagg[1].divide(xagg.sum(axis=1)),
                    # frequency per modality
                    "frequency": xagg.sum(axis=1) / xagg.sum().sum(),
                }
            )

        return stats
