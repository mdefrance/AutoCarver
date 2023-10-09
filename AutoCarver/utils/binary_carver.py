"""Tool to build optimized buckets out of Quantitative and Qualitative features
for a binary classification model.
"""

from typing import Callable

from numpy import add, array, searchsorted, sqrt, zeros
from pandas import DataFrame, Series, crosstab, unique
from scipy.stats import chi2_contingency

from .base_carver import BaseCarver
from ..discretizers import GroupedList

class BinaryCarver(BaseCarver):
    """Automatic carving of continuous, discrete, categorical and ordinal
    features that maximizes association with a binary or continuous target.

    First fits a :ref:`Discretizer`. Raw data should be provided as input (not a result of ``Discretizer.transform()``).
    """

    def __init__(
        self,
        min_freq: float,
        sort_by: str,
        *,
        quantitative_features: list[str] = None,
        qualitative_features: list[str] = None,
        ordinal_features: list[str] = None,
        values_orders: dict[str, GroupedList] = None,
        max_n_mod: int = 5,
        output_dtype: str = "float",
        dropna: bool = True,
        copy: bool = False,
        verbose: bool = False,
        pretty_print: bool = False,
        **kwargs,
    ) -> None:
        """Initiates a ``BinaryCarver``.

        Parameters
        ----------
        min_freq : float
            Minimum frequency per grouped modalities.

            * Features whose most frequent modality is less frequent than ``min_freq`` will not be carved.
            * Sets the number of quantiles in which to discretize the continuous features.
            * Sets the minimum frequency of a quantitative feature's modality.

            **Tip**: should be set between 0.02 (slower, preciser, less robust) and 0.05 (faster, more robust)

        sort_by : str
            To be choosen amongst ``["tschuprowt", "cramerv", "kruskal"]``
            Metric to be used to perform association measure between features and target.

            * Binary target: use ``"tschuprowt"``, for Tschuprow's T.
            * Binary target: use ``"cramerv"``, for Cramér's V.
            * Continuous target: use ``"kruskal"``, for Kruskal-Wallis' H test statistic.

            **Tip**: use ``"tschuprowt"`` for more robust, or less output modalities,
            use ``"cramerv"`` for more output modalities.

        quantitative_features : list[str], optional
            List of column names of quantitative features (continuous and discrete) to be carved, by default ``None``

        qualitative_features : list[str], optional
            List of column names of qualitative features (non-ordinal) to be carved, by default ``None``

        ordinal_features : list[str], optional
            List of column names of ordinal features to be carved. For those features a list of
            values has to be provided in the ``values_orders`` dict, by default ``None``

        values_orders : dict[str, GroupedList], optional
            Dict of feature's column names and there associated ordering.
            If lists are passed, a GroupedList will automatically be initiated, by default ``None``

        max_n_mod : int, optional
            Maximum number of modality per feature, by default ``5``

            All combinations of modalities for groups of modalities of sizes from 1 to ``max_n_mod`` will be tested.
            The combination with the greatest association (as defined by ``sort_by``) will be the selected one.

            **Tip**: should be set between 4 (faster, more robust) and 7 (slower, preciser, less robust)

        output_dtype : str, optional
            To be choosen amongst ``["float", "str"]``, by default ``"float"``

            * ``"float"``, grouped modalities will be converted to there corresponding floating rank.
            * ``"str"``, a per-group modality will be set for all the modalities of a group.

        dropna : bool, optional
            * ``True``, ``AutoCarver`` will try to group ``numpy.nan`` with other modalities.
            * ``False``, ``AutoCarver`` all non-``numpy.nan`` will be grouped, by default ``True``

        copy : bool, optional
            If ``True``, feature processing at transform is applied to a copy of the provided DataFrame, by default ``False``

        verbose : bool, optional
            If ``True``, prints raw Discretizers Fit and Transform steps, as long as
            information on AutoCarver's processing and tables of target rates and frequencies for
            X, by default ``False``

        pretty_print : bool, optional
            If ``True``, adds to the verbose some HTML tables of target rates and frequencies for X and, if provided, X_dev.
            Overrides the value of ``verbose``, by default ``False``

        **kwargs
            Pass values for ``str_default``and ``str_nan`` of ``Discretizer`` (default string values).

        Examples
        --------
        See `AutoCarver examples <https://autocarver.readthedocs.io/en/latest/index.html>`_
        """
        # association measure used to find the best groups for binary targets
        implemented_measures = ["tschuprowt", "cramerv"]  
        assert sort_by in implemented_measures, (
            f" - [BinaryCarver] Measure '{sort_by}' not yet implemented for binary targets. "
            f"Choose from: {str(implemented_measures)}."
        )

        # Initiating BaseCarver
        super().__init__(
            min_freq = min_freq,
            sort_by = sort_by,
            quantitative_features = quantitative_features,
            qualitative_features = qualitative_features,
            ordinal_features = ordinal_features,
            values_orders = values_orders,
            max_n_mod = max_n_mod,
            output_dtype = output_dtype,
            dropna = dropna,
            copy = copy,
            verbose = verbose,
            pretty_print = pretty_print,
            **kwargs
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
        tuple[DataFrame, DataFrame, dict[str, Callable]]
            Copies of (X, X_dev) and helpers to be used according to target type
        """
        # Checking for binary target and copying X
        x_copy, x_dev_copy = super()._prepare_data(X, y, X_dev=X_dev, y_dev=y_dev)

        # binary target, checking values
        y_values = unique(y)
        assert (0 in y_values) & (1 in y_values), (
            " - [BinaryCarver] y must be a binary Series (int or float, not object)"
        )
        assert len(y_values) == 2, (
            " - [BinaryCarver] y must be a binary Series (int or float, not object)"
        )

        return x_copy, x_dev_copy

    def _aggregator(
        self, features: list[str], X: DataFrame, y: Series, labels_orders: dict[str, GroupedList]
    ) -> dict[str, DataFrame]:
        """Computes crosstabs for specified features and ensures that the crosstab is ordered according to the known labels

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
        xtabs = {feature: None for feature in features}
        if X is not None:
            # crosstab for each feature
            for feature in features:
                # computing crosstab with str_nan
                xtab = crosstab(X[feature], y)

                # reordering according to known_order
                xtab = xtab.reindex(labels_orders[feature])

                # storing results
                xtabs.update({feature: xtab})

        return xtabs


    def _grouper(self, xtab: DataFrame, groupby: list[str]) -> DataFrame:
        """Groups a crosstab by groupby and sums column values by groups (vectorized)

        Parameters
        ----------
        xtab : DataFrame
            _description_
        groupby : list[str]
            _description_

        Returns
        -------
        DataFrame
            _description_
        """
        # all indices that may be duplicated
        index_values = array(groupby)

        # all unique indices deduplicated
        unique_indices = unique(index_values)

        # initiating summed up array with zeros
        summed_values = zeros((len(unique_indices), len(xtab.columns)))

        # for each unique_index found in index_values sums xtab.Values at corresponding position in summed_values
        add.at(summed_values, searchsorted(unique_indices, index_values), xtab.values)

        # converting back to dataframe
        grouped_xtab = DataFrame(summed_values, index=unique_indices, columns=xtab.columns)

        return grouped_xtab

    def _association_measure(self, xtab: DataFrame, n_obs: int) -> dict[str, float]:
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
        n_mod_x = xtab.shape[0]

        # Chi2 statistic
        chi2 = chi2_contingency(xtab)[0]

        # Cramér's V
        cramerv = sqrt(chi2 / n_obs)

        # Tschuprow's T
        tschuprowt = cramerv / sqrt(sqrt(n_mod_x - 1))

        return {"cramerv": cramerv, "tschuprowt": tschuprowt}

    def _target_rate(self, xtab: DataFrame) -> DataFrame:
        """Computes target rate per row for a binary target (column) in a crosstab

        Parameters
        ----------
        xtab : DataFrame
            _description_

        Returns
        -------
        DataFrame
            _description_
        """
        return xtab[1].divide(xtab.sum(axis=0)).sort_values()

    def _combination_formatter(self, combination: list[list[str]]) -> list[str]:
        formatted_combination = [
            value for values in ([group[0]] * len(group) for group in combination) for value in values
        ]

        return formatted_combination

    def _printer(self, xtab: DataFrame = None) -> DataFrame:
        """Prints a binary xtab's statistics

        Parameters
        ----------
        xtab : Dataframe
            A crosstab, by default None

        Returns
        -------
        DataFrame
            Target rate and frequency per modality
        """
        # checking for an xtab
        stats = None
        if xtab is not None:
            # target rate and frequency statistics per modality
            stats = DataFrame(
                {
                    # target rate per modality
                    "target_rate": xtab[1].divide(xtab.sum(axis=1)),
                    # frequency per modality
                    "frequency": xtab.sum(axis=1) / xtab.sum().sum(),
                }
            )

            # rounding up stats
            stats = stats.round(3)

        return stats

    def fit(
        self,
        X: DataFrame,
        y: Series,
        *,
        X_dev: DataFrame = None,
        y_dev: Series = None,
    ) -> None:
        """Finds the combination of modalities of X that provides the best association with y.

        Parameters
        ----------
        X : DataFrame
            Dataset used to discretize. Needs to have columns has specified in ``AutoCarver.features``.

        y : Series
            Binary target feature with wich the association is maximized.

        X_dev : DataFrame, optional
            Dataset to evalute the robustness of discretization, by default None
            It should have the same distribution as X.

        y_dev : Series, optional
            Binary target feature with wich the robustness of discretization is evaluated, by default None
        """
        # preparing datasets and checking for wrong values
        x_copy, x_dev_copy = self._prepare_data(X, y, X_dev, y_dev)

        # Fitting BaseCarver
        super().fit(x_copy, y, X_dev=x_dev_copy, y_dev=y_dev)

        return self
