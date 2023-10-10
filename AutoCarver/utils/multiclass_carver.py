"""Tool to build optimized buckets out of Quantitative and Qualitative features
for a binary classification model.
"""

from typing import Any, Callable

from pandas import DataFrame, Series, unique

from .base_carver import BaseCarver
from .binary_carver import BinaryCarver
from ..discretizers import GroupedList, BaseDiscretizer

class MulticlassCarver(BaseCarver):
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
        """Initiates a ``MulticlassCarver``.

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
        # association measure used to find the best groups for multiclass targets
        implemented_measures = ["tschuprowt", "cramerv"]  
        assert sort_by in implemented_measures, (
            f" - [MulticlassCarver] Measure '{sort_by}' not yet implemented for multiclass targets"
            f". Choose from: {str(implemented_measures)}."
        )

        # Initiating BinaryCarver
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
        self.kwargs = kwargs
        self.str_nan = kwargs.get("str_nan", "__NAN__"),
        self.str_default = kwargs.get("str_default", "__OTHER__"),

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

        # converting target as str
        y_copy = y.astype(str)

        # multiclass target, checking values
        y_values = unique(y_copy)
        assert len(y_values) > 2, (
            " - [MulticlassCarver] provided y is binary, consider using BinaryCarver instead."
        )

        if y_dev is not None:
            # converting target as str
            y_dev_copy = y_dev.astype(str)

            # check that classes of y are classes of y_dev
            unique_y_dev = y_dev.unique()
            unique_y = y.unique()
            assert all(mod_y in unique_y_dev for mod_y in unique_y), (
                "- [MulticlassCarver] Some classes of y are missing from y_dev: "
                f"{str([mod_y for mod_y in unique_y if mod_y not in unique_y_dev])}"
            )
            assert all(mod_y_dev in unique_y for mod_y_dev in unique_y_dev), (
                "- [MulticlassCarver] Some classes of y_dev are missing from y: "
                f"{str([mod_y_dev for mod_y_dev in unique_y_dev if mod_y_dev not in unique_y])}"
            )

        return x_copy, y_copy, x_dev_copy, y_dev_copy


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
            Multiclass target feature with wich the association is maximized.

        X_dev : DataFrame, optional
            Dataset to evalute the robustness of discretization, by default None
            It should have the same distribution as X.

        y_dev : Series, optional
            Multiclass target feature with wich the robustness of discretization is evaluated, by default None
        """
        # preparing datasets and checking for wrong values
        x_copy, y_copy, x_dev_copy, y_dev_copy = self._prepare_data(X, y, X_dev, y_dev)
        
        # getting distinct y classes
        y_classes = sorted(list(y_copy.unique()))[1:]  # removing one of the classes

        # copying raw values_orders (contains previous discretizations)
        raw_values_orders = {feature: order for feature, order in self.values_orders.items()}

        # inititing casted features input_dtypes and values_orders
        casted_values_orders: dict[str, GroupedList] = {}
        casted_input_dtypes: dict[str, str] = {}
        casted_features = {feature: [] for feature in self.features}

        # iterating over each class minus one
        for n, y_class in enumerate(y_classes):
            if self.verbose:  # verbose if requested
                print(f"\n---------\n[MulticlassCarver] Fit y={y_class} ({n+1}/{len(y_classes)})\n------")

            # identifying this y_class
            target_class = (y_copy == y_class).astype(int)
            if y_dev is not None:
                target_class_dev = (y_dev_copy == y_class).astype(int)

            # initiating BinaryCarver for y_class
            binary_carver = BinaryCarver(
                min_freq = self.min_freq,
                sort_by = self.sort_by,
                quantitative_features = self.quantitative_features,
                qualitative_features = self.qualitative_features,
                ordinal_features = self.ordinal_features,
                values_orders = raw_values_orders,
                max_n_mod = self.max_n_mod,
                output_dtype = self.output_dtype,
                dropna = self.dropna,
                copy = True,  # copying x to keep raw columns as is
                verbose = self.verbose,
                pretty_print = self.pretty_print,
                **self.kwargs,
            )
    
            # fitting BinaryCarver for y_class
            binary_carver.fit(x_copy, target_class, X_dev=x_dev_copy, y_dev=target_class_dev)
 
            # renaming BinaryCarver's fitted values_orders/input_dtype/output_dtype for y_class
            casted_values_orders.update(dict_append_class(binary_carver.values_orders, y_class))
            casted_input_dtypes.update(dict_append_class(binary_carver.input_dtypes, y_class))
            for feature in binary_carver.features:
                # feature only present in binary_carver.feature if not removed
                casted_features.update(
                    {
                        feature: casted_features.get(feature) + [append_class(feature, y_class)]
                    }
                )

            if self.verbose:  # verbose if requested
                print("---------\n")

        # initiating BaseDiscretizer with features_casting
        base_discretizer = BaseDiscretizer(
            features=[feature for castings in casted_features.values() for feature in castings],
            values_orders=casted_values_orders,
            input_dtypes=casted_input_dtypes,
            output_dtype=self.output_dtype,
            str_nan=self.kwargs.get("str_nan", "__NAN__"),
            str_default=self.kwargs.get("str_default", "__OTHER__"),
            dropna=self.dropna,
            copy=self.copy,
            verbose=bool(max(self.verbose, self.pretty_print)),
            features_casting=casted_features
        )

        # fitting BaseDiscretizer
        base_discretizer.fit(x_copy, y)

        return self

def append_class(string: str, to_append: str):
    """ add the to_append string to a string"""
    
    return f"{string}_{to_append}"

def dict_append_class(dic: dict[str, Any], to_append: str):
    """ add the to_append string to all keys if dic"""
    
    return {append_class(feature, to_append): values for feature, values in dic.items()}
