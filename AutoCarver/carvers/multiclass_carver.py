"""Tool to build optimized buckets out of Quantitative and Qualitative features
for multiclass classification tasks.
"""

from typing import Any, Callable

from pandas import DataFrame, Series, unique

from ..discretizers import BaseDiscretizer, GroupedList
from ..discretizers.utils.base_discretizers import extend_docstring
from .base_carver import BaseCarver
from .binary_carver import BinaryCarver


class MulticlassCarver(BaseCarver):
    """Automatic carving of continuous, discrete, categorical and ordinal
    features that maximizes association with a multiclass target.
    """

    @extend_docstring(BinaryCarver.__init__)
    def __init__(
        self,
        sort_by: str,
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
        **kwargs,
    ) -> None:
        # association measure used to find the best groups for multiclass targets
        implemented_measures = ["tschuprowt", "cramerv"]
        assert sort_by in implemented_measures, (
            f" - [MulticlassCarver] Measure '{sort_by}' not yet implemented for multiclass targets"
            f". Choose from: {str(implemented_measures)}."
        )

        # Initiating BinaryCarver
        super().__init__(
            min_freq=min_freq,
            sort_by=sort_by,
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
        self.kwargs = kwargs

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
        assert (
            len(y_values) > 2
        ), " - [MulticlassCarver] provided y is binary, consider using BinaryCarver instead."

        y_dev_copy = y_dev
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

    @extend_docstring(BinaryCarver.fit)
    def fit(
        self,
        X: DataFrame,
        y: Series,
        *,
        X_dev: DataFrame = None,
        y_dev: Series = None,
    ) -> None:
        # preparing datasets and checking for wrong values
        x_copy, y_copy, x_dev_copy, y_dev_copy = self._prepare_data(X, y, X_dev, y_dev)

        # getting distinct y classes
        y_classes = sorted(list(y_copy.unique()))[1:]  # removing one of the classes

        # copying raw values_orders (contains previous discretizations)
        raw_values_orders = {feature: order for feature, order in self.values_orders.items()}

        # initiating casted features input_dtypes and values_orders
        casted_values_orders: dict[str, GroupedList] = {}
        casted_input_dtypes: dict[str, str] = {}
        casted_features = {feature: [] for feature in self.features}

        # iterating over each class minus one
        for n, y_class in enumerate(y_classes):
            if self.verbose:  # verbose if requested
                print(
                    f"\n---------\n[MulticlassCarver] Fit y={y_class} ({n+1}/{len(y_classes)})\n------"
                )

            # identifying this y_class
            target_class = (y_copy == y_class).astype(int)
            target_class_dev = None
            if y_dev_copy is not None:
                target_class_dev = (y_dev_copy == y_class).astype(int)

            # initiating BinaryCarver for y_class
            binary_carver = BinaryCarver(
                min_freq=self.min_freq,
                sort_by=self.sort_by,
                quantitative_features=self.quantitative_features,
                qualitative_features=self.qualitative_features,
                ordinal_features=self.ordinal_features,
                values_orders=raw_values_orders,
                max_n_mod=self.max_n_mod,
                output_dtype=self.output_dtype,
                dropna=self.dropna,
                copy=True,  # copying x to keep raw columns as is
                verbose=self.verbose,
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
                    {feature: casted_features.get(feature) + [append_class(feature, y_class)]}
                )

            if self.verbose:  # verbose if requested
                print("---------\n")

        # initiating BaseDiscretizer with features_casting
        BaseDiscretizer.__init__(
            self,
            features=[feature for castings in casted_features.values() for feature in castings],
            values_orders=casted_values_orders,
            input_dtypes=casted_input_dtypes,
            output_dtype=self.output_dtype,
            str_nan=self.kwargs.get("str_nan", "__NAN__"),
            str_default=self.kwargs.get("str_default", "__OTHER__"),
            dropna=self.dropna,
            copy=self.copy,
            verbose=self.verbose,
            features_casting=casted_features,
        )

        # fitting BaseDiscretizer
        BaseDiscretizer.fit(self, x_copy, y_copy)

        return self


def append_class(string: str, to_append: str):
    """add the to_append string to a string"""

    return f"{string}_{to_append}"


def dict_append_class(dic: dict[str, Any], to_append: str):
    """add the to_append string to all keys if dic"""

    return {append_class(feature, to_append): values for feature, values in dic.items()}
