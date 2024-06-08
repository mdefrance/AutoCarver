"""Tool to build optimized buckets out of Quantitative and Qualitative features
for multiclass classification tasks.
"""

from typing import Any, Callable

from pandas import DataFrame, Series, unique

from ..config import DEFAULT, NAN
from ..discretizers import BaseDiscretizer
from ..discretizers.utils.base_discretizer import extend_docstring
from ..features import GroupedList
from .utils.base_carver import BaseCarver
from .binary_carver import BinaryCarver
from ..features import Features
from copy import deepcopy


class MulticlassCarver(BaseCarver):
    """Automatic carving of continuous, discrete, categorical and ordinal
    features that maximizes association with a multiclass target.

    Examples
    --------
    `Multiclass Classification Example <https://autocarver.readthedocs.io/en/latest/examples/
    MulticlassClassification/multiclass_classification_example.html>`_
    """

    __name__ = "MulticlassCarver"

    @extend_docstring(BinaryCarver.__init__)
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
        """ """
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
        # converting target as str
        y_copy = y.astype(str)

        # multiclass target, checking values
        if len(unique(y_copy)) <= 2:
            raise ValueError(
                f" - [{self.__name__}] provided y is binary, consider using BinaryCarver instead."
            )

        # checking for dev target's values
        y_dev_copy = y_dev
        if y_dev is not None:
            # converting target as str
            y_dev_copy = y_dev.astype(str)

            # check that classes of y are classes of y_dev
            unique_y_dev = y_dev.unique()
            unique_y = y.unique()
            missing_y = [mod_y for mod_y in unique_y if mod_y not in unique_y_dev]
            missing_y_dev = [mod_y_dev for mod_y_dev in unique_y_dev if mod_y_dev not in unique_y]
            if len(missing_y) > 0 or len(missing_y_dev) > 0:
                raise ValueError(
                    f" - [{self.__name__}] Value mismatch between y and y_dev:"
                    f"{missing_y_dev+missing_y}"
                )

        # discretizing features
        # x_copy, x_dev_copy = super()._prepare_data(X, y, X_dev=X_dev, y_dev=y_dev)

        return X, y_copy, X_dev, y_dev_copy

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
        casted_history: dict[str, str] = {}
        casted_features = {feature: [] for feature in self.features}

        # iterating over each class minus one
        for n, y_class in enumerate(y_classes):
            if self.verbose:  # verbose if requested
                print(
                    f"\n---------\n[MulticlassCarver] Fit y={y_class} ({n+1}/{len(y_classes)})"
                    "\n------"
                )

            # identifying this y_class
            target_class = get_one_vs_rest(y_copy, y_class)
            target_class_dev = get_one_vs_rest(y_dev_copy, y_class)

            # initiating BinaryCarver for y_class
            binary_carver = BinaryCarver(
                min_freq=self.min_freq,
                sort_by=self.sort_by,
                features=deepcopy(self.features),
                values_orders=raw_values_orders,
                max_n_mod=self.max_n_mod,
                ordinal_encoding=self.ordinal_encoding,
                dropna=self.dropna,
                **dict(self.kwargs, copy=True),  # copying x to keep raw columns as is
            )

            # fitting BinaryCarver for y_class
            binary_carver.fit(x_copy, target_class, X_dev=x_dev_copy, y_dev=target_class_dev)

            # renaming BinaryCarver's fitted values_orders/input_dtype/ordinal_encoding for y_class
            casted_values_orders.update(dict_append_class(binary_carver.values_orders, y_class))
            casted_input_dtypes.update(dict_append_class(binary_carver.input_dtypes, y_class))
            casted_history.update(
                dict_append_class(binary_carver._history, y_class)  # pylint: disable=W0212
            )
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
            ordinal_encoding=self.ordinal_encoding,
            str_nan=self.kwargs.get("nan", NAN),
            str_default=self.kwargs.get("default", DEFAULT),
            dropna=self.dropna,
            copy=self.copy,
            verbose=self.verbose,
            features_casting=casted_features,
            n_jobs=self.n_jobs,
        )
        self._history = casted_history

        # fitting BaseDiscretizer
        BaseDiscretizer.fit(self, x_copy, y_copy)

        return self


def append_class(string: str, to_append: str):
    """add the to_append string to a string"""

    return f"{string}_{to_append}"


def dict_append_class(dic: dict[str, Any], to_append: str):
    """add the to_append string to all keys if dic"""

    return {append_class(feature, to_append): values for feature, values in dic.items()}


def get_one_vs_rest(y: Series, y_class: Any) -> Series:
    """converts a multiclass target into binary of specific y_class"""
    if y is not None:
        return (y == y_class).astype(int)
