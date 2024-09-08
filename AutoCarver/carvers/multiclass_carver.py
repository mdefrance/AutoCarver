"""Tool to build optimized buckets out of Quantitative and Qualitative features
for multiclass classification tasks.
"""

from typing import Any, Callable, Union

from pandas import DataFrame, Series, unique

from ..discretizers import BaseDiscretizer
from ..features import Features
from ..utils import extend_docstring
from .binary_carver import BinaryCarver
from .utils.base_carver import BaseCarver


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
                f"[{self.__name__}] Measure '{sort_by}' not implemented for binary targets. "
                f"Choose from: {str(implemented_measures)}."
            )

        # warning user for
        if kwargs.get("copy"):
            print(
                "WARNING: can't set copy=True for MulticlassCarver (no inplace DataFrame.assign)."
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
        # converting target to str
        y_copy = y.astype(str)

        # multiclass target, checking values
        if len(unique(y_copy)) <= 2:
            raise ValueError(
                f"[{self.__name__}] provided y is binary, consider using BinaryCarver instead."
            )

        # checking for dev target's values
        y_dev_copy = y_dev
        if y_dev is not None:
            # converting target to str
            y_dev_copy = y_dev.astype(str)

            # check that classes of y are classes of y_dev
            unique_y_dev = y_dev.unique()
            unique_y = y.unique()
            missing_y = [mod_y for mod_y in unique_y if mod_y not in unique_y_dev]
            missing_y_dev = [mod_y_dev for mod_y_dev in unique_y_dev if mod_y_dev not in unique_y]
            if len(missing_y) > 0 or len(missing_y_dev) > 0:
                raise ValueError(
                    f"[{self.__name__}] Mismatch between y and y_dev: {missing_y_dev+missing_y}"
                )

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

        # adding versionned features
        self.features.add_feature_versions(y_classes)

        # iterating over each class minus one
        for n, y_class in enumerate(y_classes):
            if self.verbose:  # verbose if requested
                print(
                    f"\n---------\n[{self.__name__}] Fit y={y_class} ({n+1}/{len(y_classes)})"
                    "\n------"
                )

            # identifying this y_class
            target_class = get_one_vs_rest(y_copy, y_class)
            target_class_dev = get_one_vs_rest(y_dev_copy, y_class)

            # features for specific group
            class_features = Features(self.features.get_version_group(y_class))

            # initiating BinaryCarver for y_class
            binary_carver = BinaryCarver(
                min_freq=self.min_freq,
                sort_by=self.sort_by,
                features=class_features,
                max_n_mod=self.max_n_mod,
                **dict(self.kwargs, copy=True),  # copying x to keep raw columns as is
            )

            # fitting BinaryCarver for y_class
            binary_carver.fit_transform(
                x_copy, target_class, X_dev=x_dev_copy, y_dev=target_class_dev
            )

            # filtering out dropped features whilst keeping other version tags
            kept_features = binary_carver.features.versions
            kept_features += [
                feature.version for feature in self.features if feature.version_tag != y_class
            ]
            self.features.keep(kept_features)

            if self.verbose:  # verbose if requested
                print("---------\n")

        # initiating BaseDiscretizer with features for each y_class
        BaseDiscretizer.__init__(self, features=self.features, **self.kwargs)

        # fitting BaseDiscretizer
        BaseDiscretizer.fit(self, x_copy, y_copy)

        return self

    def _aggregator(self, X: DataFrame, y: Series) -> Union[Series, DataFrame]:
        """Helper that aggregates X by y into crosstab or means (carver specific)"""
        _, _ = X, y

    def _association_measure(self, xagg: DataFrame, n_obs: int) -> Union[Series, DataFrame]:
        """Helper to measure association between X and y (carver specific)"""
        _, _ = xagg, n_obs

    def _grouper(self, xagg: DataFrame, groupby: list[str]) -> DataFrame:
        """Helper to group XAGG's values by groupby (carver specific)"""
        _, _ = xagg, groupby

    def _printer(self, xagg: DataFrame = None) -> DataFrame:
        """helper to print an XAGG (carver specific)"""
        _ = xagg


def get_one_vs_rest(y: Series, y_class: Any) -> Series:
    """converts a multiclass target into binary of specific y_class"""
    if y is not None:
        return (y == y_class).astype(int)
