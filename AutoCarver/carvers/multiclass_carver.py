"""Tool to build optimized buckets out of Quantitative and Qualitative features
for multiclass classification tasks.
"""

from typing import Any

from pandas import DataFrame, Series, unique

from ..discretizers.utils.base_discretizer import BaseDiscretizer, Sample
from ..features import Features
from ..utils import extend_docstring
from .binary_carver import BinaryCarver
from .utils.base_carver import Samples


class MulticlassCarver(BinaryCarver):
    """Automatic carving of continuous, discrete, categorical and ordinal
    features that maximizes association with a multiclass target.


    Examples
    --------
    `Multiclass Classification Example <https://autocarver.readthedocs.io/en/latest/examples/
    Carvers/MulticlassClassification/multiclass_classification_example.html>`_
    """

    __name__ = "MulticlassCarver"
    is_y_binary = False
    is_y_multiclass = True

    @extend_docstring(BinaryCarver.__init__)
    def __init__(
        self,
        features: Features,
        min_freq: float,
        dropna: bool = True,
        max_n_mod: int = 5,
        **kwargs,
    ) -> None:
        """ """
        # Initiating BinaryCarver
        super().__init__(
            features=features, min_freq=min_freq, dropna=dropna, max_n_mod=max_n_mod, **kwargs
        )

        # warning user for
        if "copy" in kwargs and kwargs["copy"] is True:
            print(
                "WARNING: can't set copy=True for MulticlassCarver (no inplace DataFrame.assign)."
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
        # converting target to str
        samples.train.y = samples.train.y.astype(str)

        # multiclass target, checking values
        if len(unique(samples.train.y)) <= 2:
            raise ValueError(
                f"[{self.__name__}] provided y is binary, consider using BinaryCarver instead."
            )

        # checking for dev target's values
        if samples.dev.y is not None:
            # converting target to str
            samples.dev.y = samples.dev.y.astype(str)

            # check that classes of y are classes of y_dev
            unique_y_dev = samples.dev.y.unique()
            unique_y = samples.train.y.unique()
            missing_y = [mod_y for mod_y in unique_y if mod_y not in unique_y_dev]
            missing_y_dev = [mod_y_dev for mod_y_dev in unique_y_dev if mod_y_dev not in unique_y]
            if len(missing_y) > 0 or len(missing_y_dev) > 0:
                raise ValueError(
                    f"[{self.__name__}] Mismatched classes between y and y_dev"
                    f": train({missing_y_dev}), dev({missing_y})"
                )

        return samples

    @extend_docstring(BinaryCarver.fit)
    def fit(
        self,
        X: DataFrame,
        y: Series,
        *,
        X_dev: DataFrame = None,
        y_dev: Series = None,
    ) -> None:
        # initiating samples
        samples = Samples(train=Sample(X, y), dev=Sample(X_dev, y_dev))

        # preparing datasets and checking for wrong values
        samples = self._prepare_data(samples)

        # getting distinct y classes
        y_classes = sorted(list(samples.train.y.unique()))[1:]  # removing one of the classes

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
            train_y_class = get_one_vs_rest(samples.train.y, y_class)
            dev_y_class = get_one_vs_rest(samples.dev.y, y_class)

            # features for specific group
            class_features = Features(self.features.get_version_group(y_class))

            # initiating BinaryCarver for y_class
            binary_carver = BinaryCarver(
                features=class_features,
                combinations=self.combinations,
                **dict(
                    self.kwargs,
                    copy=True,
                    min_freq=self.min_freq,
                ),  # copying x to keep raw columns as is
            )

            # fitting BinaryCarver for y_class
            binary_carver.fit_transform(
                samples.train.X, train_y_class, X_dev=samples.dev.X, y_dev=dev_y_class
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
        BaseDiscretizer.__init__(
            self,
            features=self.features,
            **dict(
                self.kwargs,
                min_freq=self.min_freq,
                dropna=self.dropna,
                combinations=self.combinations,
            ),
        )

        # fitting BaseDiscretizer
        BaseDiscretizer.fit(self, samples.train.X, samples.train.y)

        return self


def get_one_vs_rest(y: Series, y_class: Any) -> Series:
    """converts a multiclass target into binary of specific y_class"""
    if y is not None:
        return (y == y_class).astype(int)
    return None
