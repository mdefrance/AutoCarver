"""Tool to build optimized buckets out of Quantitative and Qualitative features
for multiclass classification tasks.
"""

from dataclasses import replace
from typing import Any, Self

import pandas as pd

from AutoCarver.carvers.binary_carver import BinaryCarver
from AutoCarver.carvers.utils.base_carver import Samples
from AutoCarver.combinations import CombinationEvaluator
from AutoCarver.discretizers.utils.base_discretizer import BaseDiscretizer, DiscretizerConfig, Sample
from AutoCarver.features import Features
from AutoCarver.utils import extend_docstring


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
        max_n_mod: int,
        *,
        combination_evaluator: CombinationEvaluator | None = None,
        config: DiscretizerConfig | None = None,
    ) -> None:
        """ """
        super().__init__(
            features=features,
            min_freq=min_freq,
            max_n_mod=max_n_mod,
            combination_evaluator=combination_evaluator,
            config=config,
        )

        # multiclass cannot copy inplace
        if self.config.copy:
            print("WARNING: can't set copy=True for MulticlassCarver (no inplace DataFrame.assign).")

    def _prepare_data(self, samples: Samples) -> Samples:
        """Validates format and content of X and y."""
        # converting target to str (y is required by Carver.fit)
        if samples.train.y is None:
            raise ValueError(f"[{self.__name__}] y must be provided")
        samples.train.y = samples.train.y.astype(str)

        # multiclass target, checking values
        if len(pd.unique(samples.train.y)) <= 2:
            raise ValueError(f"[{self.__name__}] provided y is binary, consider using BinaryCarver instead.")

        # checking for dev target's values
        if samples.dev.y is not None:
            samples.dev.y = samples.dev.y.astype(str)

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
        X: pd.DataFrame,
        y: pd.Series,
        *,
        X_dev: pd.DataFrame | None = None,
        y_dev: pd.Series | None = None,
    ) -> Self:
        # initiating samples
        samples = Samples(train=Sample(X, y), dev=Sample(X_dev, y_dev))

        # preparing datasets and checking for wrong values
        samples = self._prepare_data(samples)

        # getting distinct y classes (_prepare_data raises if y is missing)
        # removing one of the classes
        y_classes = sorted(samples.train.y.unique().tolist())[1:]  # type: ignore

        # adding versionned features
        self.features.add_feature_versions(y_classes)

        # iterating over each class minus one
        for n, y_class in enumerate(y_classes):
            if self.config.verbose:
                print(f"\n---------\n[{self.__name__}] Fit y={y_class} ({n + 1}/{len(y_classes)})\n------")

            train_y_class = get_one_vs_rest(samples.train.y, y_class)
            dev_y_class = get_one_vs_rest(samples.dev.y, y_class)

            class_features = Features.from_list(self.features.get_version_group(y_class))

            # spawn a BinaryCarver per class; copy X so each carver sees clean raw columns.
            # Each child rebuilds its own evaluator from the same class + max_n_mod, so
            # runtime state stays isolated per class fit.
            # fresh evaluator instance per class fit so runtime state (samples,
            # _feature) doesn't leak across iterations.
            binary_carver = BinaryCarver(
                features=class_features,
                min_freq=self.min_freq,
                max_n_mod=self.max_n_mod,
                combination_evaluator=type(self.combination_evaluator)(),
                config=replace(self.config, copy=True),
            )

            binary_carver.fit_transform(
                samples.train.X,
                train_y_class,
                X_dev=samples.dev.X if samples.dev.has_X else None,
                y_dev=dev_y_class,
            )

            # filtering out dropped features whilst keeping other version tags
            kept_features = binary_carver.features.versions
            kept_features += [feature.version for feature in self.features if feature.version_tag != y_class]
            self.features.keep(kept_features)

            if self.config.verbose:
                print("---------\n")

        # re-init BaseDiscretizer state to reflect the final multiclass features,
        # then mark fitted. Preserve combination_evaluator (BaseDiscretizer.__init__
        # resets self.combinations to None).
        combination_evaluator = self.combination_evaluator
        BaseDiscretizer.__init__(self, features=self.features, min_freq=self.min_freq, config=self.config)
        self.combination_evaluator = combination_evaluator
        self.combinations = combination_evaluator

        BaseDiscretizer.fit(self, samples.train.X, samples.train.y)

        return self


def get_one_vs_rest(y: pd.Series | None, y_class: Any) -> pd.Series | None:
    """converts a multiclass target into binary of specific y_class"""
    if y is not None:
        return (y == y_class).astype(int)
    return None
