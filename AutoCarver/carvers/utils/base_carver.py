"""Tool to build optimized buckets out of Quantitative and Qualitative features
for any task.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from functools import partial
from multiprocessing import Pool
from typing import Self

import pandas as pd

from AutoCarver.carvers.utils.pretty_print import index_mapper, prettier_xagg
from AutoCarver.combinations import (
    CombinationEvaluator,
    CramervCombinations,
    KruskalCombinations,
    TschuprowtCombinations,
)
from AutoCarver.discretizers import BaseDiscretizer, Discretizer, Sample
from AutoCarver.discretizers.utils.base_discretizer import DiscretizerConfig
from AutoCarver.features import BaseFeature, Features
from AutoCarver.features.qualitatives import CategoricalFeature, OrdinalFeature
from AutoCarver.features.quantitatives import QuantitativeFeature
from AutoCarver.utils import extend_docstring, has_idisplay

# trying to import extra dependencies
_has_idisplay = has_idisplay()
if _has_idisplay:
    from IPython.display import display_html


@dataclass
class Samples:
    """
    A container for storing training and development samples.

    Attributes:
        train (Sample): The training sample, containing features (X) and target (y).
        dev (Sample): The development sample, containing features (X) and target (y).
    """

    train: Sample = field(default_factory=Sample)
    dev: Sample = field(default_factory=Sample)

    def fillna(self, features: Features) -> None:
        """fills up nans in X and X_dev"""
        self.train.X = features.fillna(self.train.X)
        if self.dev.has_X:
            self.dev.X = features.fillna(self.dev.X)


def _carve_feature_worker(
    payload: tuple[BaseFeature, pd.Series | pd.DataFrame | None, pd.Series | pd.DataFrame | None],
    *,
    evaluator: CombinationEvaluator,
    max_n_mod: int,
    min_freq: float,
    dropna: bool,
) -> tuple[BaseFeature, bool]:
    """Picklable worker: scores best combination for a single feature.

    Each pool task receives a pickled deep copy of ``evaluator`` and a single
    ``(feature, xagg, xagg_dev)`` triple; mutations stay local to the worker
    process. The parent reattaches the returned (mutated) feature to its
    ``Features`` container.
    """
    feature, xagg, xagg_dev = payload
    # workers never print per-feature progress; the parent prints a single banner
    evaluator.verbose = False
    best = evaluator.get_best_combination(
        feature, xagg, xagg_dev, max_n_mod=max_n_mod, min_freq=min_freq, dropna=dropna
    )
    return feature, best is not None


def _replace_feature_in_features(features: Features, updated: BaseFeature) -> None:
    """Swaps an existing feature (by version) for the worker-returned copy."""
    if isinstance(updated, CategoricalFeature):
        categoricals = features.categoricals
        for i, existing in enumerate(categoricals):
            if existing.version == updated.version:
                categoricals[i] = updated
                return
    elif isinstance(updated, OrdinalFeature):
        ordinals = features.ordinals
        for i, existing in enumerate(ordinals):
            if existing.version == updated.version:
                ordinals[i] = updated
                return
    elif isinstance(updated, QuantitativeFeature):
        quantitatives = features.quantitatives
        for i, existing in enumerate(quantitatives):
            if existing.version == updated.version:
                quantitatives[i] = updated
                return
    raise KeyError(f"[BaseCarver] feature {updated.version!r} not in Features")


class BaseCarver(BaseDiscretizer, ABC):
    """Automatic carving of continuous, discrete, categorical and ordinal
    features that maximizes association with a binary or continuous target.

    First fits a :class:`Discretizer`. Raw data should be provided as input (not a result of
    ``Discretizer.transform()``).
    """

    __name__ = "AutoCarver"
    is_y_binary = False
    is_y_continuous = False
    is_y_multiclass = False

    @extend_docstring(BaseDiscretizer.__init__)
    def __init__(
        self,
        features: Features,
        min_freq: float,
        max_n_mod: int,
        *,
        combination_evaluator: CombinationEvaluator | None = None,
        config: DiscretizerConfig | None = None,
    ) -> None:
        """
        Parameters
        ----------

        max_n_mod : int
            Maximum number of modalities per feature. Forwarded to the configured
            :class:`CombinationEvaluator`.

            * The combination with the best association will be selected.
            * All combinations of sizes from 1 to :attr:`max_n_mod` are tested out.

            .. tip::
                Set between ``3`` (faster, more robust) and ``7`` (slower, less robust)

        combination_evaluator : CombinationEvaluator, optional
            Pre-built :class:`CombinationEvaluator` instance used to measure
            association. Subclasses default this to a task-appropriate instance
            (e.g. :class:`TschuprowtCombinations` for binary). The carver
            forwards ``verbose`` onto the instance and passes ``max_n_mod`` /
            ``min_freq`` / ``dropna`` directly to each
            :meth:`~CombinationEvaluator.get_best_combination` call.

        config : DiscretizerConfig, optional
            Behavioral toggles inherited from :class:`BaseDiscretizer`. Defaults
            to ``DiscretizerConfig(dropna=True, ordinal_encoding=True)`` which are
            the carver-friendly defaults (group ``nan``, ordinal-encode features
            for downstream sklearn estimators).
        """
        if combination_evaluator is None:
            raise ValueError(
                f"[{self.__name__}] combination_evaluator must be provided (subclasses set a task-appropriate default)."
            )

        # carver-friendly defaults differ from BaseDiscretizer's (which default
        # both to False): carvers group nans and ordinal-encode by default.
        if config is None:
            config = DiscretizerConfig(dropna=True, ordinal_encoding=True)
        super().__init__(features, min_freq=min_freq, config=config)

        self.max_n_mod = max_n_mod
        combination_evaluator.verbose = self.config.verbose
        self.combination_evaluator: CombinationEvaluator = combination_evaluator

    @property
    def pretty_print(self) -> bool:
        """Returns the pretty_print attribute"""
        return self.config.verbose and _has_idisplay

    def to_json(self, light_mode: bool = False) -> dict:
        content = super().to_json(light_mode)
        content["max_n_mod"] = self.max_n_mod
        content["combination_evaluator"] = self.combination_evaluator.to_json()
        return content

    def _prepare_data(self, samples: Samples) -> Samples:  # type: ignore
        # Why: deliberate signature divergence vs `BaseDiscretizer._prepare_data(sample: Sample)`.
        # Carvers operate on the train+dev `Samples` pair (the parent only knows about a single
        # `Sample`); this method is internal and never invoked through a `BaseDiscretizer` handle.
        """Validates format and content of X and y."""
        if samples.train.y is None:
            raise ValueError(f"[{self.__name__}] y must be provided, got {samples.train.y}")

        # Checking for binary target and copying X
        samples.train = super()._prepare_data(samples.train)
        samples.dev = super()._prepare_data(samples.dev)

        # discretizing features at half min_freq so the carver has a finer
        # granularity to combine when forming optimal groups
        samples = discretize(self.features, samples, self.min_freq / 2, self.config)

        # setting dropna to True for filling up nans
        self.features.dropna = True

        # filling up nans
        samples.fillna(self.features)

        return samples

    def fit(  # type: ignore  # pylint: disable=W0222
        # Why: deliberate signature divergence vs `BaseDiscretizer.fit(X=None, y=None)`. Carvers
        # require X/y and additionally accept `X_dev`/`y_dev` for robustness checks; the W0222
        # pylint disable already records that intent for the linter, this mirror records it for ty.
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        X_dev: pd.DataFrame | None = None,
        y_dev: pd.Series | None = None,
    ) -> Self:
        """Finds the combination of modalities of X that provides the best association with y.
        If provided, X_dev set should be large enough to have the same distribution as X.

        Parameters
        ----------
        X : pd.DataFrame
            Dataset to determine :class:`Features`' optimal carving.

        y : pd.Series
            Target with wich the association is maximized.

        X_dev : pd.DataFrame, optional
            Dataset to evaluate robustness of :class:`Features`, by default ``None``

        y_dev : pd.Series, optional
            Target associated to ``X_dev``, by default ``None``
        """

        # checking for fitted features
        if self.features.is_fitted:
            raise ValueError(
                f"[{self.__name__}] features are already fitted or previous fit failed. Please reset your features."
            )

        # setting is_fitted
        self.features.is_fitted = True

        # initiating samples
        samples = Samples(Sample(X, y), Sample(X_dev, y_dev))

        # preparing datasets and checking for wrong values
        samples = self._prepare_data(samples)

        # logging if requested
        super()._log_if_verbose("---------\n------")

        # computing crosstabs for each feature on train/test
        xaggs = self._aggregator(**samples.train)
        xaggs_dev = self._aggregator(**samples.dev)

        # getting all features to carve (features are removed from self.features)
        all_features = self.features.versions

        # carving each feature (parallel across features when n_jobs > 1)
        if self.config.n_jobs > 1 and len(all_features) > 1:
            self._carve_features_parallel(all_features, xaggs, xaggs_dev)
        else:
            for n, feature in enumerate(all_features):
                num_iter = f"{n + 1}/{len(all_features)}"  # logging iteration number
                self._carve_feature(self.features(feature), xaggs, xaggs_dev, num_iter)

        # discretizing features based on each feature's values_order
        super().fit(X, y)

        return self

    def _carve_features_parallel(
        self,
        all_features: list[str],
        xaggs: dict[str, pd.Series | pd.DataFrame | None],
        xaggs_dev: dict[str, pd.Series | pd.DataFrame | None],
    ) -> None:
        """Dispatches ``_carve_feature`` across a process pool, one task per feature.

        Per-feature workers receive only the feature instance + its xagg /
        xagg_dev slice (not the full dict). Verbose per-feature logging is
        silenced; a single banner is printed when verbose is on.
        """
        if self.config.verbose:
            print(f"--- [{self.__name__}] Carving {len(all_features)} features on {self.config.n_jobs} workers")

        payloads = [(self.features(version), xaggs[version], xaggs_dev[version]) for version in all_features]
        worker = partial(
            _carve_feature_worker,
            evaluator=self.combination_evaluator,
            max_n_mod=self.max_n_mod,
            min_freq=self.min_freq,
            dropna=self.config.dropna,
        )

        with Pool(processes=self.config.n_jobs) as pool:
            for updated_feature, viable in pool.imap_unordered(worker, payloads):
                if viable:
                    _replace_feature_in_features(self.features, updated_feature)
                else:
                    print(
                        f"WARNING: No robust combination for {updated_feature}. Consider "
                        "increasing the size of X_dev or dropping the feature (X not "
                        "representative of X_dev for this feature)."
                    )
                    self.features.remove(updated_feature.version)

    @abstractmethod
    def _aggregator(self, X: pd.DataFrame, y: pd.Series) -> dict[str, pd.Series | pd.DataFrame | None]:
        """Helper that aggregates X by y into per-feature crosstabs or means
        (carver specific). Returns ``{feature.version: xagg}`` for each feature.
        """

    def _carve_feature(
        self,
        feature: BaseFeature,
        xaggs: dict[str, pd.Series | pd.DataFrame | None],
        xaggs_dev: dict[str, pd.Series | pd.DataFrame | None],
        num_iter: str,
    ) -> None:
        """Carves a feature into buckets that maximize association with the target"""

        # verbose if requested
        if self.config.verbose:
            print(f"--- [{self.__name__}] Fit {feature} ({num_iter})")

        # getting xtabs on train/test
        xagg = xaggs[feature.version]
        xagg_dev = xaggs_dev[feature.version]

        # printing raw distribution
        self._print_xagg(feature, xagg=xagg, xagg_dev=xagg_dev, message="Raw distribution")

        # getting best combination
        best_combination = self.combination_evaluator.get_best_combination(
            feature, xagg, xagg_dev, max_n_mod=self.max_n_mod, min_freq=self.min_freq, dropna=self.config.dropna
        )

        # printing carved distribution, for found, suitable combination
        if best_combination is not None:
            dev_sample = self.combination_evaluator.samples.dev
            self._print_xagg(
                feature,
                xagg=self.combination_evaluator.samples.train.xagg,
                xagg_dev=dev_sample.xagg if dev_sample.has_xagg else None,
                message="Carved distribution",
            )

        # no suitable combination has been found -> removing feature
        else:
            print(
                f"WARNING: No robust combination for {feature}. Consider increasing the size of "
                "X_dev or dropping the feature (X not representative of X_dev for this feature)."
            )
            self.features.remove(feature.version)

    def _print_xagg(
        self,
        feature: BaseFeature,
        xagg: pd.Series | pd.DataFrame | None,
        message: str,
        *,
        xagg_dev: pd.Series | pd.DataFrame | None = None,
    ) -> None:
        """Prints crosstabs' target rates and frequencies per modality, in raw or html format"""
        if self.config.verbose:
            print(f" [{self.__name__}] {message}")

            formatted_xagg, formatted_xagg_dev = self._format_xagg(feature, xagg, xagg_dev)

            nice_xagg, nice_xagg_dev = self._pretty_print(formatted_xagg, formatted_xagg_dev)

            if not self.pretty_print:  # no pretty hmtl printing
                self._print_raw(nice_xagg, nice_xagg_dev, xagg_dev)
            else:  # pretty html printing
                self._print_html(nice_xagg, nice_xagg_dev)

    def _format_xagg(
        self,
        feature: BaseFeature,
        xagg: pd.Series | pd.DataFrame | None,
        xagg_dev: pd.Series | pd.DataFrame | None = None,
    ) -> tuple[pd.Series | pd.DataFrame | None, pd.Series | pd.DataFrame | None]:
        """Formats the XAGG DataFrame."""
        formatted_xagg = index_mapper(feature, xagg)
        formatted_xagg_dev = index_mapper(feature, xagg_dev)
        return formatted_xagg, formatted_xagg_dev

    def _pretty_print(
        self,
        formatted_xagg: pd.Series | pd.DataFrame | None,
        formatted_xagg_dev: pd.Series | pd.DataFrame | None,
    ) -> tuple[pd.Series | pd.DataFrame | None, pd.Series | pd.DataFrame | None]:
        """Returns pretty-printed XAGG DataFrames."""
        nice_xagg = (
            self.combination_evaluator.target_rate.compute(formatted_xagg) if formatted_xagg is not None else None
        )
        nice_xagg_dev = (
            self.combination_evaluator.target_rate.compute(formatted_xagg_dev)
            if formatted_xagg_dev is not None
            else None
        )
        return nice_xagg, nice_xagg_dev

    def _print_raw(
        self,
        nice_xagg: pd.Series | pd.DataFrame | None,
        nice_xagg_dev: pd.Series | pd.DataFrame | None,
        xagg_dev: pd.Series | pd.DataFrame | None = None,
    ) -> None:
        """Prints raw XAGG DataFrames."""
        print(nice_xagg, "\n")
        if xagg_dev is not None:
            print("X_dev distribution\n", nice_xagg_dev, "\n")

    def _print_html(
        self,
        nice_xagg: pd.Series | pd.DataFrame | None,
        nice_xagg_dev: pd.Series | pd.DataFrame | None,
    ) -> None:
        """Prints XAGG DataFrames in HTML format."""
        # getting prettier xtabs
        nicer_xagg = prettier_xagg(nice_xagg, caption="X distribution")
        nicer_xagg_dev = prettier_xagg(nice_xagg_dev, caption="X_dev distribution", hide_index=True)

        # merging outputs
        nicer_xaggs = nicer_xagg + "          " + nicer_xagg_dev

        # displaying html of colored DataFrame
        display_html(nicer_xaggs, raw=True)

    @classmethod
    def load(cls, file_name: str) -> "BaseCarver":
        """Allows one to load a Carver saved as a .json file."""
        with open(file_name, encoding="utf-8") as json_file:
            data = json.load(json_file)

        # deserializing features
        features = Features.load(data.pop("features"))

        # deserializing Combinations: identify the evaluator class from sort_by
        combinations_json = data.pop("combination_evaluator")
        sort_by = combinations_json.pop("sort_by", None)
        if sort_by == "tschuprowt":
            evaluator_cls: type[CombinationEvaluator] = TschuprowtCombinations
        elif sort_by == "cramerv":
            evaluator_cls = CramervCombinations
        elif sort_by == "kruskal":
            evaluator_cls = KruskalCombinations
        else:
            raise ValueError(f"[{cls.__name__}] Unknown combinations sort_by={sort_by!r}")

        is_fitted = data.pop("is_fitted", False)
        min_freq = data.pop("min_freq", None)
        max_n_mod = data.pop("max_n_mod")
        config_data = data.pop("config", {})
        config = DiscretizerConfig(
            dropna=config_data.get("dropna", True),
            ordinal_encoding=config_data.get("ordinal_encoding", True),
            verbose=config_data.get("verbose", False),
            n_jobs=config_data.get("n_jobs", 1),
            copy=config_data.get("copy", True),
        )

        instance = cls(
            features=features,
            min_freq=min_freq,
            max_n_mod=max_n_mod,
            combination_evaluator=evaluator_cls(),
            config=config,
        )
        instance.is_fitted = is_fitted
        return instance


def discretize(
    features: Features,
    samples: Samples,
    discretizer_min_freq: float,
    config: DiscretizerConfig,
) -> Samples:
    """Discretizes X and X_dev according to the frequency of each feature's modalities."""

    # discretizing all features, always copying, to keep discretization from start to finish
    discretizer = Discretizer(
        features=features,
        min_freq=discretizer_min_freq,
        config=replace(config, dropna=False, copy=True, ordinal_encoding=False),
    )

    # fitting discretizer on X
    samples.train.X = discretizer.fit_transform(**samples.train)

    # applying discretizer on X_dev if provided
    if samples.dev.has_X:
        samples.dev.X = discretizer.transform(**samples.dev)

    return samples
