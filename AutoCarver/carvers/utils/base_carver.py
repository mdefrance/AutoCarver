"""Tool to build optimized buckets out of Quantitative and Qualitative features
for any task.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
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

    train: Sample = field(default_factory=lambda: Sample(X=None))
    dev: Sample = field(default_factory=lambda: Sample(X=None))

    def fillna(self, features: Features) -> None:
        """fills up nans in X and X_dev"""
        self.train.X = features.fillna(self.train.X)
        if self.dev.X is not None:
            self.dev.X = features.fillna(self.dev.X)


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
        combinations: CombinationEvaluator,
        *,
        dropna: bool = True,
        ordinal_encoding: bool = True,
        discretizer_min_freq: float | None = None,
        config: DiscretizerConfig | None = None,
    ) -> None:
        """
        Parameters
        ----------

        combinations : CombinationEvaluator
            Metric to perform association measure between :class:`Features` and target.

        dropna : bool, optional
            * ``True``, try to group ``nan`` with other modalities.
            * ``False``, ``nan`` are ignored (not grouped), by default ``True``

        ordinal_encoding : bool, optional
            Whether or not to ordinal encode :class:`Features`, by default ``True``
            (carver-specific override of :attr:`DiscretizerConfig.ordinal_encoding`).

        max_n_mod : int, optional
            Maximum number of modalities per feature, by default ``5``

            * The combination with the best association will be selected.
            * All combinations of sizes from 1 to :attr:`max_n_mod` are tested out.

            .. tip::
                Set between ``3`` (faster, more robust) and ``7`` (slower, less robust)

        discretizer_min_freq : float, optional
            Specific :attr:`min_freq` used by the underlying :class:`Discretizer`, by default
            ``min_freq / 2``.
        """
        # carver-level toggles flow through dedicated kwargs (not config) because their
        # historical defaults differ from BaseDiscretizer's
        if config is None:
            config = DiscretizerConfig()
        config = replace(config, dropna=dropna, ordinal_encoding=ordinal_encoding)

        super().__init__(features, min_freq=min_freq, config=config)

        self.discretizer_min_freq = discretizer_min_freq if discretizer_min_freq is not None else min_freq / 2

        # attach combinations and sync the toggles it cares about
        self.combinations: CombinationEvaluator = combinations
        self.combinations.min_freq = self.min_freq
        self.combinations.verbose = self.config.verbose
        self.combinations.dropna = self.config.dropna

    @property
    def pretty_print(self) -> bool:
        """Returns the pretty_print attribute"""
        return self.config.verbose and _has_idisplay

    @property
    def max_n_mod(self) -> int:
        """Returns the max_n_mod attribute"""
        return self.combinations.max_n_mod

    @max_n_mod.setter
    def max_n_mod(self, value: int) -> None:
        """Sets the max_n_mod attribute"""
        self.combinations.max_n_mod = value

    def _prepare_data(self, samples: Samples) -> Samples:
        """Validates format and content of X and y."""
        if samples.train.y is None:
            raise ValueError(f"[{self.__name__}] y must be provided, got {samples.train.y}")

        # Checking for binary target and copying X
        samples.train = super()._prepare_data(samples.train)
        samples.dev = super()._prepare_data(samples.dev)

        # discretizing features according to min_freq
        samples = discretize(self.features, samples, self.discretizer_min_freq, self.config)

        # setting dropna to True for filling up nans
        self.features.dropna = True

        # filling up nans
        samples.fillna(self.features)

        return samples

    def fit(  # pylint: disable=W0222
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

        # carving each feature
        for n, feature in enumerate(all_features):
            num_iter = f"{n + 1}/{len(all_features)}"  # logging iteration number
            self._carve_feature(self.features(feature), xaggs, xaggs_dev, num_iter)

        # discretizing features based on each feature's values_order
        super().fit(X, y)

        return self

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
        best_combination = self.combinations.get_best_combination(feature, xagg, xagg_dev=xagg_dev)

        # printing carved distribution, for found, suitable combination
        if best_combination is not None:
            self._print_xagg(
                feature,
                xagg=self.combinations.samples.train.xagg,
                xagg_dev=self.combinations.samples.dev.xagg,
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
        nice_xagg = self.combinations.target_rate.compute(formatted_xagg) if formatted_xagg is not None else None
        nice_xagg_dev = (
            self.combinations.target_rate.compute(formatted_xagg_dev) if formatted_xagg_dev is not None else None
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

    def to_json(self, light_mode: bool = False) -> dict:
        """Converts to JSON format. Adds carver-specific fields on top of the base content."""
        content = super().to_json(light_mode)
        content["discretizer_min_freq"] = self.discretizer_min_freq
        return content

    @classmethod
    def load(cls, file_name: str) -> "BaseCarver":
        """Allows one to load a Carver saved as a .json file."""
        with open(file_name, encoding="utf-8") as json_file:
            data = json.load(json_file)

        # deserializing features
        features = Features.load(data.pop("features"))

        # deserializing Combinations
        combinations_json = data.pop("combinations")
        if combinations_json["sort_by"] == "tschuprowt":
            combinations = TschuprowtCombinations.load(combinations_json)
        elif combinations_json["sort_by"] == "cramerv":
            combinations = CramervCombinations.load(combinations_json)
        elif combinations_json["sort_by"] == "kruskal":
            combinations = KruskalCombinations.load(combinations_json)
        else:
            combinations = CombinationEvaluator.load(combinations_json)

        is_fitted = data.pop("is_fitted", False)
        min_freq = data.pop("min_freq", None)
        discretizer_min_freq = data.pop("discretizer_min_freq", None)
        dropna = data.pop("dropna", True)
        ordinal_encoding = data.pop("ordinal_encoding", True)
        config = DiscretizerConfig(
            verbose=data.pop("verbose", False),
            n_jobs=data.pop("n_jobs", 1),
        )

        instance = cls(
            features=features,
            min_freq=min_freq,
            combinations=combinations,
            dropna=dropna,
            ordinal_encoding=ordinal_encoding,
            discretizer_min_freq=discretizer_min_freq,
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
    if samples.dev.X is not None:
        samples.dev.X = discretizer.transform(**samples.dev)

    return samples
