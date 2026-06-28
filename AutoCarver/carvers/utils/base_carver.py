"""Tool to build optimized buckets out of Quantitative and Qualitative features
for any task.
"""

import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from functools import partial
from multiprocessing import Pool
from typing import Self
from warnings import warn

import pandas as pd

from AutoCarver.carvers.utils.pretty_print import index_mapper, prettier_xagg
from AutoCarver.combinations import (
    CombinationEvaluator,
    CramervCombinations,
    KendallTauBCombinations,
    KendallTauCCombinations,
    KruskalCombinations,
    SomersDCombinations,
    TschuprowtCombinations,
)
from AutoCarver.discretizers import BaseDiscretizer, Discretizer, Sample
from AutoCarver.discretizers.utils.base_discretizer import ProcessingConfig
from AutoCarver.features import BaseFeature, Features, get_versions
from AutoCarver.features.qualitatives import CategoricalFeature, NestedFeature, OrdinalFeature
from AutoCarver.features.quantitatives import DatetimeFeature, NumericalFeature
from AutoCarver.utils import extend_docstring, has_idisplay

# trying to import extra dependencies
_has_idisplay = has_idisplay()
if _has_idisplay:
    from IPython.display import display_html

# maps a serialized combination_evaluator.sort_by back to its evaluator class (used by load)
_EVALUATORS_BY_SORT_BY: dict[str, type[CombinationEvaluator]] = {
    evaluator_cls.sort_by: evaluator_cls
    for evaluator_cls in (
        TschuprowtCombinations,
        CramervCombinations,
        KruskalCombinations,
        KendallTauCCombinations,
        KendallTauBCombinations,
        SomersDCombinations,
    )
}


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
    min_freq_alpha: float,
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
        feature,
        xagg,
        xagg_dev,
        max_n_mod=max_n_mod,
        min_freq=min_freq,
        dropna=dropna,
        min_freq_alpha=min_freq_alpha,
    )
    return feature, best is not None


def _drop_reason_from_history(history: pd.DataFrame) -> str:
    """Synthesizes a human-readable drop reason from a dropped feature's history.

    Picks the most frequent failing-test message across ``train``/``dev`` blocks
    of historized non-viable combinations.
    """
    if history.empty:
        return "No combination possible"

    info_counts: dict[str, int] = {}
    for _, row in history.iterrows():
        if bool(row.get("viable", False)):
            continue
        for block_key in ("train", "dev"):
            block = row.get(block_key)
            if isinstance(block, dict):
                msg = block.get("info") or ""
                if msg:
                    info_counts[msg] = info_counts.get(msg, 0) + 1

    if not info_counts:
        return "No robust combination"
    msg, _ = max(info_counts.items(), key=lambda kv: kv[1])
    return f"No robust combination ({msg})"


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
    is_y_ordinal = False

    # carvers group nans and ordinal-encode by default; these override the
    # BaseDiscretizer context defaults used to resolve the ``None`` toggles of
    # ProcessingConfig (see ProcessingConfig docstring).
    _default_dropna = True
    _default_ordinal_encoding = True

    @extend_docstring(BaseDiscretizer.__init__, exclude=["min_freq", "config"])
    def __init__(
        self,
        features: Features,
        min_freq: float,
        max_n_mod: int,
        *,
        combination_evaluator: CombinationEvaluator | None = None,
        config: ProcessingConfig | None = None,
    ) -> None:
        """
        Parameters
        ----------
        min_freq : float
            Minimum frequency per modality. Tested via a Wilson upper bound at
            significance :attr:`ProcessingConfig.min_freq_alpha` (see
            :ref:`MinFreqViability`).

            * Features need at least one modality with frequency significantly
              above :attr:`min_freq`.
            * Drives the :ref:`viability filter <MinFreqViability>` on both
              **train** and **dev** crosstabs during the combination search.
            * The pre-search discretization runs at the **halved** threshold
              :attr:`half_min_freq` (= ``min_freq / 2``) so the combination
              evaluator has a finer granularity to recombine.

            .. tip::
                Set between ``0.01`` (slower, less robust) and ``0.05`` (faster,
                more robust).

        max_n_mod : int
            Maximum number of modalities per carved feature. Forwarded to the
            configured :class:`CombinationEvaluator`.

            * The combination with the best association will be selected.
            * All combinations of sizes from ``1`` to :attr:`max_n_mod` are
              tested out.

            .. tip::
                Set between ``5`` (faster, more robust) and ``7`` (slower, less
                robust).

        combination_evaluator : CombinationEvaluator, optional
            Pre-built :class:`CombinationEvaluator` instance used to measure
            association. Subclasses default this to a task-appropriate instance
            (e.g. :class:`TschuprowtCombinations` for binary, :class:`KruskalCombinations`
            for continuous). The carver forwards ``verbose`` onto the instance
            and passes ``max_n_mod`` / ``min_freq`` / ``dropna`` /
            ``min_freq_alpha`` directly to each
            :meth:`~CombinationEvaluator.get_best_combination` call.

        config : ProcessingConfig, optional
            Behavioral toggles inherited from :class:`BaseDiscretizer`. Its
            ``dropna`` and ``ordinal_encoding`` toggles default to ``None`` and
            are resolved to the carver-friendly ``True`` here (group ``nan``,
            ordinal-encode features for downstream sklearn estimators). Passing
            a partial config (e.g. ``ProcessingConfig(verbose=True)``) therefore
            keeps those carver defaults; set them explicitly to override.
        """
        if combination_evaluator is None:
            raise ValueError(
                f"[{self.__name__}] combination_evaluator must be provided (subclasses set a task-appropriate default)."
            )

        # carver-friendly defaults (dropna / ordinal_encoding True) are applied
        # by BaseDiscretizer.__init__ when those toggles are left ``None``, so a
        # partial config only changes the fields it sets explicitly.
        super().__init__(features, min_freq=min_freq, config=config)

        self.max_n_mod = max_n_mod
        combination_evaluator.verbose = self.config.verbose
        self.combination_evaluator: CombinationEvaluator = combination_evaluator

        # features dropped by the carver because no robust combination was found.
        # Kept (not cleared on re-fit) so users can inspect why each dropped via
        # the marker columns added to ``summary`` / ``history``.
        self.dropped_features: list[BaseFeature] = []

    @property
    def half_min_freq(self) -> float:
        """Half of :attr:`min_freq` — the tolerant frequency floor the carver
        applies when discretizing prior to combination search. Halving here gives
        the combination evaluator a finer granularity to recombine, while the
        underlying discretizers themselves compare directly against ``min_freq``
        (with a 1-row tolerance). Owning the halving in the carver — rather than
        inside individual discretizers — keeps the per-discretizer semantic uniform.
        """
        return self.min_freq / 2

    @property
    def pretty_print(self) -> bool:
        """Returns the pretty_print attribute"""
        return self.config.verbose and _has_idisplay

    def to_json(self, light_mode: bool = False) -> dict:
        content = super().to_json(light_mode)
        content["max_n_mod"] = self.max_n_mod
        content["combination_evaluator"] = self.combination_evaluator.to_json()
        content["dropped_features"] = [f.to_json(light_mode) for f in self.dropped_features]
        return content

    @property
    def summary(self) -> pd.DataFrame:
        """Per-feature carving summary, extended with one block per dropped feature.

        Rows from features that the carver dropped (no robust combination on
        train and/or dev) are appended at the end with two marker columns:

        - ``dropped`` (bool): ``True`` for dropped features, ``False`` otherwise.
        - ``dropped_reason`` (str | None): synthesized from the feature's history
          — the dominant failing test message across attempted combinations.
        """
        rows: list[dict] = []
        for feature in self.features:
            for row in feature.summary:
                rows.append({**row, "dropped": False, "dropped_reason": None})
        for feature in self.dropped_features:
            reason = _drop_reason_from_history(feature.history)
            for row in feature.summary:
                rows.append({**row, "dropped": True, "dropped_reason": reason})

        summaries = pd.DataFrame(rows)
        if summaries.empty:
            return summaries

        # per-modality stats (count, target_mean, frequency) stay columns; only per-feature
        # metrics (sort_by association, n_mod) become index levels so they collapse to one
        # row per feature instead of repeating across every modality.
        excluded = {"feature", "label", "content", "target_mean", "frequency", "count", "dropped", "dropped_reason"}
        indices = [col for col in summaries.columns if col not in excluded]
        indices = ["feature"] + indices + ["label"]
        return summaries.set_index(indices)

    @property
    def history(self) -> pd.DataFrame:
        """Combined combination-history of carved + dropped features.

        Dropped features' rows are appended with ``dropped=True``; carved
        features' rows get ``dropped=False``.
        """
        frames: list[pd.DataFrame] = []
        current = self.features.history
        if not current.empty:
            frames.append(current.assign(dropped=False))
        for feature in self.dropped_features:
            df = feature.history
            if len(df) > 0:
                frames.append(df.assign(feature=str(feature), dropped=True))
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _prepare_samples(self, samples: Samples) -> Samples:
        """Validates format and content of X and y."""
        if samples.train.y is None:
            raise ValueError(f"[{self.__name__}] y must be provided, got {samples.train.y}")

        # Checking for binary target and copying X
        samples.train = super()._prepare_sample(samples.train)
        samples.dev = super()._prepare_sample(samples.dev)

        # discretizing features at half min_freq so the carver has a finer
        # granularity to combine when forming optimal groups
        samples = discretize(self.features, samples, self.half_min_freq, self.config)

        # setting dropna to True for filling up nans
        self.features.dropna = True

        # filling up nans
        samples.fillna(self.features)

        return samples

    def _drop_target_from_features(self, X: pd.DataFrame, y: pd.Series | None) -> None:
        """Drops the target column from ``self.features`` if it leaked in.

        ``Features.from_dataframe`` maps every column of the input, target included; the
        target reaches the carver as the named ``y`` Series, so it is removed here rather
        than at feature-construction time.

        The guard fires only when ``y`` is genuinely a column of ``X`` (same name *and*
        values): pandas propagates column names through arithmetic, so a derived target
        like ``X[col] * 0.5 + noise`` can share a feature's name without being it.
        """
        if y is None or y.name is None:
            return
        name = str(y.name)
        if name in self.features and name in X.columns and y.equals(X[name]):
            warn(f"[{self.__name__}] dropping target column {name!r} from features", UserWarning, stacklevel=2)
            self.features.remove(name)

    def fit(  # type: ignore
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        X_dev: pd.DataFrame | None = None,
        y_dev: pd.Series | None = None,
    ) -> Self:
        """Finds the combination of modalities of X that provides the best association with y.
        If provided, X_dev set should be large enough to have the same distribution as X.

        Features for which no candidate combination survives the viability filter
        (Wilson ``min_freq`` on train + dev, distinct target rates, train/dev rank
        preservation) are dropped from ``self.features`` and retained on
        ``self.dropped_features``. With ``ProcessingConfig(n_jobs=k)`` and
        ``k > 1`` and more than one feature, the per-feature combination search
        runs in parallel through ``multiprocessing.Pool.imap_unordered``.

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

        # dropping the target column if it leaked into the features
        self._drop_target_from_features(X, y)

        # setting is_fitted
        self.features.is_fitted = True

        # initiating samples
        samples = Samples(Sample(X, y), Sample(X_dev, y_dev))

        # preparing datasets and checking for wrong values
        samples = self._prepare_samples(samples)

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

    def transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Applies the fitted carving to ``X``, feature-chunked across processes when ``n_jobs > 1``.

        The final output transform is otherwise a single serial pass (the qualitative ``map`` holds
        the GIL, so threads don't help). Each column is pickled once per chunk, so coarse
        feature-chunk process parallelism trims it. Multiclass/nested/datetime feature sets (which
        need cross-column context) fall back to the serial :meth:`BaseDiscretizer.transform`.
        """
        feature_list = list(self.features)
        if not self.is_fitted:
            raise RuntimeError(f"[{self.__name__}] Call fit method first.")
        if self.config.n_jobs <= 1 or len(feature_list) <= 1 or not _discretizable_in_chunks(self.features):
            return super().transform(X, y)

        n_chunks = min(self.config.n_jobs, len(feature_list))
        chunks = [feature_list[i::n_chunks] for i in range(n_chunks)]
        chunk_config = replace(self.config, n_jobs=1, copy=True)
        payloads = [(chunk, X[get_versions(chunk)], chunk_config) for chunk in chunks]

        with Pool(processes=n_chunks) as pool:
            results = pool.map(_transform_chunk_worker, payloads)

        # preserve all original columns + index; overwrite only the carved feature columns, matching
        # the serial transform (which leaves non-feature columns untouched)
        transformed_X = X.copy()
        for transformed in results:
            transformed_X[transformed.columns] = transformed
        return transformed_X

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
            dropna=bool(self.config.dropna),
            min_freq_alpha=self.config.min_freq_alpha,
        )

        with Pool(processes=self.config.n_jobs) as pool:
            for updated_feature, viable in pool.imap_unordered(worker, payloads):
                if viable:
                    self.features.replace_feature(updated_feature)
                else:
                    print(
                        f"WARNING: No robust combination for {updated_feature}. Consider "
                        "increasing the size of X_dev or dropping the feature (X not "
                        "representative of X_dev for this feature)."
                    )
                    self.dropped_features.append(updated_feature)
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
            feature,
            xagg,
            xagg_dev,
            max_n_mod=self.max_n_mod,
            min_freq=self.min_freq,
            dropna=bool(self.config.dropna),
            min_freq_alpha=self.config.min_freq_alpha,
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
            self.dropped_features.append(feature)
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
        evaluator_cls = _EVALUATORS_BY_SORT_BY.get(sort_by)
        if evaluator_cls is None:
            raise ValueError(f"[{cls.__name__}] Unknown combinations sort_by={sort_by!r}")

        is_fitted = data.pop("is_fitted", False)
        min_freq = data.pop("min_freq", None)
        max_n_mod = data.pop("max_n_mod")
        config_data = data.pop("config", {})
        config = ProcessingConfig(
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

        # deserializing dropped_features (mirrors Features.load type-dispatch)
        for fjson in data.pop("dropped_features", []):
            if fjson.get("is_nested"):
                instance.dropped_features.append(NestedFeature.load(fjson))
            elif fjson.get("is_categorical"):
                instance.dropped_features.append(CategoricalFeature.load(fjson))
            elif fjson.get("is_ordinal"):
                instance.dropped_features.append(OrdinalFeature.load(fjson))
            elif fjson.get("is_datetime"):
                instance.dropped_features.append(DatetimeFeature.load(fjson))
            elif fjson.get("is_quantitative"):
                instance.dropped_features.append(NumericalFeature.load(fjson))

        return instance


def _discretizable_in_chunks(features: Features) -> bool:
    """Whether ``features`` can be partitioned across processes for discretization.

    Discretization is per-feature independent *except* for features with cross-column
    dependencies: nested features need their parent columns, datetime features may reference
    another column, and multiclass version-aliasing (``version != name``) duplicates columns.
    Any of those → fall back to the serial path so a chunk never misses a column it needs.
    """
    if len(features.nested) > 0 or len(features.datetimes) > 0:
        return False
    return all(feature.version == feature.name for feature in features)


def _discretize_chunk_worker(
    payload: tuple[list[BaseFeature], pd.DataFrame, "pd.Series | None", pd.DataFrame | None, float, ProcessingConfig],
) -> tuple[list[BaseFeature], pd.DataFrame, pd.DataFrame | None]:
    """Picklable worker: discretizes one feature-chunk end to end.

    Each chunk owns a disjoint set of columns, so every column is pickled exactly once (to one
    worker). Returns the fitted feature objects plus the transformed train/dev columns for the
    parent to merge — no per-feature round-trips.
    """
    chunk_features, x_train, y_train, x_dev, min_freq, config = payload
    sub_features = Features.from_list(chunk_features)
    discretizer = Discretizer(features=sub_features, min_freq=min_freq, config=config)
    x_train_t = discretizer.fit_transform(x_train, y_train)
    x_dev_t = discretizer.transform(x_dev) if x_dev is not None else None
    return list(sub_features), x_train_t, x_dev_t


def _transform_chunk_worker(
    payload: tuple[list[BaseFeature], pd.DataFrame, ProcessingConfig],
) -> pd.DataFrame:
    """Picklable worker: applies the fitted bucketization to one feature-chunk's columns.

    The features are already fitted/carved (``label_per_value`` is baked in), so the worker just
    rebuilds a serial :class:`BaseDiscretizer` over its chunk and transforms its own columns — one
    pickle per column, results merged by the parent.
    """
    chunk_features, x_chunk, config = payload
    discretizer = BaseDiscretizer(features=Features.from_list(chunk_features), config=config)
    discretizer.is_fitted = True
    return discretizer.transform(x_chunk)


def parallel_aggregate(
    agg_fn: Callable,
    features: Features,
    X: pd.DataFrame | None,
    y: "pd.Series | None",
    n_jobs: int,
) -> dict[str, pd.Series | pd.DataFrame | None]:
    """Computes ``{feature.version: xagg}`` per feature, threaded across features.

    Uses **threads**, not processes: each per-feature aggregation (e.g. ``get_crosstab``) is light
    vectorized pandas whose C internals release the GIL, so threads overlap them with zero pickling
    / spawn overhead. A process pool here loses — its per-call dispatch cost exceeds the crosstab
    compute (measured). ``agg_fn`` is an ``(X, y, feature) -> xagg`` callable; ``X`` is read-only and
    shared, so this is thread-safe. Returns all-``None`` when ``X`` is absent (empty dev sample).
    """
    feature_list = list(features)
    if X is None:
        return {feature.version: None for feature in feature_list}

    # serial path (also when too few features to amortize the threads)
    if n_jobs <= 1 or len(feature_list) <= 1:
        return {feature.version: agg_fn(X, y, feature) for feature in feature_list}

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        results = executor.map(lambda feature: (feature.version, agg_fn(X, y, feature)), feature_list)
        return dict(results)


def discretize(
    features: Features,
    samples: Samples,
    discretizer_min_freq: float,
    config: ProcessingConfig,
) -> Samples:
    """Discretizes X and X_dev according to the frequency of each feature's modalities.

    With ``n_jobs > 1`` and an independently-chunkable feature set, the features are partitioned
    across worker processes — each discretizes its own columns end to end (one pickle per column)
    and the results are merged back. This is the coarse-grained parallelism the pre-pass needs:
    per-feature work is cheap vectorized pandas, so per-task dispatch would lose to pickling.
    """

    # always copying, to keep discretization from start to finish; workers run fully serial
    # (n_jobs=1) so they never spawn nested pools.
    chunk_config = replace(config, dropna=False, copy=True, ordinal_encoding=False, n_jobs=1)

    feature_list = list(features)
    if config.n_jobs <= 1 or len(feature_list) <= 1 or not _discretizable_in_chunks(features):
        # serial path (and fallback for feature sets that aren't independently chunkable)
        discretizer = Discretizer(features=features, min_freq=discretizer_min_freq, config=chunk_config)
        samples.train.X = discretizer.fit_transform(**samples.train)
        if samples.dev.has_X:
            samples.dev.X = discretizer.transform(**samples.dev)
        return samples

    # parallel path: more chunks than workers + dynamic scheduling. Per-feature cost varies a lot
    # (high-cardinality categoricals cost far more than numerics), so finer chunks let an idle worker
    # pick up the next task instead of stalling while one worker grinds a heavy chunk — measured ~9%
    # faster than one-chunk-per-worker. Total column pickling is unchanged (each column ships once).
    n_workers = min(config.n_jobs, len(feature_list))
    n_chunks = min(config.n_jobs * 4, len(feature_list))
    chunks = [feature_list[i::n_chunks] for i in range(n_chunks)]
    has_dev = samples.dev.has_X

    payloads = [
        (
            chunk,
            samples.train.X[get_versions(chunk)],
            samples.train.y,
            samples.dev.X[get_versions(chunk)] if has_dev else None,
            discretizer_min_freq,
            chunk_config,
        )
        for chunk in chunks
    ]

    with Pool(processes=n_workers) as pool:
        results = list(pool.imap_unordered(_discretize_chunk_worker, payloads))

    # copy before merging so we never mutate the caller's frame in place — the serial path returns
    # a fresh (discretizer-copied) frame regardless of ``config.copy``, so we match that here.
    samples.train.X = samples.train.X.copy()
    if has_dev:
        samples.dev.X = samples.dev.X.copy()

    # merge fitted features + transformed columns back into the parent frame
    for fitted_features, x_train_t, x_dev_t in results:
        for feature in fitted_features:
            features.replace_feature(feature)
        samples.train.X[x_train_t.columns] = x_train_t
        if has_dev and x_dev_t is not None:
            samples.dev.X[x_dev_t.columns] = x_dev_t

    return samples
