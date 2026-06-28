"""Base tools to build simple buckets out of Quantitative and Qualitative features
for a binary classification model.
"""

import json
from abc import ABC
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from AutoCarver.features import BaseFeature, Features, NestedFeature
from AutoCarver.utils import extend_docstring


@dataclass
class ProcessingConfig:
    """Behavioral configuration applied to a :class:`BaseDiscretizer`.

    Carries cross-cutting toggles that propagate unchanged to sub-discretizers.
    Pure domain values (``min_freq``, ``combinations`` …) remain explicit
    constructor arguments; ``min_freq_alpha`` lives here because it tunes
    *how* ``min_freq`` is tested, not the target itself.

    ``copy=True`` is the default so that BaseDiscretizer doesn't mutate caller
    DataFrames in place — set to ``False`` when nested inside a pipeline that
    already owns the dataframe.

    ``min_freq_alpha`` is the two-sided significance level of the Wilson
    interval used to decide whether a modality's observed frequency is
    significantly below ``min_freq``. Smaller values are more lenient
    (wider CI → fewer merges); ``0.05`` matches a 95% interval.

    ``n_jobs`` controls per-feature parallelism inside :class:`BaseCarver`:
    with ``n_jobs > 1`` and more than one feature, the per-feature combination
    search runs through ``multiprocessing.Pool.imap_unordered``. Worth it only
    on hundreds-to-thousands of features (pool startup + pickle overhead
    dominate below that).

    ``ordinal_encoding`` and ``dropna`` default to ``None`` meaning *use the
    context default*: discretizers leave them ``False``, carvers turn them
    ``True`` (group ``nan``, ordinal-encode for downstream sklearn estimators).
    They are resolved to a concrete ``bool`` in :meth:`BaseDiscretizer.__init__`.
    Leaving them ``None`` is what lets a partial config (e.g.
    ``ProcessingConfig(verbose=True)``) toggle one field without silently
    flipping the carver-friendly defaults — set them explicitly to override.
    """

    copy: bool = True
    ordinal_encoding: bool | None = None
    dropna: bool | None = None
    verbose: bool = False
    n_jobs: int = 1
    min_freq_alpha: float = 0.05


# Backward-compatible alias: this config was historically named ``DiscretizerConfig`` but is
# shared by discretizers, carvers and selectors — the neutral ``ProcessingConfig`` is preferred.
DiscretizerConfig = ProcessingConfig


class Sample:
    """Sample class to store X and y.

    The public ``X`` is typed as mandatory :class:`pd.DataFrame`. The constructor
    accepts ``None`` so that placeholders (default factories, optional dev samples)
    are expressible, but reading ``.X`` on an unset Sample raises. Use
    :attr:`has_X` to check presence without triggering.
    """

    def __init__(self, X: pd.DataFrame | None = None, y: pd.Series | None = None) -> None:
        self._X: pd.DataFrame | None = X
        self.y: pd.Series | None = y

    @property
    def X(self) -> pd.DataFrame:
        """Returns the stored DataFrame, or raises if not set."""
        if self._X is None:
            raise RuntimeError("[Sample] X is not set")
        return self._X

    @X.setter
    def X(self, value: pd.DataFrame) -> None:
        self._X = value

    @property
    def has_X(self) -> bool:
        """Whether X is set."""
        return self._X is not None

    def __getitem__(self, key):
        """Returns the DataFrame or the Series"""
        if key == "X":
            return self._X
        if key == "y":
            return self.y

        raise KeyError(key)

    def __iter__(self):
        """Returns an iterator over the DataFrame"""
        return iter(["X", "y"])

    def keys(self):
        """Returns the keys of the DataFrame"""
        return ["X", "y"]

    @property
    def shape(self):
        """Returns the shape of the DataFrame"""
        return self.X.shape

    @property
    def index(self):
        """Returns the index of the DataFrame"""
        return self.X.index

    @property
    def columns(self):
        """Returns the columns of the DataFrame"""
        return self.X.columns

    def __len__(self):
        return len(self.X)

    def fillna(self, features: Features) -> None:
        """fills up nans for features that have some"""
        self.X = features.fillna(self.X)

    def unfillna(self, features: Features) -> pd.DataFrame:
        """reinstating nans when not supposed to group them"""
        return features.unfillna(self.X)


class BaseDiscretizer(ABC, BaseEstimator, TransformerMixin):
    """Applies discretization using a dict of GroupedList to transform a DataFrame's columns.

    Examples
    --------
    See `Discretizers examples <https://autocarver.readthedocs.io/en/latest/index.html>`_
    """

    __name__ = "BaseDiscretizer"

    # context defaults for the ``None`` (unset) toggles of ProcessingConfig;
    # BaseCarver overrides both to True.
    _default_dropna: bool = False
    _default_ordinal_encoding: bool = False

    def __init__(
        self,
        features: "Features | Iterable[BaseFeature]",
        *,
        min_freq: float | None = None,
        config: ProcessingConfig | None = None,
    ) -> None:
        """
        Parameters
        ----------

        features : Features
            A set of :class:`Features` to be processed.

        min_freq : float
            Minimum frequency per modality. Tested via a Wilson upper bound at
            significance :attr:`ProcessingConfig.min_freq_alpha` (see
            :ref:`MinFreqViability`).

            * Features need at least one modality with frequency significantly
              above :attr:`min_freq`.
            * For continuous features, drives the number of quantiles (roughly
              ``1 / min_freq``).
            * Modalities significantly below :attr:`min_freq` are merged with
              the closest one (ordinal) or with a default group (categorical).

            .. tip::
                Set between ``0.01`` (slower, less robust) and ``0.05`` (faster,
                more robust).

        config : ProcessingConfig, optional
            Behavioral toggles (``copy`` / ``ordinal_encoding`` / ``dropna`` /
            ``verbose`` / ``n_jobs`` / ``min_freq_alpha``). Defaults to a
            default-initialized :class:`ProcessingConfig` — see
            :ref:`ProcessingConfig` for each field.
        """
        # accept either a Features collection or an iterable of BaseFeature
        if isinstance(features, Features):
            self.features: Features = features
        else:
            self.features = Features.from_list(features)

        config = config if config is not None else ProcessingConfig()
        # resolve context-dependent toggles: ``None`` means "use the context
        # default" (False here, overridden to True by BaseCarver) so a partial
        # config doesn't silently flip the unset fields.
        self.config: ProcessingConfig = replace(
            config,
            dropna=self._default_dropna if config.dropna is None else config.dropna,
            ordinal_encoding=(
                self._default_ordinal_encoding if config.ordinal_encoding is None else config.ordinal_encoding
            ),
        )
        self._min_freq = min_freq

        # set by subclasses; serialized for round-trip but not used by BaseDiscretizer itself
        # lifecycle flag — set by fit(), or by load() after restoring state
        self.is_fitted: bool = False

    @property
    def min_freq(self) -> float:
        """Public ``min_freq`` typed as mandatory ``float``.

        ``__init__`` accepts ``None`` so plain :class:`BaseDiscretizer` can be
        constructed without it (e.g. for the base-transform path that only
        re-applies an already-fitted bucketization). Reading raises when unset.
        Use :attr:`has_min_freq` to check presence without triggering.
        """
        if self._min_freq is None:
            raise RuntimeError(f"[{self.__name__}] min_freq is not set")
        return self._min_freq

    @min_freq.setter
    def min_freq(self, value: float | None) -> None:
        self._min_freq = value

    @property
    def has_min_freq(self) -> bool:
        """Whether ``min_freq`` is set."""
        return self._min_freq is not None

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:
        """Returns the string representation of the Discretizer"""
        str_features = str(self.features)
        if len(str_features) > N_CHAR_MAX:
            str_features = str_features[:N_CHAR_MAX] + "..."
        return f"{self.__name__}({str_features})"

    def _cast_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Casts the features of a DataFrame using feature versions to duplicate columns

        Parameters
        ----------
        X : pd.DataFrame
            Dataset used to discretize. Needs to have columns has specified in
            ``BaseDiscretizer.features``, by default None.

        Returns
        -------
        pd.DataFrame
            A formatted X
        """

        # duplicating columns that have several versions
        casted_columns = {
            feature.version: X[feature.name]
            for feature in self.features
            if feature.version != feature.name and feature.version not in X
        }

        # duplicating features with versions disctinct from names (= multiclass target)
        if len(casted_columns) > 0:  # checking for casted feature to not break inplace
            # converting to DataFrame to avoid PerformanceWarning
            casted_columns = pd.DataFrame(casted_columns)
            X = pd.concat([X, casted_columns], axis=1)

        return X

    def _prepare_y(self, y: pd.Series | None) -> None:
        """Validates input y"""

        if not isinstance(y, pd.Series):  # checking for y's type
            raise ValueError(f"[{self.__name__}] y must be a pandas.Series, passed {type(y)}")

        if any(y.isna()):  # checking for nans in the target
            raise ValueError(f"[{self.__name__}] y should not contain numpy.nan")

    def _prepare_X(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validates input X"""

        # checking for X's type
        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"[{self.__name__}] X must be a pandas.DataFrame, passed {type(X)}")

        # copying X
        x_copy = X
        if self.config.copy:
            x_copy = X.copy()

        # checking for input columns by feature name
        missing_columns = [feature for feature in self.features if feature.name not in x_copy]
        if len(missing_columns) > 0:
            raise ValueError(
                f"[{self.__name__}] Requested discretization of {str(missing_columns)} but "
                "those columns are missing from provided X. Please check your inputs! "
            )

        # coercing pandas Categorical-dtype columns to object so that in-place
        # .replace can introduce grouped labels not in the original category set
        # (astype(object), not astype(str), to preserve NaN as np.nan)
        cat_qualitatives = [
            feature.name
            for feature in self.features.qualitatives
            if isinstance(x_copy[feature.name].dtype, pd.CategoricalDtype)
        ]
        if cat_qualitatives:
            x_copy[cat_qualitatives] = x_copy[cat_qualitatives].astype(object)

        # casting features for multiclass targets
        x_copy = self._cast_features(x_copy)

        # checking for input columns by feature version
        missing_columns = [feature for feature in self.features if feature.version not in x_copy]
        if len(missing_columns) > 0:
            raise ValueError(
                f"[{self.__name__}] Requested discretization of {str(missing_columns)} but "
                "those columns are missing from provided X. Please check your inputs! "
            )

        return x_copy

    def _prepare_sample(self, sample: Sample) -> Sample:
        """Validates format and content of X and y.

        Parameters
        ----------
        X : pd.DataFrame
            Dataset used to discretize. Needs to have columns has specified in
            ``BaseDiscretizer.features``, by default None.

        y : pd.Series
            Binary target feature, by default None.

        Returns
        -------
        DataFrame
            A formatted copy of X
        """

        # checking DataFrame of features
        if sample.has_X:
            sample.X = self._prepare_X(sample.X)

            # checking target Series
            if sample.y is not None:
                self._prepare_y(sample.y)

                # checking for matching indices
                if not len(sample.y.index) == len(sample.X.index) or not all(sample.y.index == sample.X.index):
                    raise ValueError(f"[{self.__name__}] X and y must have the same indices.")

        return sample

    # name-mangled alias used by transform() so subclass overrides of _prepare_sample
    # (which add fit-time-only checks) don't break the transform path
    __prepare_sample = _prepare_sample

    def fit(self, X: pd.DataFrame | None = None, y: pd.Series | None = None) -> Self:
        """Learns simple discretization of values of X according to values of y.

        Parameters
        ----------
        X : pd.DataFrame
            Training dataset, to determine features' optimal carving
            Needs to have columns has specified in ``features`` attribute.

        y : pd.Series
            Target with wich the association is maximized.
        """
        _, _ = X, y  # unused arguments

        # checking for previous fits of the discretizer that could cause unwanted errors
        if self.is_fitted:
            raise RuntimeError(
                f"[{self.__name__}] Already fitted. Fitting it anew could break it. Please initialize a new one."
            )

        # checking that all features were fitted
        missing_features = [feature.version for feature in self.features if not feature.is_fitted]
        if len(missing_features) != 0:
            raise RuntimeError(f"[{self.__name__}] Features not fitted: {str(missing_features)}.")

        # setting features in ordinal encoding mode
        self.features.ordinal_encoding = bool(self.config.ordinal_encoding)

        # setting fitted as True to raise alerts
        self.is_fitted = True

        return self

    def transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Applies discretization to a DataFrame's columns.

        Parameters
        ----------
        X : pd.DataFrame
            Dataset to be carved.
            Needs to have columns from provided :class:`Features`.
        y : pd.Series, optional
            Target, by default ``None``

        Returns
        -------
        DataFrame
            Discretized X.
        """
        # checking that it was fitted
        if not self.is_fitted:
            raise RuntimeError(f"[{self.__name__}] Call fit method first.")

        # copying dataframes and casting for multiclass
        sample = self.__prepare_sample(Sample(X, y))

        # converting datetime features to numeric timedeltas before any nan-filling
        sample.X = cast_datetime_features(self.features, sample.X)

        # filling up nans for features that have some
        sample.fillna(self.features)

        # resolving finest modalities of nested features seen only at transform: roll them up to a
        # known parent bucket (parent-aware) or to the default modality, so they don't trip
        # check_values below (see remap_nested_unseen)
        if len(self.features.nested) > 0:
            sample.X = remap_nested_unseen(self.features.nested, sample.X)

        # checking that all unique values in X are in features
        self.features.check_values(sample.X)

        # transforming quantitative features
        if len(self.features.quantitatives) > 0:
            sample = self._transform_quantitative(sample)

        # transforming qualitative features
        if len(self.features.qualitatives) > 0:
            sample = self._transform_qualitative(sample)

        # reinstating nans when not supposed to group them
        return sample.unfillna(self.features)

    def _transform_quantitative(self, sample: Sample) -> Sample:
        """Applies discretization to a DataFrame's Quantitative columns."""
        quantitatives = self.features.quantitatives
        x_len = sample.shape[0]

        # the per-feature transform is a vectorized searchsorted whose C internals release the GIL,
        # so threads overlap the columns with zero pickling — unlike a process pool, whose per-column
        # ship-and-return costs more than the compute (measured). ``n_jobs`` gates the width and stays
        # 1 inside the carver's discretize workers, which keep the no-copy serial path below.
        if self.config.n_jobs > 1 and len(quantitatives) > 1:
            # threaded: hand each thread its own column copy (pandas 2.x has no copy-on-write) so the
            # in-place nan-masking never races on the shared frame.
            def _one(feature: BaseFeature) -> tuple[str, np.ndarray]:
                return transform_quantitative_feature(feature, sample.X[feature.version].copy(), x_len)

            with ThreadPoolExecutor(max_workers=self.config.n_jobs) as executor:
                transformed = list(executor.map(_one, quantitatives))
        else:
            transformed = [
                transform_quantitative_feature(feature, sample.X[feature.version], x_len) for feature in quantitatives
            ]

        # unpacking transformed series — infer_objects restores numeric dtype for ordinal-encoded
        # labels (the per-feature arrays are object dtype); string interval labels stay object,
        # matching the inference the previous list-of-lists path got for free.
        sample.X[[feature for feature, _ in transformed]] = pd.DataFrame(
            dict(transformed), index=sample.index
        ).infer_objects()

        return sample

    def _transform_qualitative(self, sample: Sample) -> Sample:
        """Applies discretization to a DataFrame's Qualitative columns."""
        # list of qualitative features
        qualitatives = self.features.qualitatives

        # replacing values for there corresponding label — per-column ``map`` is far faster than a
        # dict-of-dict ``DataFrame.replace`` on a wide frame. Unmapped values are restored to match
        # ``replace``'s leave-untouched semantics, but only when some value is actually unmapped (the
        # common case is full coverage, which skips ``fillna``'s whole-column alignment pass).
        for feature in qualitatives:
            col = sample.X[feature.version]
            mapped = col.map(feature.label_per_value)
            if mapped.isna().any():
                mapped = mapped.fillna(col)
            sample.X[feature.version] = mapped

        # ordinal_encoding produces integer labels, but the in-place ``replace`` above keeps the
        # column's original object dtype. Cast those columns to numeric so downstream estimators
        # (e.g. XGBoost) accept them, matching the numeric dtype quantitatives get from np.select.
        encoded = [feature.version for feature in qualitatives if feature.ordinal_encoding]
        if encoded:
            sample.X[encoded] = sample.X[encoded].apply(pd.to_numeric)

        return sample

    def _log_if_verbose(self, prefix: str = " -") -> None:
        """prints logs if requested"""
        if self.config.verbose:
            print(f"{prefix} [{self.__name__}] Fit {str(self.features)}")

    def to_json(self, light_mode: bool = False) -> dict:
        """Converts to JSON format.

        To be used with ``json.dump``.

        Parameters
        ----------
        light_mode: bool, optional
            Whether or not to save features' history and statistics, by default False

        Returns
        -------
        str
            JSON serialized object
        """
        content = {
            "features": self.features.to_json(light_mode),
            "min_freq": self._min_freq,
            "is_fitted": self.is_fitted,
            "config": {
                "dropna": self.config.dropna,
                "n_jobs": self.config.n_jobs,
                "verbose": self.config.verbose,
                "ordinal_encoding": self.config.ordinal_encoding,
                "copy": self.config.copy,
                "min_freq_alpha": self.config.min_freq_alpha,
            },
        }

        return content

    def save(self, file_name: str, light_mode: bool = False) -> None:
        """Saves pipeline to .json file.

        Parameters
        ----------
        file_name : str
            String of .json file name.
        light_mode: bool, optional
            Whether or not to save features' history and statistics, by default False

        Returns
        -------
        str
            JSON serialized object
        """
        # checking for input
        if file_name.endswith(".json"):
            with open(file_name, "w", encoding="utf-8") as json_file:
                json.dump(self.to_json(light_mode), json_file)
        # raising for non-json file name
        else:
            raise ValueError(f"[{self.__name__}] Provide a file_name that ends with .json.")

    @classmethod
    def load(cls, file_name: str) -> "BaseDiscretizer":
        """Allows one to load an Discretizer saved as a .json file.

        The Discretizer has to be saved with ``Discretizer.save()``, otherwise there
        can be no guarantee for it to be restored.

        Parameters
        ----------
        file_name : str
            String of saved Discretizer's .json file name.

        Returns
        -------
        BaseDiscretizer
            A fitted Discretizer.
        """
        # reading file
        with open(file_name, encoding="utf-8") as json_file:
            data = json.load(json_file)

        return cls._from_json(data)

    @classmethod
    def _from_json(cls, data: dict) -> "BaseDiscretizer":
        """Builds an instance from the parsed JSON dict (combinations stays a dict
        unless a Carver subclass deserializes it before calling this)."""

        features = Features.load(data.pop("features"))
        is_fitted = data.pop("is_fitted", False)
        min_freq = data.pop("min_freq", None)
        config_data = data.pop("config", {})
        config = ProcessingConfig(
            ordinal_encoding=config_data.get("ordinal_encoding", False),
            dropna=config_data.get("dropna", False),
            verbose=config_data.get("verbose", False),
            n_jobs=config_data.get("n_jobs", 1),
            copy=config_data.get("copy", True),
            min_freq_alpha=config_data.get("min_freq_alpha", 0.05),
        )

        instance = cls(features=features, min_freq=min_freq, config=config)
        instance.is_fitted = is_fitted
        return instance

    @extend_docstring(Features.summary.fget)  # type: ignore
    @property
    def summary(self) -> pd.DataFrame:
        return self.features.summary

    @property
    def history(self) -> pd.DataFrame:
        """History of discretization process for all features"""
        return self.features.history


def remap_nested_unseen(nested_features: list[NestedFeature], X: pd.DataFrame) -> pd.DataFrame:
    """Resolves finest-column modalities of nested features that were never seen at fit.

    For each :class:`NestedFeature`, finest values absent from the learned grouping are rolled up
    to the nearest parent-column value that *is* a known bucket leader (parent-aware), falling back
    to the feature's default modality (``__OTHER__``) when no ancestor resolves or the parent
    columns are absent. Operates in place on ``X`` and returns it.
    """
    for feature in nested_features:
        column = feature.version
        if column not in X:
            continue

        known_values = set(feature.label_per_value)
        bucket_leaders = set(feature.values)

        series = X[column]
        unseen = ~series.isin(known_values) & series.notna() & (series != feature.nan)
        if not bool(unseen.to_numpy().any()):
            continue

        # default fallback for everything unresolved
        resolved = pd.Series(feature.default, index=series.index[unseen], dtype=object)
        unresolved = pd.Series(True, index=resolved.index)

        # walk parent columns nearest→farthest, mapping to the first ancestor that is a bucket
        for parent in feature.parents:
            if parent not in X or not bool(unresolved.to_numpy().any()):
                continue
            parent_values = X.loc[resolved.index, parent]
            hit = unresolved & parent_values.isin(bucket_leaders)
            resolved[hit] = parent_values[hit]
            unresolved &= ~hit

        X.loc[resolved.index, column] = resolved

    return X


def cast_datetime_features(features: Features, X: pd.DataFrame) -> pd.DataFrame:
    """Converts datetime feature columns to a number of seconds since their ``reference_date``.

    Idempotent: columns that are already numeric (converted at fit time) are left untouched,
    so this is safe to call on both raw datetime input and already-converted frames.
    """
    for feature in features.datetimes:
        column = X[feature.version]
        if not pd.api.types.is_numeric_dtype(column):
            reference = None
            if feature.reference_is_column:
                if feature.reference_date not in X:
                    raise ValueError(
                        f"[{feature}] reference column {feature.reference_date!r} is missing "
                        "from provided X. Please check your inputs!"
                    )
                reference = X[feature.reference_date]
            X[feature.version] = feature.to_timedelta(column, reference)
    return X


def transform_quantitative_feature(feature: BaseFeature, df_feature: pd.Series, x_len: int) -> tuple[str, np.ndarray]:
    """Transforms a quantitative feature"""
    del x_len  # kept for call-signature compatibility; searchsorted no longer needs it

    # keeping track of original index
    raw_index = df_feature.index

    # identifying nans
    feature_nans = (df_feature == feature.nan) | df_feature.isna()

    # converting nans to there corresponding quantile (if it was grouped to a quantile)
    if any(feature_nans):
        # quantile with which nans have been grouped
        nan_group = feature.values.get_group(feature.nan)

        # checking that nans have been grouped to a quantile
        if nan_group == feature.nan:
            nan_group = np.nan

        # converting to quantile value if grouped else keeping np.nan
        df_feature.mask(feature_nans, nan_group, inplace=True)

    # ascending quantile thresholds (the last is +inf) and their respective group labels
    thresholds = [value for value in feature.values if value != feature.nan]

    if len(thresholds) == 0:
        # no non-nan modality: mirror np.select's ``default`` (values left untouched)
        grouped = df_feature.to_numpy()
    else:
        threshold_arr = np.asarray(thresholds, dtype=float)
        labels = np.asarray([feature.label_per_value[value] for value in thresholds], dtype=object)

        # post-mask the column is numeric (nans were replaced by their group value above); cast to
        # float so searchsorted compares numerically — the column can arrive as object dtype (it
        # carried the str nan sentinel before masking), which would derail searchsorted otherwise.
        values = np.asarray(df_feature.to_numpy(), dtype=float)

        # "first threshold >= x" is exactly the first matching ``df_feature <= value`` in the
        # original np.select (thresholds are ascending) — one O(N log M) vectorized pass instead
        # of M full-length boolean masks + M length-N python label lists.
        idx = np.searchsorted(threshold_arr, values, side="left")

        # out-of-range indices are exactly the (masked) nan positions — overwritten below; clip so
        # the fancy-index stays valid (matches np.select leaving nans to ``default``).
        np.clip(idx, 0, len(labels) - 1, out=idx)
        grouped = labels[idx]

    df_feature = pd.Series(grouped, index=raw_index)

    # reinstating nans otherwise nan is converted to 'nan' by numpy
    if any(feature_nans):
        df_feature[feature_nans] = feature.label_per_value.get(feature.nan, np.nan)

    # returning the ndarray (not list(...)): the caller rebuilds a DataFrame from these, and
    # materialising a length-N python list per feature was pure overhead across the transform passes.
    return feature.version, df_feature.to_numpy()
