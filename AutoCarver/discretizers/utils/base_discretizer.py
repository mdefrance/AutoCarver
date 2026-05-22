"""Base tools to build simple buckets out of Quantitative and Qualitative features
for a binary classification model.
"""

import json
from abc import ABC
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from AutoCarver.discretizers.utils.multiprocessing import apply_async_function
from AutoCarver.features import BaseFeature, Features
from AutoCarver.utils import extend_docstring


@dataclass
class DiscretizerConfig:
    """Behavioral configuration applied to a :class:`BaseDiscretizer`.

    Carries only cross-cutting toggles that propagate unchanged to sub-discretizers.
    Domain parameters (``min_freq``, ``combinations`` …) are explicit constructor
    arguments, not config.

    ``copy=True`` is the default so that BaseDiscretizer doesn't mutate caller
    DataFrames in place — set to ``False`` when nested inside a pipeline that
    already owns the dataframe.
    """

    copy: bool = True
    ordinal_encoding: bool = False
    dropna: bool = False
    verbose: bool = False
    n_jobs: int = 1


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

    def __init__(
        self,
        features: "Features | Iterable[BaseFeature]",
        *,
        min_freq: float | None = None,
        config: DiscretizerConfig | None = None,
    ) -> None:
        """
        Parameters
        ----------

        features : Features
            A set of :class:`Features` to be processed

        min_freq : float, optional
            Minimum frequency per modality per feature, by default ``None``

            * Features need at least one modality more frequent than :attr:`min_freq`
            * Defines number of quantiles of continuous features
            * Minimum frequency of modality of quantitative features

            .. tip::
                Set between ``0.01`` (slower, less robust) and ``0.2`` (faster, more robust)

        config : DiscretizerConfig, optional
            Behavioral toggles (``copy``/``ordinal_encoding``/``dropna``/``verbose``/``n_jobs``),
            by default a default-initialized :class:`DiscretizerConfig`.
        """
        # accept either a Features collection or an iterable of BaseFeature
        if isinstance(features, Features):
            self.features: Features = features
        else:
            self.features = Features.from_list(features)

        self.config: DiscretizerConfig = config if config is not None else DiscretizerConfig()
        self.min_freq = min_freq

        # set by subclasses; serialized for round-trip but not used by BaseDiscretizer itself
        # lifecycle flag — set by fit(), or by load() after restoring state
        self.is_fitted: bool = False

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

    def _prepare_y(self, y: pd.Series) -> None:
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

    def _prepare_data(self, sample: Sample) -> Sample:
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

    # name-mangled alias used by transform() so subclass overrides of _prepare_data
    # (which add fit-time-only checks) don't break the transform path
    __prepare_data = _prepare_data

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
        self.features.ordinal_encoding = self.config.ordinal_encoding

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
        sample = self.__prepare_data(Sample(X, y))

        # filling up nans for features that have some
        sample.fillna(self.features)

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
        # transforming all features
        transformed = apply_async_function(
            transform_quantitative_feature,
            self.features.quantitatives,
            self.config.n_jobs,
            sample.X,
            sample.shape[0],
        )

        # unpacking transformed series
        sample.X[[feature for feature, _ in transformed]] = pd.DataFrame(dict(transformed), index=sample.index)

        return sample

    def _transform_qualitative(self, sample: Sample) -> Sample:
        """Applies discretization to a DataFrame's Qualitative columns."""
        # list of qualitative features
        qualitatives = self.features.qualitatives

        # replacing values for there corresponding label
        sample.X.replace({feature.version: feature.label_per_value for feature in qualitatives}, inplace=True)

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
            "min_freq": self.min_freq,
            "is_fitted": self.is_fitted,
            "config": {
                "dropna": self.config.dropna,
                "n_jobs": self.config.n_jobs,
                "verbose": self.config.verbose,
                "ordinal_encoding": self.config.ordinal_encoding,
                "copy": self.config.copy,
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
        config = DiscretizerConfig(
            ordinal_encoding=config_data.get("ordinal_encoding", False),
            dropna=config_data.get("dropna", False),
            verbose=config_data.get("verbose", False),
            n_jobs=config_data.get("n_jobs", 1),
            copy=config_data.get("copy", True),
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


def transform_quantitative_feature(feature: BaseFeature, df_feature: pd.Series, x_len: int) -> tuple[str, list]:
    """Transforms a quantitative feature"""

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

    # list of masks of values to replace with there respective group
    values_to_group = [df_feature <= value for value in feature.values if value != feature.nan]

    # corressponding group for each value
    group_labels = [[feature.label_per_value[value]] * x_len for value in feature.values if value != feature.nan]

    df_feature = pd.Series(np.select(values_to_group, group_labels, default=df_feature), index=raw_index)

    # reinstating nans otherwise nan is converted to 'nan' by numpy
    if any(feature_nans):
        df_feature[feature_nans] = feature.label_per_value.get(feature.nan, np.nan)

    return feature.version, list(df_feature)
