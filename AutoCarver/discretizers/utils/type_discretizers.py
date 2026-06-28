"""Base tools to convert values into specific types."""

from typing import Self

import pandas as pd

from AutoCarver.discretizers.utils.base_discretizer import (
    BaseDiscretizer,
    ProcessingConfig,
    Sample,
    cast_datetime_features,
)
from AutoCarver.discretizers.utils.multiprocessing import apply_async_function
from AutoCarver.features import BaseFeature, DatetimeFeature, Features, GroupedList
from AutoCarver.features.qualitatives import nan_unique
from AutoCarver.utils import extend_docstring


class StringDiscretizer(BaseDiscretizer):
    """Converts specified columns of a DataFrame into strings.
    First step of a Qualitative discretization pipeline.

    * Keeps NaN inplace
    * Converts floats of int to int
    """

    __name__ = "StringDiscretizer"

    @extend_docstring(BaseDiscretizer.__init__, exclude=["min_freq"])
    def __init__(
        self,
        features: Features,
        *,
        config: ProcessingConfig | None = None,
    ) -> None:
        features_obj = Features.from_list(features)
        super().__init__(features=features_obj, config=config)

    @extend_docstring(BaseDiscretizer.fit)
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        self._log_if_verbose()  # verbose if requested

        # checking for binary target and copying X
        sample = self._prepare_sample(Sample(X, y))

        # transforming all features — kept serial (n_jobs=1): per-feature string-casting is cheap and
        # pickling each column to a worker dominates. n_jobs is reserved for the carver's search.
        all_orders = apply_async_function(fit_feature, self.features, 1, sample.X)

        # updating features accordingly
        self.features.update(dict(all_orders), replace=True)
        self.features.fit(**sample)

        # discretizing features based on each feature's values_order
        super().fit(X, y)

        return self


def fit_feature(feature: BaseFeature, df_feature: pd.Series):
    """fits one feature"""

    # unique feature values
    unique_values = nan_unique(df_feature)

    # initiating feature values
    order = GroupedList(unique_values)
    if not feature.values.is_empty():
        order = GroupedList(feature.values)

    # formatting values
    for value in unique_values:
        # case 0: the value is an integer
        if isinstance(value, float) and float.is_integer(value):
            str_value = str(int(value))  # converting value to string

        # case 1: converting to string
        else:
            str_value = str(value)

        # adding missing str_value to the order (for provided orders)
        if str_value not in order:
            order.append(str_value)

        # adding float/int values to the order (for provided orders)
        if value not in order:
            order.append(value)

        # grouping float/int value into the str value
        order.group(value, str_value)

    return feature.version, order


def ensure_qualitative_dtypes(
    features: Features,
    X: pd.DataFrame,
    *,
    config: ProcessingConfig | None = None,
) -> pd.DataFrame:
    """Checks features' data types and converts int/float to str"""

    # a column needs string-conversion unless every non-null value is already a python str.
    # ``infer_dtype`` is a C-level scan that short-circuits on the first off-type value — far cheaper
    # than ``pd.unique`` + a python ``type()`` loop on high-cardinality object columns. nan is filled
    # with the str sentinel first so an all-nan column (else inferred "empty") still counts as string.
    to_convert = [
        feature
        for feature in features
        if pd.api.types.infer_dtype(X[feature.version].fillna(feature.nan), skipna=True) != "string"
    ]

    # converting detected non-string features
    if to_convert:
        string_discretizer = StringDiscretizer(features=to_convert, config=config)
        X = string_discretizer.fit_transform(X)

    # pandas 3.0 infers StringDtype for string columns; enforce object dtype for consistency — but
    # only for columns that aren't already object (the common case), so we don't rebuild the whole
    # qualitative block on every call.
    non_object = [feature.version for feature in features if X[feature.version].dtype != object]
    if non_object:
        X[non_object] = X[non_object].astype(object)

    return X


class TimedeltaDiscretizer(BaseDiscretizer):
    """Converts datetime features into floats: the number of seconds elapsed since each
    feature's ``reference_date``.

    Quantitative counterpart of :class:`StringDiscretizer`: a type-conversion step run
    before :class:`ContinuousDiscretizer` so that datetime columns can be discretized as
    ordinary quantitative features.

    * Keeps NaN inplace
    """

    __name__ = "TimedeltaDiscretizer"

    @extend_docstring(BaseDiscretizer.__init__)
    def __init__(
        self,
        features: list[DatetimeFeature],
        *,
        config: ProcessingConfig | None = None,
    ) -> None:
        """
        Parameters
        ----------
        features : list[DatetimeFeature]
            List of datetime features to be converted to second-based timedeltas
        """
        features_obj = Features.from_list(features)
        super().__init__(features=features_obj, config=config)

    @extend_docstring(BaseDiscretizer.fit)
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        self._log_if_verbose()  # verbose if requested

        # checking for binary target and copying X
        sample = self._prepare_sample(Sample(X, y))

        # fitting features on the raw datetime columns (records has_nan)
        self.features.fit(**sample)

        # marking the discretizer as fitted (values are set later by ContinuousDiscretizer)
        super().fit(X, y)

        return self

    def transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Converts each datetime feature's column to seconds since its ``reference_date``."""
        if not self.is_fitted:
            raise RuntimeError(f"[{self.__name__}] Call fit method first.")

        # validating and casting columns, then converting datetimes to numeric timedeltas
        X = self._prepare_X(X)
        return cast_datetime_features(self.features, X)


def ensure_datetime_dtypes(
    features: Features,
    X: pd.DataFrame,
    *,
    config: ProcessingConfig | None = None,
) -> pd.DataFrame:
    """Converts datetime features into second-based timedeltas (no-op when there are none)."""

    datetimes = features.datetimes
    if datetimes:
        timedelta_discretizer = TimedeltaDiscretizer(features=datetimes, config=config)
        X = timedelta_discretizer.fit_transform(X)

    return X
