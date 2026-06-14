"""Set of tests for the TimedeltaDiscretizer."""

import numpy as np
import pandas as pd

from AutoCarver.discretizers.discretizer import Discretizer
from AutoCarver.discretizers.utils.base_discretizer import ProcessingConfig
from AutoCarver.discretizers.utils.type_discretizers import TimedeltaDiscretizer, ensure_datetime_dtypes
from AutoCarver.features import DatetimeFeature, Features


def test_timedelta_discretizer_initialization() -> None:
    """Tests the initialization of the TimedeltaDiscretizer class"""
    feature = DatetimeFeature("feature1", reference_date="2020-01-01")
    timedelta_discretizer = TimedeltaDiscretizer([feature])
    assert isinstance(timedelta_discretizer.features, Features)
    assert timedelta_discretizer.features.datetimes[0].reference_date == "2020-01-01"


def test_timedelta_discretizer_fit_transform() -> None:
    """Tests fit/transform: datetime columns are converted to seconds since reference_date"""
    feature = DatetimeFeature("feature1", reference_date="2020-01-01")
    timedelta_discretizer = TimedeltaDiscretizer([feature])

    X = pd.DataFrame({"feature1": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", np.nan])})

    timedelta_discretizer.fit(X)
    assert timedelta_discretizer.features["feature1"].is_fitted
    assert timedelta_discretizer.features["feature1"].has_nan

    transformed_x = timedelta_discretizer.transform(X)
    result = transformed_x["feature1"].tolist()
    assert result[:3] == [0.0, 86400.0, 172800.0]
    assert np.isnan(result[3])


def test_timedelta_discretizer_column_reference() -> None:
    """fit/transform converts a datetime column to seconds since another datetime column"""
    feature = DatetimeFeature("event", reference_date="signup")
    timedelta_discretizer = TimedeltaDiscretizer([feature])

    X = pd.DataFrame(
        {
            "event": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-11", np.nan]),
            "signup": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-01", "2020-01-01"]),
        }
    )

    timedelta_discretizer.fit(X)
    assert timedelta_discretizer.features["event"].reference_is_column

    transformed_x = timedelta_discretizer.transform(X)
    result = transformed_x["event"].tolist()
    assert result[:3] == [86400.0, 172800.0, 864000.0]
    assert np.isnan(result[3])


def test_ensure_datetime_dtypes_converts_only_datetimes() -> None:
    """ensure_datetime_dtypes converts datetime features and leaves numerics untouched"""
    datetime_feature = DatetimeFeature("dt", reference_date="2020-01-01")
    features = Features.from_list([datetime_feature])

    X = pd.DataFrame({"dt": pd.to_datetime(["2020-01-01", "2020-01-02"])})
    converted = ensure_datetime_dtypes(features, X)

    assert converted["dt"].tolist() == [0.0, 86400.0]


def test_ensure_datetime_dtypes_noop_without_datetimes() -> None:
    """ensure_datetime_dtypes is a no-op when there are no datetime features"""
    features = Features(numericals=["num"])
    X = pd.DataFrame({"num": [1.0, 2.0, 3.0]})

    converted = ensure_datetime_dtypes(features, X)
    assert converted["num"].tolist() == [1.0, 2.0, 3.0]


def test_datetime_feature_in_full_discretizer_pipeline() -> None:
    """A DatetimeFeature is discretized into quantile buckets through the Discretizer,
    and a subsequent transform of fresh raw datetimes reuses the learned buckets."""
    n = 200
    feature = DatetimeFeature("dt", reference_date="2020-01-01")
    features = Features.from_list([feature])

    X = pd.DataFrame({"dt": pd.date_range("2020-01-01", periods=n, freq="D")})
    X.loc[5, "dt"] = pd.NaT
    y = pd.Series(np.arange(n) % 2, index=X.index)

    discretizer = Discretizer(features, min_freq=0.1, config=ProcessingConfig(copy=True))
    discretizer.fit(X, y)

    # quantiles are learned on seconds-since-reference and capped with np.inf
    assert feature.is_fitted
    assert feature.has_nan
    assert feature.values[-1] == np.inf
    assert all(isinstance(value, float) for value in feature.values)

    # transforming fresh raw datetimes routes through the learned buckets
    transformed = discretizer.transform(X)
    assert transformed["dt"].notna().sum() == n - 1  # the single NaT stays NaN
    assert set(transformed["dt"].dropna().unique()).issubset(set(feature.labels))
