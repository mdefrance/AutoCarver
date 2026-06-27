"""set of tests for datetime features"""

import numpy as np
import pandas as pd
from pytest import fixture

from AutoCarver.features.quantitatives.datetime_feature import DatetimeFeature, get_datetime_features
from AutoCarver.features.quantitatives.quantitative_feature import QuantitativeFeature
from AutoCarver.features.utils.base_feature import BaseFeature

BaseFeature.__abstractmethods__ = set()


@fixture
def sample_datetime_feature() -> DatetimeFeature:
    """Create a sample DatetimeFeature for testing"""
    return DatetimeFeature("test_feature", reference_date="2020-01-01")


def test_datetime_feature_type(sample_datetime_feature: DatetimeFeature) -> None:
    """a datetime feature is a quantitative feature flagged as datetime"""
    assert sample_datetime_feature.is_datetime
    assert sample_datetime_feature.is_quantitative
    assert not sample_datetime_feature.is_qualitative
    assert not sample_datetime_feature.is_ordinal
    assert not sample_datetime_feature.is_categorical
    assert sample_datetime_feature.reference_date == "2020-01-01"


def test_to_timedelta(sample_datetime_feature: DatetimeFeature) -> None:
    """datetime values are converted to seconds since reference_date"""
    series = pd.Series(
        [
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2020-01-02"),
            pd.Timestamp("2020-01-01 01:00:00"),
            pd.NaT,
        ]
    )
    result = sample_datetime_feature.to_timedelta(series)

    assert result.tolist()[:3] == [0.0, 86400.0, 3600.0]
    assert np.isnan(result.tolist()[3])


def test_to_timedelta_negative(sample_datetime_feature: DatetimeFeature) -> None:
    """values before reference_date give negative timedeltas"""
    series = pd.Series(["2019-12-31"])
    result = sample_datetime_feature.to_timedelta(series)
    assert result.tolist() == [-86400.0]


def test_to_timedelta_column_reference(sample_datetime_feature: DatetimeFeature) -> None:
    """a reference Series gives per-row seconds between the two datetime columns"""
    series = pd.Series([pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-02"), pd.NaT])
    reference = pd.Series([pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01")])
    result = sample_datetime_feature.to_timedelta(series, reference)

    assert result.tolist()[:2] == [172800.0, 86400.0]
    assert np.isnan(result.tolist()[2])


def test_to_timedelta_column_reference_with_nat_reference(sample_datetime_feature: DatetimeFeature) -> None:
    """a NaT in the reference column yields NaN for that row"""
    series = pd.Series([pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")])
    reference = pd.Series([pd.NaT, pd.Timestamp("2020-01-01")])
    result = sample_datetime_feature.to_timedelta(series, reference)

    assert np.isnan(result.tolist()[0])
    assert result.tolist()[1] == 172800.0


def test_fit_resolves_reference_is_column() -> None:
    """fit flags whether reference_date names a column or is a literal date"""
    # literal date -> not a column
    literal = DatetimeFeature("event", reference_date="2020-01-01")
    X_literal = pd.DataFrame({"event": pd.to_datetime(["2020-01-02", "2020-01-03"])})
    literal.fit(X_literal)
    assert literal.reference_is_column is False

    # column name -> resolved as a column reference
    column = DatetimeFeature("event", reference_date="signup")
    X_column = pd.DataFrame(
        {
            "event": pd.to_datetime(["2020-01-02", "2020-01-03"]),
            "signup": pd.to_datetime(["2020-01-01", "2020-01-01"]),
        }
    )
    column.fit(X_column)
    assert column.reference_is_column is True


def test_reference_is_column_defaults_false(sample_datetime_feature: DatetimeFeature) -> None:
    """before fit, a feature behaves as a fixed-date reference"""
    assert sample_datetime_feature.reference_is_column is False


def test_fit_raises_on_missing_reference_column() -> None:
    """a reference_date that is neither a column nor a parseable date raises a clear error"""
    feature = DatetimeFeature("event", reference_date="DT_MEL2")
    X = pd.DataFrame({"event": pd.to_datetime(["2020-01-02", "2020-01-03"])})

    from pytest import raises

    with raises(ValueError, match="reference column"):
        feature.fit(X)


def test_column_reference_json_round_trip() -> None:
    """reference_is_column survives a to_json/load round trip"""
    feature = DatetimeFeature("event", reference_date="signup")
    X = pd.DataFrame(
        {
            "event": pd.to_datetime(["2020-01-02"]),
            "signup": pd.to_datetime(["2020-01-01"]),
        }
    )
    feature.fit(X)

    feature_json = feature.to_json()
    assert feature_json["reference_date"] == "signup"
    assert feature_json["reference_is_column"] is True

    loaded = DatetimeFeature.load(feature_json)
    assert loaded.reference_date == "signup"
    assert loaded.reference_is_column is True


def test_get_datetime_features() -> None:
    """test function get_datetime_features"""
    # no value
    assert get_datetime_features([]) == []

    # mixed
    feature1 = DatetimeFeature("feature1", reference_date="2020-01-01")
    feature2 = QuantitativeFeature("feature2")
    feature3 = BaseFeature("feature3")
    assert get_datetime_features([feature1, feature2, feature3]) == [feature1]


def test_datetime_feature_json_round_trip(sample_datetime_feature: DatetimeFeature) -> None:
    """reference_date and type survive a to_json/load round trip"""
    feature_json = sample_datetime_feature.to_json()
    assert feature_json["is_datetime"] is True
    assert feature_json["reference_date"] == "2020-01-01"

    loaded = DatetimeFeature.load(feature_json)
    assert isinstance(loaded, DatetimeFeature)
    assert loaded.reference_date == "2020-01-01"
    assert loaded.is_datetime
