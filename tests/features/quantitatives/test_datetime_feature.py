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
