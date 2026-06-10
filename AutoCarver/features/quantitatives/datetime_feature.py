"""Defines a datetime feature"""

from collections.abc import Sequence
from typing import Any

import pandas as pd

from AutoCarver.features.quantitatives.quantitative_feature import QuantitativeFeature
from AutoCarver.features.utils.base_feature import BaseFeature


class DatetimeFeature(QuantitativeFeature):
    """Defines a datetime feature.

    A datetime feature is processed as a :class:`QuantitativeFeature` after its
    values have been converted to a number of seconds elapsed since
    ``reference_date`` (see :meth:`to_timedelta`). The conversion is applied by
    the :class:`TimedeltaDiscretizer` before continuous discretization.

    ``reference_date`` may be either a fixed date literal (e.g. ``"2020-01-01"``)
    or the name of another datetime column in ``X``. The two are disambiguated at
    fit time: if ``reference_date`` matches a column of the fitted ``X``, the
    conversion is computed row-wise against that column; otherwise it is parsed as
    a fixed date.
    """

    __name__ = "Datetime"
    is_datetime = True

    def __init__(self, name: str, reference_date: str) -> None:
        super().__init__(name)
        self.reference_date = reference_date  # fixed date literal or reference column name
        self.reference_is_column = False  # resolved at fit time against X's columns

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        # disambiguate reference_date: a column name resolves to a row-wise reference
        self.reference_is_column = self.reference_date in X.columns
        super().fit(X, y)

        # a NaT in the reference column yields NaN after conversion
        if self.reference_is_column and any(X[self.reference_date].isna()):
            self.has_nan = True

    def to_timedelta(self, series: pd.Series, reference: pd.Series | None = None) -> pd.Series:
        """Converts datetime values to a float number of seconds since the reference.

        When ``reference`` is ``None`` the fixed ``reference_date`` literal is used;
        otherwise ``reference`` is a datetime Series subtracted row-wise (column
        reference). Non-datetime entries (``numpy.nan``, the ``nan`` placeholder,
        unparseable values) are coerced to ``numpy.nan`` so the result is a plain
        float Series.
        """
        dates = pd.to_datetime(series, errors="coerce")
        if reference is None:
            ref = pd.to_datetime(self.reference_date)
        else:
            ref = pd.to_datetime(reference, errors="coerce")
        return (dates - ref).dt.total_seconds()

    def to_json(self, light_mode: bool = False) -> dict[str, Any]:
        content = super().to_json(light_mode)
        content["reference_date"] = self.reference_date
        content["reference_is_column"] = self.reference_is_column
        return content

    def _restore_from_json(self, feature_json: dict) -> None:
        self.reference_date = feature_json["reference_date"]
        self.reference_is_column = feature_json.get("reference_is_column", False)
        super()._restore_from_json(feature_json)


def get_datetime_features(features: Sequence[BaseFeature]) -> list[DatetimeFeature]:
    """returns datetime features amongst provided features"""
    return [feature for feature in features if isinstance(feature, DatetimeFeature)]
