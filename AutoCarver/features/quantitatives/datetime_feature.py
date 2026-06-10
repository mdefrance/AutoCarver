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
    """

    __name__ = "Datetime"
    is_datetime = True

    def __init__(self, name: str, reference_date: str) -> None:
        super().__init__(name)
        self.reference_date = reference_date  # date of reference to compare with

    def to_timedelta(self, series: pd.Series) -> pd.Series:
        """Converts datetime values to a float number of seconds since ``reference_date``.

        Non-datetime entries (``numpy.nan``, the ``nan`` placeholder, unparseable
        values) are coerced to ``numpy.nan`` so the result is a plain float Series.
        """
        reference = pd.to_datetime(self.reference_date)
        dates = pd.to_datetime(series, errors="coerce")
        return (dates - reference).dt.total_seconds()

    def to_json(self, light_mode: bool = False) -> dict[str, Any]:
        content = super().to_json(light_mode)
        content["reference_date"] = self.reference_date
        return content

    def _restore_from_json(self, feature_json: dict) -> None:
        self.reference_date = feature_json["reference_date"]
        super()._restore_from_json(feature_json)


def get_datetime_features(features: Sequence[BaseFeature]) -> list[DatetimeFeature]:
    """returns datetime features amongst provided features"""
    return [feature for feature in features if isinstance(feature, DatetimeFeature)]
