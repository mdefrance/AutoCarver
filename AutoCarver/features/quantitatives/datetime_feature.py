"""Defines a datetime feature"""

from collections.abc import Sequence
from typing import Any, Literal, get_args

import pandas as pd

from AutoCarver.features.quantitatives.quantitative_feature import QuantitativeFeature
from AutoCarver.features.utils.base_feature import BaseFeature, features_of_type

Resolution = Literal["s", "ms", "us", "ns"]  # units accepted by pandas' as_unit
RESOLUTIONS = get_args(Resolution)


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

    ``resolution`` is the unit (``"s"``, ``"ms"``, ``"us"`` or ``"ns"``) both sides are cast
    to before subtracting. pandas stores datetimes as int64 counts of that unit, so the
    default ``"s"`` handles dates up to ~292 billion years apart, while ``"ns"`` overflows
    beyond ~292 years (e.g. a 1683 placeholder date against a 2025 reference). Finer units
    are only needed for sub-second precision.
    """

    __name__ = "Datetime"
    is_datetime = True

    def __init__(self, name: str, reference_date: str, resolution: Resolution = "s") -> None:
        super().__init__(name)
        self.reference_date = reference_date  # fixed date literal or reference column name
        self.resolution = resolution  # unit both sides are cast to before subtracting
        # one column can be carved against several references: each gets its own version (column)
        self.version = f"{name}__ref={reference_date}"
        self.reference_is_column = False  # resolved at fit time against X's columns

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        if self.resolution not in RESOLUTIONS:
            raise ValueError(f"[{self}] resolution {self.resolution!r} must be one of {RESOLUTIONS}.")

        # disambiguate reference_date: a column name resolves to a row-wise reference
        self.reference_is_column = self.reference_date in X.columns

        # otherwise it must be a parseable fixed-date literal: catch the common case
        # of a reference column that was dropped before fit (e.g. df[features] selection)
        if not self.reference_is_column and pd.isna(pd.to_datetime(self.reference_date, errors="coerce")):
            raise ValueError(
                f"[{self}] reference_date {self.reference_date!r} is neither a column of X nor a "
                "parseable date. If it is meant to be a reference column, make sure it is kept in X."
            )

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

        When only one side is timezone-aware, the naive side is read in that timezone
        (local times that don't exist or are ambiguous in it become ``numpy.nan``).

        Both sides are cast to ``resolution`` before subtracting (see the class docstring).
        """
        dates = pd.to_datetime(series, errors="coerce")
        if reference is None:
            ref = pd.to_datetime(self.reference_date)
        else:
            ref = pd.to_datetime(reference, errors="coerce")

        dates_tz, ref_tz = _timezone(dates), _timezone(ref)
        if dates_tz is None and ref_tz is not None:
            dates = dates.dt.tz_localize(ref_tz, ambiguous="NaT", nonexistent="NaT")
        elif dates_tz is not None and ref_tz is None:
            ref = _localize(ref, dates_tz)

        try:
            return (dates.dt.as_unit(self.resolution) - _as_unit(ref, self.resolution)).dt.total_seconds()
        except OverflowError as error:
            raise OverflowError(
                f"[{self}] dates are too far apart from reference_date {self.reference_date!r} at "
                f"resolution {self.resolution!r}; use a coarser resolution (e.g. 's')."
            ) from error

    def to_json(self, light_mode: bool = False) -> dict[str, Any]:
        content = super().to_json(light_mode)
        content["reference_date"] = self.reference_date
        content["reference_is_column"] = self.reference_is_column
        content["resolution"] = self.resolution
        return content

    def _restore_from_json(self, feature_json: dict) -> None:
        self.reference_date = feature_json["reference_date"]
        self.reference_is_column = feature_json.get("reference_is_column", False)
        self.resolution = feature_json.get("resolution", "s")
        super()._restore_from_json(feature_json)


def _as_unit(value: "pd.Series | pd.Timestamp", unit: Resolution) -> "pd.Series | pd.Timestamp":
    """Casts a datetime Series or Timestamp to ``unit`` resolution."""
    return value.as_unit(unit) if isinstance(value, pd.Timestamp) else value.dt.as_unit(unit)


def _timezone(value: "pd.Series | pd.Timestamp") -> Any:
    """Timezone of a datetime Series or Timestamp (``None`` when naive)."""
    return value.tz if isinstance(value, pd.Timestamp) else value.dt.tz


def _localize(value: "pd.Series | pd.Timestamp", tz: Any) -> "pd.Series | pd.Timestamp":
    """Reads a naive datetime Series or Timestamp in ``tz``."""
    if isinstance(value, pd.Timestamp):
        return value.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    return value.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")


def get_datetime_features(features: Sequence[BaseFeature]) -> list[DatetimeFeature]:
    """returns datetime features amongst provided features"""
    return features_of_type(features, DatetimeFeature)
