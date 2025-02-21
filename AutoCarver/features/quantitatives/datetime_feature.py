""" Defines a datetime feature"""

from enum import Enum

from pandas import Series, to_datetime

from ..utils.base_feature import BaseFeature
from .quantitative_feature import QuantitativeFeature


class DatetimeUnit(str, Enum):
    """Unit of the timedelta"""

    SECONDS = "sc"
    MINUTES = "mn"
    HOURS = "h"
    DAYS = "d"
    WEEKS = "w"
    MONTHS = "m"
    YEARS = "y"

    @classmethod
    def available_units(cls) -> list[str]:
        """returns the available units"""
        return [unit.value for unit in cls]


class DatetimeFeature(QuantitativeFeature):
    """A datetime feature"""

    __name__ = "Datetime"

    is_datetime = True

    def __init__(self, name: str, reference: str, format: str = None, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.reference = reference  # date of reference to be compared with
        self.format = format  # format of the datetime
        if format is None:
            self.format = "mixed"
        self._unit: DatetimeUnit = None  # unit of the timedelta

    @property
    def unit(self) -> DatetimeUnit:
        """returns the unit of the timedelta"""
        return self._unit

    @unit.setter
    def unit(self, unit: DatetimeUnit) -> None:
        """sets the unit of the timedelta"""
        if self._unit is not None:
            raise AttributeError("unit has already been set")
        if unit not in DatetimeUnit.available_units():
            raise ValueError(f"unit must be one of {DatetimeUnit.available_units()}, got {unit}")
        self._unit = unit

    @property
    def unit_value(self) -> int:
        """returns the unit of the timedelta"""
        value = 1
        if self.unit == DatetimeUnit.MINUTES.value:
            value = 60
        elif self.unit == DatetimeUnit.HOURS.value:
            value = 3600
        elif self.unit == DatetimeUnit.DAYS.value:
            value = 86400
        elif self.unit == DatetimeUnit.WEEKS.value:
            value = 604800
        elif self.unit == DatetimeUnit.MONTHS.value:
            value = 2628000
        elif self.unit == DatetimeUnit.YEARS.value:
            value = 31536000
        return value

    def convert_to_timedelta(self, X: Series) -> Series:
        """converts the feature to a timedelta"""

        # convert feature and reference to datetime inplace
        feature_datetime = to_datetime(X[self.version], errors="coerce", format=self.format)
        ref_datetime = to_datetime(X[self.reference], errors="coerce", format=self.format)

        # getting number of seconds
        td_series = (feature_datetime - ref_datetime).dt.total_seconds()

        # converting to the right unit
        return td_series / self.unit_value


def get_datetime_features(features: list[BaseFeature]) -> list[DatetimeFeature]:
    """returns datetime features amongst provided features"""
    return [feature for feature in features if feature.is_datetime]
