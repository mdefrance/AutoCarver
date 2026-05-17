"""Defines a datetime feature"""

from AutoCarver.features.quantitatives.quantitative_feature import QuantitativeFeature


class DatetimeFeature(QuantitativeFeature):
    """TODO"""

    def __init__(self, name: str, reference_date: str) -> None:
        super().__init__(name)
        self.reference_date = reference_date  # date of reference to compare with
