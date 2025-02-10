""" Defines a datetime feature"""

from .quantitative_feature import QuantitativeFeature


class DatetimeFeature(QuantitativeFeature):
    """TODO"""

    def __init__(self, name: str, reference_date: str, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.reference_date = reference_date  # date of reference to compare with
