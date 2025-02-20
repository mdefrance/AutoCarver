""" defines a target rate"""

from abc import ABC, abstractmethod
from typing import Union

from pandas import DataFrame, Series


class TargetRate(ABC):
    """Target rate class."""

    __name__ = "target_rate"

    @abstractmethod
    def _compute(self, xagg: Union[DataFrame, Series]) -> Series:
        """Computes the target rate.

        Parameters
        ----------
        xagg : DataFrame
            A crosstab.

        Returns
        -------
        Series
            Target rate.
        """

    @abstractmethod
    def compute(self, xagg: Union[DataFrame, Series]) -> Series:
        """Computes the target rate.

        Parameters
        ----------
        xagg : DataFrame
            A crosstab.

        Returns
        -------
        Series
            Target rate.
        """
