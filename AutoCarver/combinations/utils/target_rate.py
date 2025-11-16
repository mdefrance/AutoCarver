"""defines a target rate"""

from abc import ABC, abstractmethod

from pandas import DataFrame, Series


class TargetRate(ABC):
    """Target rate class."""

    __name__ = "target_rate"

    @abstractmethod
    def _compute(self, xagg: DataFrame | Series) -> Series:
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
    def compute(self, xagg: DataFrame | Series) -> Series:
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
