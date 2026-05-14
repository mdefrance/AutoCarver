"""defines a target rate"""

from abc import ABC, abstractmethod

import pandas as pd
from pandas import DataFrame


class TargetRate(ABC):
    """Target rate class."""

    __name__ = "target_rate"

    @abstractmethod
    def _compute(self, xagg: pd.Series | DataFrame) -> pd.Series:
        """Computes the target rate.

        Parameters
        ----------
        xagg : pd.DataFrame
            A crosstab.

        Returns
        -------
        Series
            Target rate.
        """

    @abstractmethod
    def compute(self, xagg: pd.Series | DataFrame) -> pd.Series:
        """Computes the target rate.

        Parameters
        ----------
        xagg : pd.DataFrame
            A crosstab.

        Returns
        -------
        Series
            Target rate.
        """
