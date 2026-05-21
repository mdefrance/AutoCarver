"""defines a target rate"""

from abc import ABC, abstractmethod

import pandas as pd


class TargetRate(ABC):
    """Target rate class."""

    __name__ = "target_rate"

    @abstractmethod
    def _compute(self, xagg: pd.Series | pd.DataFrame) -> pd.Series:
        """Computes the target rate."""

    @abstractmethod
    def compute(self, xagg: pd.Series | pd.DataFrame | None) -> pd.Series:
        """Computes the target rate.

        Parameters
        ----------
        xagg : pd.Series | pd.DataFrame | None
            A crosstab.

        Returns
        -------
        pd.Series
            Target rate.
        """
