"""defines a target rate"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, overload

import pandas as pd

# parametric type: target rates and evaluators are split by the *kind* of
# aggregated input — DataFrame crosstabs for binary, Series-of-lists for
# continuous. A value-constrained TypeVar prevents accidental third instantiations.
XAgg = TypeVar("XAgg", pd.Series, pd.DataFrame)


class TargetRate(ABC, Generic[XAgg]):
    """Target rate class.

    Generic over ``XAgg`` for the inner :meth:`_compute` worker so that
    binary (``DataFrame`` crosstabs) and continuous (``Series`` of y-lists)
    subclasses don't violate LSP by narrowing the worker's parameter type.
    The outer :meth:`compute` keeps a wide ``Series | DataFrame | None``
    signature because call sites in :class:`CombinationEvaluator` and
    :class:`BaseCarver` carry that union directly from
    :class:`AggregatedSample.raw` / pretty-printer plumbing.
    """

    __name__ = "target_rate"

    @abstractmethod
    def _compute(self, xagg: XAgg) -> pd.Series:
        """Computes the target rate."""

    # `compute` is overloaded so that callers passing a non-None ``xagg`` get a
    # non-Optional ``pd.DataFrame`` back — required by `_test_viability_*` and
    # the `BaseFeature.statistics` setter, which don't accept ``None``.
    @overload
    def compute(self, xagg: pd.Series | pd.DataFrame) -> pd.DataFrame: ...
    @overload
    def compute(self, xagg: None) -> None: ...
    @abstractmethod
    def compute(self, xagg: pd.Series | pd.DataFrame | None) -> pd.DataFrame | None:
        """Computes the target rate.

        Parameters
        ----------
        xagg : pd.Series | pd.DataFrame | None
            A crosstab (binary) or Series-of-y-lists (continuous).

        Returns
        -------
        pd.DataFrame | None
            Target rate frame, or ``None`` if ``xagg`` was ``None``.
        """
