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

    def _counts(self, xagg: XAgg) -> pd.Series:
        """Per-modality observation count. Crosstab families sum the row; continuous counts the list."""
        return xagg.sum(axis=1)  # type: ignore

    def _extra_columns(self, xagg: XAgg) -> dict:
        """Extra per-modality columns beyond rate/frequency/count (continuous adds ``std``)."""
        return {}

    # `compute` is overloaded so that callers passing a non-None ``xagg`` get a
    # non-Optional ``pd.DataFrame`` back — required by `_test_viability_*` and
    # the `BaseFeature.statistics` setter, which don't accept ``None``.
    @overload
    def compute(self, xagg: pd.Series | pd.DataFrame) -> pd.DataFrame: ...
    @overload
    def compute(self, xagg: None) -> None: ...
    def compute(self, xagg: pd.Series | pd.DataFrame | None) -> pd.DataFrame | None:
        """Computes the target rate.

        Parameters
        ----------
        xagg : pd.Series | pd.DataFrame | None
            A crosstab (binary/ordinal/multiclass) or Series-of-y-lists (continuous).

        Returns
        -------
        pd.DataFrame | None
            Target rate frame, or ``None`` if ``xagg`` was ``None``.
        """
        if xagg is None:
            return None
        # count + frequency per modality (count carried for CI-based viability tests).
        # `_counts`/`_compute`/`_extra_columns` expect XAgg (Generic); compute()'s wide
        # signature is for LSP matching, callers always pass this rate's own XAgg kind here.
        count = self._counts(xagg)  # type: ignore
        frequency = count / count.sum()
        return pd.DataFrame(
            {
                self.__name__: self._compute(xagg),  # type: ignore
                "frequency": frequency,
                "count": count,
                **self._extra_columns(xagg),  # type: ignore
            }
        )

    def reference_to_json(self) -> dict | None:
        """JSON-safe snapshot of any per-feature state this rate was fit on.

        Rates carrying fitted state (``TargetMeanRidit``, ``CAScoreRate``) override
        this so a carved feature can recompute its rate on a new sample once the
        evaluator's transient state is gone (see :mod:`AutoCarver.stability`).
        ``None`` means the rate is stateless.
        """
        return None

    def load_reference(self, payload: dict | None) -> None:
        """Restores state produced by :meth:`reference_to_json` (no-op when stateless)."""
