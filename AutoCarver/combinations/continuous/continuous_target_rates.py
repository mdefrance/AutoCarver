""" set of target rates for binary classification """

from abc import ABC

from numpy import mean, median
from pandas import DataFrame, Series

from ..utils import TargetRate


class ContinuousTargetRate(TargetRate, ABC):
    """Continuous target rate class."""

    __name__ = "continuous_target_rate"

    def compute(self, xagg: Series) -> DataFrame:
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
        # checking for an xtab
        if xagg is not None:
            # frequency per modality
            frequency = xagg.apply(len) / xagg.apply(len).sum()

            # computing target rate
            return DataFrame({self.__name__: self._compute(xagg), "frequency": frequency})
        return None


class TargetMean(ContinuousTargetRate):
    """Mean target rate class."""

    __name__ = "target_mean"

    def _compute(self, xagg: DataFrame) -> Series:
        """Computes the mean target rate."""
        return xagg.apply(mean)


class TargetMedian(ContinuousTargetRate):
    """Median of target per class."""

    __name__ = "target_median"

    def _compute(self, xagg: DataFrame) -> Series:
        """Computes the mean target rate."""
        return xagg.apply(median)


# class TargetVariance(ContinuousTargetRate):
#     """Variance of target per class."""

#     __name__ = "target_variance"

#     def _compute(self, xagg: DataFrame) -> Series:
#         """Computes the mean target rate."""
#         return xagg.apply(var)


# class TargetStd(ContinuousTargetRate):
#     """Std of target per class."""

#     __name__ = "target_std"

#     def _compute(self, xagg: DataFrame) -> Series:
#         """Computes the mean target rate."""
#         return xagg.apply(std)


# class TargetIqr(ContinuousTargetRate):
#     """IQR of target per class."""

#     __name__ = "target_iqr"

#     def _compute(self, xagg: DataFrame) -> Series:
#         """Computes the mean target rate."""
#         return xagg.apply(iqr)


# class TargetRange(ContinuousTargetRate):
#     """Range of target per class."""

#     __name__ = "target_range"

#     def _compute(self, xagg: DataFrame) -> Series:
#         """Computes the mean target rate."""
#         return xagg.apply(lambda x: x.max() - x.min())
