"""Base measures that defines Quantitative and Qualitative features."""

from abc import ABC, abstractmethod

import pandas as pd

from AutoCarver.features import BaseFeature


class BaseMeasure(ABC):
    """Base measure of association between x and y"""

    __name__ = "BaseMeasure"

    is_x_quantitative = False
    """ wether x is quantitative or not """

    is_x_qualitative = False
    """ wether x is qualitative or not """

    is_y_qualitative = False
    """ wether y is qualitative or not """

    is_y_quantitative = False
    """ wether y is quantitative or not """

    is_y_binary = False
    """ wether y is binary or not """

    is_sortable = True
    """ wether the metric can be sorted or not """

    # info
    is_default = False
    """ wether the measure is an association measure or not """

    higher_is_better = True
    """ wether higher values are better or not """

    correlation_with = "target"
    """ info about correlation with which other feature """

    is_absolute = False
    """ wether the measure needs absolute value for comparison or not """

    is_reversible = False
    """ wether the measure's input can be reversed depending on there type or not """

    def __init__(self, threshold: float = 0.0) -> None:
        """
        Parameters
        ----------
        threshold : float, optional
            Minimum threshold to reach, by default ``0.0``
        """
        self.threshold = threshold
        self.value = None
        self._info = {}

    def __repr__(self) -> str:
        return self.__name__

    @abstractmethod
    def compute_association(self, x: pd.Series, y: pd.Series) -> float:
        """Computes association measure between ``x`` and ``y``

        Parameters
        ----------
        x : pd.Series
            Feature
        y : pd.Series
            Target feature

        Returns
        -------
        float
            Measure of association between ``x`` and ``y``
        """

    @property
    def info(self) -> dict:
        """gives info about the measure"""

        return dict(
            self._info,
            # adding default info about the measure
            higher_is_better=self.higher_is_better,
            correlation_with=self.correlation_with,
            is_default=self.is_default,
            is_absolute=self.is_absolute,
        )

    @info.setter
    def info(self, content: dict) -> None:
        """updates info"""
        self._info.update(content)

    def validate(self) -> bool:
        """Checks if :attr:`threshold` is reached

        Returns
        -------
        bool
            Whether the test is passed or not
        """

        if not pd.isnull(self.value) and pd.notna(self.value):
            return self.value >= self.threshold  # type: ignore
        return False

    def to_dict(self) -> dict:
        """converts to a dict"""
        return {
            self.__name__: {
                "value": self.value,
                "threshold": self.threshold,
                "valid": self.validate(),
                "info": self.info,
            }
        }

    def _update_feature(self, feature: BaseFeature) -> None:
        """adds measure to specified feature"""

        # checking for a value
        if self.value is None:
            raise ValueError(f"[{self}] Use compute_association first!")

        # updating statistics of the feature accordingly
        feature.measures.update(self.to_dict())

    def reverse_xy(self) -> bool:
        """reverses values of x and y in compute_association"""

        # when its not implemented
        return False

    def compute_all(self, X: pd.DataFrame, y: pd.Series, features: list[BaseFeature]) -> dict[str, dict]:
        """Batch-computes this measure for every feature, returning ``{version: measure_dict}``.

        This default falls back to a per-feature loop over
        :meth:`compute_association`, so user-supplied custom measures keep
        working. Built-in measures override it with a vectorized kernel (see
        :mod:`AutoCarver.selectors.measures._vectorized`). The returned dict carries
        the same ``{value, threshold, valid, info}`` payload as :meth:`to_dict`.
        """
        results = {}
        for feature in features:
            self.compute_association(X[feature.version], y)
            results[feature.version] = self.to_dict()[self.__name__]
        return results

    def _result(self, value: float | None) -> dict:
        """Builds this measure's per-feature payload from a precomputed ``value``.

        Reuses :meth:`validate` / :attr:`info` so vectorized overrides produce
        the exact same dict shape as the scalar path.
        """
        self.value = value
        return self.to_dict()[self.__name__]


class AbsoluteMeasure(BaseMeasure):
    """Absolute measure of association between x and y"""

    is_absolute = True

    def validate(self) -> bool:
        """Checks if :attr:`threshold` is reached

        Returns
        -------
        bool
            Whether the test is passed or not
        """
        if not pd.isnull(self.value) and pd.notna(self.value):
            return abs(self.value) >= self.threshold  # type: ignore
        return False


class OutlierMeasure(BaseMeasure):
    """Outlier measure of association for a Quantitative feature"""

    is_default = True
    is_x_quantitative = True
    is_x_qualitative = False
    is_sortable = False

    # info
    higher_is_better = False
    correlation_with = "itself"

    def __init__(self, threshold: float = 1.0) -> None:
        """
        Parameters
        ----------
        threshold : float, optional
            Maximum threshold to reach, by default ``1.0``
        """
        super().__init__(threshold)

    @abstractmethod
    def compute_association(self, x: pd.Series, y: pd.Series | None = None) -> float:
        """Computes outlier measure on ``x``

        Parameters
        ----------
        x : pd.Series
            Feature
        y : pd.Series, optional
            Target feature, by default ``None``

        Returns
        -------
        float
            Measure of outliers on ``x``
        """

    def validate(self) -> bool:
        """Checks if :attr:`threshold` is reached

        Returns
        -------
        bool
            Whether the test is passed or not
        """
        if not pd.isnull(self.value) and pd.notna(self.value):
            return self.value < self.threshold  # type: ignore
        return True


class NanMeasure(OutlierMeasure):
    """Measure of the percentage of NaNs"""

    __name__ = "Nan"
    is_x_quantitative = True
    is_x_qualitative = True

    def compute_association(self, x: pd.Series, y: pd.Series | None = None) -> float:
        """Computes frequency of ``nan`` in ``x``

        Parameters
        ----------
        x : pd.Series
            Feature
        y : pd.Series, optional
            Target feature, by default ``None``

        Returns
        -------
        float
            Measure of ``nan`` in ``x``
        """
        _ = y
        self.value = (x.isna() | x.isnull()).mean()
        return self.value


class ModeMeasure(OutlierMeasure):
    """Measure of the percentage of the mode"""

    __name__ = "Mode"
    is_x_quantitative = True
    is_x_qualitative = True

    def compute_association(self, x: pd.Series, y: pd.Series | None = None) -> float:
        """Computes frequency of ``x``'s mode

        Parameters
        ----------
        x : pd.Series
            Feature
        y : pd.Series, optional
            Target feature, by default ``None``

        Returns
        -------
        float
            Measure of ``x``'s mode
        """
        _ = y
        mode = x.mode(dropna=True).values[0]  # computing mode
        self.value = (x == mode).mean()  # Computing percentage of the mode
        return self.value
