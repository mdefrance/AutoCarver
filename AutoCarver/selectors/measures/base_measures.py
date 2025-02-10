""" Base measures that defines Quantitative and Qualitative features.
"""

from abc import ABC, abstractmethod

from pandas import Series, isnull, notna

from ...features import BaseFeature


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
    def compute_association(self, x: Series, y: Series) -> float:
        """Computes association measure between ``x`` and ``y``

        Parameters
        ----------
        x : Series
            Feature
        y : Series
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

        if not isnull(self.value) and notna(self.value):
            return self.value >= self.threshold
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
        if not isnull(self.value) and notna(self.value):
            return abs(self.value) >= self.threshold
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
    def compute_association(self, x: Series, y: Series = None) -> float:
        """Computes outlier measure on ``x``

        Parameters
        ----------
        x : Series
            Feature
        y : Series, optional
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
        if not isnull(self.value) and notna(self.value):
            return self.value < self.threshold
        return True


class NanMeasure(OutlierMeasure):
    """Measure of the percentage of NaNs"""

    __name__ = "Nan"
    is_x_quantitative = True
    is_x_qualitative = True

    def compute_association(self, x: Series, y: Series = None) -> float:
        """Computes frequency of ``nan`` in ``x``

        Parameters
        ----------
        x : Series
            Feature
        y : Series, optional
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

    def compute_association(self, x: Series, y: Series = None) -> float:
        """Computes frequency of ``x``'s mode

        Parameters
        ----------
        x : Series
            Feature
        y : Series, optional
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
