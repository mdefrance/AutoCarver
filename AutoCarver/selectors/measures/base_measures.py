""" Base measures that defines Quantitative and Qualitative features.
"""

from pandas import Series, isnull, notna

from abc import ABC, abstractmethod
from ...features import BaseFeature


class BaseMeasure(ABC):

    is_x_quantitative = False
    is_x_qualitative = False

    is_y_qualitative = False
    is_y_quantitative = False
    is_y_binary = False
    is_sortable = True

    # info
    is_default = False
    higher_is_better = True
    correlation_with = "target"
    # absolute_threshold = False

    __name__ = "BaseMeasure"

    def __init__(self, threshold: float = 0.0) -> None:
        self.threshold = threshold
        self.value = None
        self._info = {}

    def __repr__(self) -> str:
        return self.__name__

    @abstractmethod
    def compute_association(self, x: Series, y: Series) -> float:
        pass

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
        """checks if measured correlation is above specified threshold -> keep the feature"""
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

    def update_feature(self, feature: BaseFeature) -> None:
        """adds measure to specified feature"""

        # checking for a value
        if self.value is None:
            raise ValueError(f" - [{self}] Use compute_association first!")

        # existing stats
        measures = feature.statistics.get("measures", {})

        # adding new measure
        measures.update(self.to_dict())

        # updating statistics of the feature accordingly
        feature.statistics.update({"measures": measures})

    def reverse_xy(self) -> bool:
        """reverses values of x and y in compute_association"""

        # when its not implemented
        return False


class AbsoluteMeasure(BaseMeasure):

    # info
    # absolute_threshold = False

    def validate(self) -> bool:
        """checks if measured correlation is above specified threshold -> keep the feature"""
        if not isnull(self.value) and notna(self.value):
            return abs(self.value) >= self.threshold
        return False


class OutlierMeasure(BaseMeasure):
    is_default = True
    is_x_quantitative = True
    is_x_qualitative = False
    is_sortable = False

    # info
    higher_is_better = False
    correlation_with = "itself"

    def __init__(self, threshold: float = 1.0) -> None:
        super().__init__(threshold)

    @abstractmethod
    def compute_association(self, x: Series, y: Series = None) -> float:
        pass

    def validate(self) -> bool:
        """checks if measured outlier rate is below specified threshold -> keep the feature
        (by default keeps feature if outliermeasure is not defined)
        """
        if not isnull(self.value) and notna(self.value):
            return self.value < self.threshold
        return True


class NanMeasure(OutlierMeasure):
    __name__ = "NaN"
    is_x_quantitative = True
    is_x_qualitative = True

    def compute_association(self, x: Series, y: Series = None) -> float:
        """Measure of the percentage of NaNs

        Parameters
        ----------
        x : Series
            Feature to measure
        y : Series, optional
            Binary target feature, by default ``None``
        thresh_nan : float, optional
            Maximum percentage of NaNs in a feature, by default ``0.999``

        Returns
        -------
        tuple[bool, dict[str, Any]]
            Whether or not there are to many NaNs and the percentage of NaNs
        """
        _ = y
        self.value = (x.isna() | x.isnull()).mean()
        return self.value


class ModeMeasure(OutlierMeasure):
    __name__ = "Mode"
    is_x_quantitative = True
    is_x_qualitative = True

    def compute_association(self, x: Series, y: Series = None) -> float:
        """Measure of the percentage of NaNs

        Parameters
        ----------
        x : Series
            Feature to measure
        y : Series, optional
            Binary target feature, by default ``None``
        thresh_nan : float, optional
            Maximum percentage of NaNs in a feature, by default ``0.999``

        Returns
        -------
        tuple[bool, dict[str, Any]]
            Whether or not there are to many NaNs and the percentage of NaNs
        """
        _ = y
        mode = x.mode(dropna=True).values[0]  # computing mode
        self.value = (x == mode).mean()  # Computing percentage of the mode
        return self.value
