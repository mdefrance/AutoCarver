""" Base measures that defines Quantitative and Qualitative features.
"""

from typing import Any, Callable

from pandas import Series, isnull, notna

from abc import ABC, abstractmethod
from ...features import BaseFeature


class BaseMeasure(ABC):

    is_x_quantitative = False
    is_x_qualitative = False

    is_y_qualitative = False
    is_y_quantitative = False
    is_y_binary = False

    # info
    is_default = False
    higher_is_better = True
    correlation_with = "target"
    # absolute_threshold = False

    __name__ = "BaseMeasure"

    def __init__(self, threshold: float = 1.0) -> None:
        self.threshold = threshold
        self.value = None
        self._info = {}

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
            raise ValueError(f" - [{self.__name__}] Use compute_association first!")

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
    is_x_quantitative = True

    # info
    higher_is_better = False
    correlation_with = "itself"
    is_default = True

    def __init__(self, threshold: float = 0.0) -> None:
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


def reverse_xy(measure: Callable):
    """Reverses places of x and y in measure"""

    def reversed_measure(
        x: Series,
        y: Series,
        **kwargs,
    ) -> tuple[bool, dict[str, Any]]:
        """Reversed version of the measure"""
        return measure(y, x, **kwargs)

    # setting name of passed measure
    reversed_measure.__name__ = measure.__name__

    return reversed_measure


def nans_measure(
    x: Series,
    y: Series = None,
    thresh_nan: float = 0.999,
    **kwargs,
) -> tuple[bool, dict[str, Any]]:
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
    _, _ = y, kwargs  # unused attributes

    nans = x.isnull()  # ckecking for nans
    pct_nan = nans.mean()  # Computing percentage of nans

    # updating association
    measurement = {"pct_nan": pct_nan}

    # Excluding feature that have to many NaNs
    active = pct_nan < thresh_nan
    if not active:
        print(
            f"Feature {x.name} will be discarded (more than {thresh_nan:2.2%} of nans). Otherwise,"
            " set a greater value for thresh_nan."
        )

    return active, measurement


def dtype_measure(
    x: Series,
    y: Series = None,
    **kwargs,
) -> tuple[bool, dict[str, Any]]:
    """Feature's dtype

    Parameters
    ----------
    x : Series
        Feature to measure
    y : Series, optional
        Binary target feature, by default ``None``

    Returns
    -------
    tuple[bool, dict[str, Any]]
        True and the feature's dtype
    """
    _, _ = y, kwargs  # unused attributes

    # getting dtype
    measurement = {"dtype": x.dtype}

    return True, measurement


def mode_measure(
    x: Series,
    y: Series = None,
    thresh_mode: float = 0.999,
    **kwargs,
) -> tuple[bool, dict[str, Any]]:
    """Measure of the percentage of the Mode

    Parameters
    ----------
    x : Series
        Feature to measure
    y : Series, optional
        Binary target feature, by default ``None``
    thresh_mode : float, optional
        Maximum percentage of a feature's mode, by default ``0.999``

    Returns
    -------
    tuple[bool, dict[str, Any]]
        Whether or not the mode is overrepresented and the percentage of mode
    """
    _, _ = y, kwargs  # unused attributes

    mode = x.mode(dropna=True).values[0]  # computing mode
    pct_mode = (x == mode).mean()  # Computing percentage of the mode

    # updating association
    measurement = {"pct_mode": pct_mode, "mode": mode}

    # Excluding feature with too frequent modes
    active = pct_mode < thresh_mode
    if not active:
        print(
            f"Feature {x.name} will be discarded (more than {thresh_mode:2.2%} of its mode). "
            "Otherwise, set a greater value for thresh_mode."
        )

    return active, measurement
