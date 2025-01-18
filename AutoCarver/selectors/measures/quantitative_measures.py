""" Measures of association between a Quantitative feature and binary target.
"""

from math import sqrt

from numpy import nan
from pandas import DataFrame, Series
from scipy.spatial.distance import correlation
from scipy.stats import kruskal, pearsonr, spearmanr
from statsmodels.formula.api import ols

from .base_measures import AbsoluteMeasure, BaseMeasure, OutlierMeasure


class ReversibleMeasure(BaseMeasure):
    """A reversible measure"""

    def __init__(self, threshold: float = 1) -> None:
        super().__init__(threshold)
        self.reversed = False

    def reverse_xy(self) -> bool:
        """reverses x and y within compute_association method"""
        self.reversed = True
        self.is_x_qualitative = not self.is_x_qualitative
        self.is_x_quantitative = not self.is_x_quantitative
        self.is_y_qualitative = not self.is_y_qualitative
        self.is_y_quantitative = not self.is_y_quantitative
        return True


class KruskalMeasure(ReversibleMeasure):
    """Kruskal-Wallis' test statistic between a Quantitative feature and a Qualitative target."""

    __name__ = "KruskalMeasure"
    is_x_quantitative = True
    is_y_qualitative = True

    def compute_association(self, x: Series, y: Series) -> float:
        """Kruskal-Wallis' test statistic between ``x`` for each value taken by ``y``.

        Parameters
        ----------
        x : Series
            Quantitative feature
        y : Series
            Qualitative target feature
        thresh_kruskal : float, optional
            Minimum Kruskal-Wallis association, by default ``0``

        Returns
        -------
        tuple[bool, dict[str, Any]]
            Whether ``x`` is sufficiently associated to ``y`` and Kruskal-Wallis' H test statistic
        """

        # reversing if requested
        if self.reversed:
            x, y = y, x

        # ckecking for nans
        nans = x.isnull()

        # getting y values
        y_values = y.unique()

        # computing Kruskal-Wallis statistic
        kw = kruskal(*tuple(x[(~nans) & (y == y_value)] for y_value in y_values))
        self.value = kw[0] if kw else nan
        return self.value


class RMeasure(BaseMeasure):
    """Square root of the coefficient of determination of linear regression model of
    a Quantitative feature by a Binary target."""

    __name__ = "RMeasure"
    is_x_quantitative = True
    is_y_qualitative = True
    is_y_binary = True

    def compute_association(self, x: Series, y: Series) -> float:
        """Square root of the coefficient of determination of linear regression model of
        ``x`` by ``y``.

        Parameters
        ----------
        x : Series
            Quantitative feature
        y : Series
            Binary target feature
        thresh_R : float, optional
            Minimum R association, by default ``0``

        Returns
        -------
        tuple[bool, dict[str, Any]]
            Whether ``x`` is sufficiently associated to ``y`` and the square root of the
            coefficient of determination
        """
        # # reversing if requested
        # if self.reversed:
        #     x, y = y, x

        # ckecking for nans
        nans = x.isnull() | x.isna()

        # checking values of y
        if len(y[~nans].unique()) != 2:
            raise ValueError(f"[{self}] Provided y is not binary")

        # grouping feature and target
        ols_df = DataFrame({"feature": x[~nans], "target": y[~nans]})

        # fitting regression of feature by target
        regression = ols("feature~C(target)", ols_df).fit()

        # computing R statistic
        self.value = (
            sqrt(regression.rsquared) if regression.rsquared and regression.rsquared >= 0 else nan
        )

        return self.value


class PearsonMeasure(AbsoluteMeasure):
    """Pearson's linear correlation coefficient between a Quantitative feature and a Quantitative
    target."""

    __name__ = "PearsonMeasure"
    is_x_quantitative = True
    is_y_quantitative = True

    def compute_association(self, x: Series, y: Series) -> float:
        """Pearson's linear correlation coefficient between ``x`` and ``y``.

        Parameters
        ----------
        x : Series
            Quantitative feature
        y : Series
            Quantitative target feature
        thresh_pearson : float, optional
            Minimum r association, by default ``0``

        Returns
        -------
        tuple[bool, dict[str, Any]]
            Whether ``x`` is sufficiently associated to ``y`` and Pearson's r
        """
        # ckecking for nans
        nans = x.isnull() | x.isna()

        # computing spearman's r
        r = pearsonr(x[~nans], y[~nans])
        self.value = r[0] if r else nan
        return self.value


class SpearmanMeasure(AbsoluteMeasure):
    """Spearman's rank correlation coefficient between a Quantitative feature and a Quantitative
    target."""

    __name__ = "SpearmanMeasure"
    is_x_quantitative = True
    is_y_quantitative = True

    def compute_association(self, x: Series, y: Series) -> float:
        """Spearman's rank correlation coefficient between ``x`` and ``y``.

        Parameters
        ----------
        x : Series
            Quantitative feature
        y : Series
            Quantitative target feature
        thresh_spearman : float, optional
            Minimum rho association, by default ``0``

        Returns
        -------
        tuple[bool, dict[str, Any]]
            Whether ``x`` is sufficiently associated to ``y`` and Spearman's rho
        """
        # ckecking for nans
        nans = x.isnull() | x.isna()
        # computing spearman's rho
        rho = spearmanr(x[~nans], y[~nans])
        self.value = rho[0] if rho else nan
        return self.value


class DistanceMeasure(AbsoluteMeasure):
    """Distance correlation between a Quantitative feature and a Quantitative target."""

    __name__ = "DistanceMeasure"
    is_x_quantitative = True
    is_y_quantitative = True

    def compute_association(self, x: Series, y: Series) -> float:
        """Distance correlation between ``x`` and ``y``.

        Parameters
        ----------
        x : Series
            Quantitative feature
        y : Series
            Quantitative target feature
        thresh_distance : float, optional
            Minimum distance association, by default ``0``

        Returns
        -------
        tuple[bool, dict[str, Any]]
            Whether ``x`` is sufficiently associated to ``y`` and Distance Correlation
        """
        # ckecking for nans
        nans = x.isnull()

        # computing distance correlation
        self.value = correlation(x[~nans], y[~nans])
        return self.value


class ZscoreOutlierMeasure(OutlierMeasure):
    """Z-score based outlier measure"""

    __name__ = "ZScore"

    def compute_association(self, x: Series, y: Series = None) -> float:
        """Computes outliers percentage based on the z-score

        Parameters
        ----------
        x : Series
            Quantitative feature
        y : Series, optional
            Any target feature, by default ``None``
        thresh_zscore : float, optional
            Maximum percentage of Outliers in a feature, by default ``1.0``

        Returns
        -------
        tuple[bool, dict[str, Any]]
            Whether or not there are too many outliers and the outlier measurement
        """

        mean = x.mean()  # mean of the feature
        std = x.std()  # standard deviation of the feature
        zscore = (x - mean) / std  # zscore per observation

        # computing outlier rate
        outliers = abs(zscore) > 3
        self.value = outliers.mean()

        # keeping additional info
        self.info.update({"min": x.min(), "max": x.max(), "mean": mean, "std": std})

        return self.value


class IqrOutlierMeasure(OutlierMeasure):
    """Interquartile range based outlier measure"""

    __name__ = "IQR"

    def compute_association(self, x: Series, y: Series = None) -> float:
        """Computes outliers percentage based on the interquartile range

        Parameters
        ----------
        x : Series
            Quantitative feature
        y : Series, optional
            Any target feature, by default ``None``
        thresh_iqr : float, optional
            Maximum percentage of Outliers in a feature, by default ``1.0``

        Returns
        -------
        tuple[bool, dict[str, Any]]
            Whether or not there are too many outliers and the outlier measurement
        """
        q3 = x.quantile(0.75)  # 3rd quartile
        q1 = x.quantile(0.25)  # 1st quartile
        iqr = q3 - q1  # inter quartile range
        iqr_bounds = q1 - 1.5 * iqr, q3 + 1.5 * iqr  # bounds of the iqr range

        # computing outlier rate
        outliers = ~x.between(*iqr_bounds)
        self.value = outliers.mean()

        # keeping additional info
        self.info.update({"q1": q1, "median": x.median(), "q3": q3})

        return self.value
