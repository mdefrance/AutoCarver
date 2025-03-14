"""Measures of association between a Quantitative feature and binary target."""

from math import sqrt

from numpy import nan
from pandas import DataFrame, Series
from scipy.spatial.distance import correlation
from scipy.stats import kruskal, pearsonr, spearmanr
from statsmodels.formula.api import ols

from ...utils import extend_docstring
from .base_measures import AbsoluteMeasure, BaseMeasure, OutlierMeasure


class ReversibleMeasure(BaseMeasure):
    """A reversible measure"""

    is_reversible = True

    def __init__(self, threshold: float = 0.0) -> None:
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

    @extend_docstring(BaseMeasure.compute_association)
    def compute_association(self, x: Series, y: Series) -> float:
        # reversing if requested
        if self.reversed:
            x, y = y, x

        # ckecking for nans
        nans = x.isnull() | x.isna()

        # getting y values
        y_values = y.unique()

        # computing Kruskal-Wallis statistic
        self.value = nan
        if not all(nans):
            try:
                kw = kruskal(*tuple(x[(~nans) & (y == y_value)] for y_value in y_values))
                if kw:
                    self.value = kw[0]
            # when all identic values
            except ValueError:
                self.value = nan
        return self.value


class RMeasure(BaseMeasure):
    """Square root of the coefficient of determination of linear regression model of
    a Quantitative feature by a Binary target."""

    __name__ = "RMeasure"
    is_x_quantitative = True
    is_y_qualitative = True
    is_y_binary = True

    @extend_docstring(BaseMeasure.compute_association)
    def compute_association(self, x: Series, y: Series) -> float:
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
    """Pearson's linear correlation coefficient between a Quantitative feature and target."""

    __name__ = "PearsonMeasure"

    is_x_quantitative = True
    is_y_quantitative = True

    @extend_docstring(BaseMeasure.compute_association)
    def compute_association(self, x: Series, y: Series) -> float:
        # ckecking for nans
        nans = x.isnull() | x.isna()

        # computing pearson's r
        self.value = nan
        if not all(nans):
            r = pearsonr(x[~nans], y[~nans])
            if r:
                self.value = abs(r[0])
        return self.value


class SpearmanMeasure(AbsoluteMeasure):
    """Spearman's rank correlation coefficient between a Quantitative feature and target."""

    __name__ = "SpearmanMeasure"
    is_x_quantitative = True
    is_y_quantitative = True

    @extend_docstring(BaseMeasure.compute_association)
    def compute_association(self, x: Series, y: Series) -> float:
        # ckecking for nans
        nans = x.isnull() | x.isna()
        # computing spearman's rho
        self.value = nan
        if not all(nans):
            rho = spearmanr(x[~nans], y[~nans])
            if rho:
                self.value = abs(rho[0])
        return self.value


class DistanceMeasure(AbsoluteMeasure):
    """Distance correlation between a Quantitative feature and target."""

    __name__ = "DistanceMeasure"
    is_x_quantitative = True
    is_y_quantitative = True

    @extend_docstring(BaseMeasure.compute_association)
    def compute_association(self, x: Series, y: Series) -> float:
        # ckecking for nans
        nans = x.isnull()

        # computing distance correlation
        self.value = nan
        if not all(nans):
            self.value = correlation(x[~nans], y[~nans])
        return self.value


class ZscoreOutlierMeasure(OutlierMeasure):
    """Z-Score based outlier measure"""

    __name__ = "ZScore"

    @extend_docstring(OutlierMeasure.compute_association)
    def compute_association(self, x: Series, y: Series = None) -> float:
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

    @extend_docstring(OutlierMeasure.compute_association)
    def compute_association(self, x: Series, y: Series = None) -> float:
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
