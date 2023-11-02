""" Measures of association between a Quantitative feature and binary target.
"""

from math import sqrt
from typing import Any

from numpy import nan
from pandas import DataFrame, Series
from scipy.stats import kruskal
from statsmodels.formula.api import ols


def kruskal_measure(
    x: Series,
    y: Series,
    thresh_kruskal: float = 0,
    **kwargs,
) -> tuple[bool, dict[str, Any]]:
    """Kruskal-Wallis' test statistic between ``x`` when ``y==1`` and ``x`` when ``y==0``.

    Parameters
    ----------
    x : Series
        Feature to measure
    y : Series, optional
        Binary target feature
    thresh_kruskal : float, optional
        Minimum Kruskal-Wallis association, by default ``0``

    Returns
    -------
    tuple[bool, dict[str, Any]]
        Whether ``x`` is sufficiently associated to ``y`` and Kruskal-Wallis' test statistic
    """
    nans = x.isnull()  # ckecking for nans

    # computation of Kruskal-Wallis statistic
    kw = kruskal(x[(~nans) & (y == 0)], x[(~nans) & (y == 1)])

    # updating association
    active, measurement = False, {"kruskal_measure": nan}
    if kw:
        measurement = {"kruskal_measure": kw[0]}

        # Excluding features not associated enough
        active = kw[0] < thresh_kruskal

    return active, measurement


def R_measure(
    x: Series,
    y: Series,
    thresh_R: float = 0,
    **kwargs,
) -> tuple[bool, dict[str, Any]]:
    """Square root of the coefficient of determination of linear regression model of ``x`` by ``y``.

    Parameters
    ----------
    x : Series
        Feature to measure
    y : Series, optional
        Binary target feature
    thresh_R : float, optional
        Minimum R association, by default ``0``

    Returns
    -------
    tuple[bool, dict[str, Any]]
        Whether ``x`` is sufficiently associated to ``y`` and the square root of the determination coefficient
    """
    nans = x.isnull()  # ckecking for nans

    # grouping feature and target
    ols_df = DataFrame({"feature": x[~nans], "target": y[~nans]})

    # fitting regression of feature by target
    regression = ols("feature~C(target)", ols_df).fit()

    # updating association
    active, measurement = False, {"R_measure": nan}
    if regression.rsquared and regression.rsquared >= 0:
        r_measure = sqrt(regression.rsquared)
        measurement = {"R_measure": r_measure}

        # Excluding features not associated enough
        active = r_measure < thresh_R

    return active, measurement


def zscore_measure(
    x: Series,
    y: Series = None,
    thresh_outlier: float = 1.0,
    **kwargs,
) -> tuple[bool, dict[str, Any]]:
    """Computes outliers percentage based on the z-score

    Parameters
    ----------
    x : Series
        Feature to measure
    y : Series, optional
        Binary target feature, by default ``None``
    thresh_outlier : float, optional
        Maximum percentage of Outliers in a feature, by default ``1.0``

    Returns
    -------
    tuple[bool, dict[str, Any]]
        Whether or not there are too many outliers and the outlier measurement
    """
    mean = x.mean()  # mean of the feature
    std = x.std()  # standard deviation of the feature
    zscore = (x - mean) / std  # zscore per observation

    # checking for outliers
    outliers = abs(zscore) > 3
    pct_zscore = outliers.mean()

    # updating association
    measurement = {
        "pct_zscore": pct_zscore,
        "min": x.min(),
        "max": x.max(),
        "mean": mean,
        "std": std,
    }

    # Excluding feature with too frequent modes
    active = pct_zscore < thresh_outlier

    return active, measurement


def iqr_measure(
    x: Series,
    y: Series = None,
    thresh_outlier: float = 1.0,
    **kwargs,
) -> tuple[bool, dict[str, Any]]:
    """Computes outliers percentage based on the interquartile range

    Parameters
    ----------
    x : Series
        Feature to measure
    y : Series, optional
        Binary target feature, by default ``None``
    thresh_outlier : float, optional
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

    # checking for outliers
    outliers = ~x.between(*iqr_bounds)
    pct_iqr = outliers.mean()

    # updating association
    measurement = {"pct_iqr": pct_iqr, "q1": q1, "median": x.median(), "q3": q3}

    # Excluding feature with too frequent modes
    active = pct_iqr < thresh_outlier

    return active, measurement
