""" Measures of association between a Quantitative feature and binary target.
"""

from math import sqrt
from typing import Any

from numpy import nan
from pandas import DataFrame, Series
from scipy.spatial.distance import correlation
from scipy.stats import kruskal, pearsonr, spearmanr
from statsmodels.formula.api import ols


def kruskal_measure(
    x: Series,
    y: Series,
    thresh_kruskal: float = 0,
    **kwargs,
) -> tuple[bool, dict[str, Any]]:
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
    # ckecking for nans
    nans = x.isnull()

    # getting y values
    y_values = y.unique()

    # computation of Kruskal-Wallis statistic
    kw = kruskal(*tuple(x[(~nans) & (y == y_value)] for y_value in y_values))

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
        Quantitative feature
    y : Series
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


def pearson_measure(
    x: Series,
    y: Series,
    thresh_pearson: float = 0,
    **kwargs,
) -> tuple[bool, dict[str, Any]]:
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
    nans = x.isnull()  # ckecking for nans

    # computing spearman's r
    r = pearsonr(x[~nans], y[~nans])

    # updating association
    active, measurement = False, {"pearson_measure": nan}
    if r:
        measurement = {"pearson_measure": r}

        # Excluding features not associated enough
        active = r < thresh_pearson

    return active, measurement


def spearman_measure(
    x: Series,
    y: Series,
    thresh_spearman: float = 0,
    **kwargs,
) -> tuple[bool, dict[str, Any]]:
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
    nans = x.isnull()  # ckecking for nans

    # computing spearman's r
    rho = spearmanr(x[~nans], y[~nans])

    # updating association
    active, measurement = False, {"spearman_measure": nan}
    if rho:
        measurement = {"spearman_measure": rho}

        # Excluding features not associated enough
        active = rho < thresh_spearman

    return active, measurement


def distance_measure(
    x: Series,
    y: Series,
    thresh_distance: float = 0,
    **kwargs,
) -> tuple[bool, dict[str, Any]]:
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
    nans = x.isnull()  # ckecking for nans

    # computing spearman's r
    d_corr = correlation(x[~nans], y[~nans])

    # updating association
    active, measurement = False, {"distance_measure": nan}
    if d_corr:
        measurement = {"distance_measure": d_corr}

        # Excluding features not associated enough
        active = d_corr < thresh_distance

    return active, measurement


def zscore_measure(
    x: Series,
    y: Series = None,
    thresh_zscore: float = 1.0,
    **kwargs,
) -> tuple[bool, dict[str, Any]]:
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
    active = pct_zscore < thresh_zscore

    return active, measurement


def iqr_measure(
    x: Series,
    y: Series = None,
    thresh_iqr: float = 1.0,
    **kwargs,
) -> tuple[bool, dict[str, Any]]:
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

    # checking for outliers
    outliers = ~x.between(*iqr_bounds)
    pct_iqr = outliers.mean()

    # updating association
    measurement = {"pct_iqr": pct_iqr, "q1": q1, "median": x.median(), "q3": q3}

    # Excluding feature with too frequent modes
    active = pct_iqr < thresh_iqr

    return active, measurement
