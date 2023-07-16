""" Measures of association between a Quantitative feature and binary target.
"""

from math import sqrt
from typing import Any

from numpy import nan
from pandas import DataFrame, Series
from scipy.stats import kruskal
from statsmodels.formula.api import ols


def kruskal_measure(
    active: bool, association: dict[str, Any], x: Series, y: Series, **params
) -> tuple[bool, dict[str, Any]]:
    """Kruskal-Wallis statistic between x (quantitative) and y (binary)"""

    # check that previous steps where passed
    if active:
        nans = x.isnull()  # ckecking for nans

        # computation of Kruskal-Wallis statistic
        kw = kruskal(x[(~nans) & (y == 0)], x[(~nans) & (y == 1)])

        # updating association
        if kw:
            association.update({"kruskal_measure": kw[0]})

    return active, association


def R_measure(
    active: bool, association: dict[str, Any], x: Series, y: Series, **params
) -> tuple[bool, dict[str, Any]]:
    """R of the linear regression of x (quantitative) by y (binary)"""

    # check that previous steps where passed
    if active:
        nans = x.isnull()  # ckecking for nans

        # grouping feature and target
        ols_df = DataFrame({"feature": x[~nans], "target": y[~nans]})

        # fitting regression of feature by target
        regression = ols("feature~C(target)", ols_df).fit()

        # updating association
        if regression.rsquared:
            if regression.rsquared >= 0:
                association.update({"R_measure": sqrt(regression.rsquared)})
            else:
                association.update({"R_measure": nan})

    return active, association


def zscore_measure(
    active: bool,
    association: dict[str, Any],
    x: Series,
    y: Series = None,
    **params,
) -> tuple[bool, dict[str, Any]]:
    """Computes outliers based on the z-score

    Parameters
    ----------
    thresh_outlier, float: default 1.
      Maximum percentage of Outliers in a feature
    """

    # check that previous steps where passed for computational optimization
    if active:
        mean = x.mean()  # mean of the feature
        std = x.std()  # standard deviation of the feature
        zscore = (x - mean) / std  # zscore per observation

        # checking for outliers
        outliers = abs(zscore) > 3
        pct_zscore = outliers.mean()

        # updating association
        association.update(
            {
                "pct_zscore": pct_zscore,
                "min": x.min(),
                "max": x.max(),
                "mean": mean,
                "std": std,
            }
        )

        # Excluding feature with too frequent modes
        active = pct_zscore < params.get("thresh_outlier", 1.0)

    return active, association


def iqr_measure(
    active: bool,
    association: dict[str, Any],
    x: Series,
    y: Series = None,
    **params,
) -> tuple[bool, dict[str, Any]]:
    """Computes outliers based on the inter-quartile range

    Parameters
    ----------
    thresh_outlier, float: default 1.
      Maximum percentage of Outliers in a feature
    """

    # check that previous steps where passed for computational optimization
    if active:
        q3 = x.quantile(0.75)  # 3rd quartile
        q1 = x.quantile(0.25)  # 1st quartile
        iqr = q3 - q1  # inter quartile range
        iqr_bounds = q1 - 1.5 * iqr, q3 + 1.5 * iqr  # bounds of the iqr range

        # checking for outliers
        outliers = ~x.between(*iqr_bounds)
        pct_iqr = outliers.mean()

        # updating association
        association.update({"pct_iqr": pct_iqr, "q1": q1, "median": x.median(), "q3": q3})

        # Excluding feature with too frequent modes
        active = pct_iqr < params.get("thresh_outlier", 1.0)

    return active, association
