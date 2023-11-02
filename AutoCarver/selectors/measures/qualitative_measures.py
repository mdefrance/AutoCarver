""" Measures of association between a Qualitative feature and binary target.
"""

from math import sqrt
from typing import Any

from pandas import Series, crosstab, notna
from scipy.stats import chi2_contingency


def chi2_measure(
    x: Series,
    y: Series,
    thresh_chi2: float = 0,
    **kwargs,
) -> tuple[bool, dict[str, Any]]:
    """Wrapper for `scipy.stats.chi2_contingency <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html>`_.
    Computes Chi2 statistic on the ``x`` by ``y`` `pandas.crosstab <https://pandas.pydata.org/docs/reference/api/pandas.crosstab.html>`_.

    Parameters
    ----------
    x : Series
        Feature to measure
    y : Series
        Binary target feature
    thresh_chi2 : float, optional
        Minimum Chi2 association, by default ``0``

    Returns
    -------
    tuple[bool, dict[str, Any]]
        Whether ``x`` is sufficiently associated to ``y`` and Pearson's chi2 between ``x`` and ``y``.
    """
    # computing crosstab between x and y
    xtab = crosstab(x, y)

    # Chi2 statistic
    chi2 = chi2_contingency(xtab)[0]

    # updating association
    measurement = {"chi2_statistic": chi2}

    # Excluding features not associated enough
    active = chi2 < thresh_chi2

    return active, measurement


def cramerv_measure(
    x: Series,
    y: Series,
    thresh_cramerv: float = 0,
    chi2_statistic: float = None,
    **kwargs,
) -> tuple[bool, dict[str, Any]]:
    """Computes Carmér's V between ``x`` and ``y`` from ``chi2_measure``.

    Parameters
    ----------
    x : Series
        Feature to measure
    y : Series
        Binary target feature
    thresh_cramerv : float, optional
        Minimum Cramér's V association, by default ``0``
    chi2_statistic : float, optional
        Pearson's chi2 between ``x`` and ``y``, by default ``None``

    Returns
    -------
    tuple[bool, dict[str, Any]]
        Whether ``x`` is sufficiently associated to ``y`` and Carmér's V between ``x`` and ``y``.
    """
    # Chi2 statistic
    if chi2_statistic is None:
        _, measurement = chi2_measure(x, y, **kwargs)
        chi2_statistic = measurement.get("chi2_statistic")

    # number of observations
    n_obs = (notna(x) & notna(y)).sum()

    # number of values taken by the features
    n_mod_x, n_mod_y = x.nunique(), y.nunique()
    min_n_mod = min(n_mod_x, n_mod_y)

    # Cramér's V
    cramerv = sqrt(chi2_statistic / n_obs / (min_n_mod - 1))

    # updating association
    measurement.update({"cramerv_measure": cramerv})

    # Excluding features not associated enough
    active = cramerv < thresh_cramerv

    return active, measurement


def tschuprowt_measure(
    x: Series,
    y: Series,
    thresh_tschuprowt: float = 0,
    chi2_statistic: float = None,
    **kwargs,
) -> tuple[bool, dict[str, Any]]:
    """Computes Tschuprow's T between ``x`` and ``y`` from ``chi2_measure``.

    Parameters
    ----------
    x : Series
        Feature to measure
    y : Series
        Binary target feature
    thresh_tschuprowt : float, optional
        Minimum Tschuprow's T association, by default ``0``
    chi2_statistic : float, optional
        Pearson's chi2 between ``x`` and ``y``, by default ``None``

    Returns
    -------
    tuple[bool, dict[str, Any]]
        Whether ``x`` is sufficiently associated to ``y`` and Tschuprow's T between ``x`` and ``y``.
    """
    # Chi2 statistic
    if chi2_statistic is None:
        _, measurement = chi2_measure(x, y, **kwargs)
        chi2_statistic = measurement.get("chi2_statistic")

    # number of observations
    n_obs = (notna(x) & notna(y)).sum()

    # number of values taken by the features
    n_mod_x, n_mod_y = x.nunique(), y.nunique()

    # Tschuprow's T
    dof_mods = sqrt((n_mod_x - 1) * (n_mod_y - 1))
    tschuprowt = 0
    if dof_mods > 0:
        tschuprowt = sqrt(chi2_statistic / n_obs / dof_mods)

    # updating association
    measurement.update({"tschuprowt_measure": tschuprowt})

    # Excluding features not associated enough
    active = tschuprowt < thresh_tschuprowt

    return active, measurement
