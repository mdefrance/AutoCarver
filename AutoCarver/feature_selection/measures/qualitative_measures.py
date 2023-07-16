""" Measures of association between a Qualitative feature and binary target.
"""

from math import sqrt
from typing import Any

from pandas import Series, crosstab, notna
from scipy.stats import chi2_contingency


def chi2_measure(
    active: bool, association: dict[str, Any], x: Series, y: Series, **params
) -> tuple[bool, dict[str, Any]]:
    """Chi2 Measure between two Series of qualitative features"""

    # check that previous steps where passed
    if active:
        # computing crosstab between x and y
        xtab = crosstab(x, y)

        # Chi2 statistic
        chi2 = chi2_contingency(xtab)[0]

        # updating association
        association.update({"chi2_measure": chi2})

    return active, association


def cramerv_measure(
    active: bool, association: dict[str, Any], x: Series, y: Series, **params
) -> tuple[bool, dict[str, Any]]:
    """Carmer's V between two Series of qualitative features"""

    # check that previous steps where passed
    if active:
        # computing chi2
        if "chi2_measure" not in association:
            active, association = chi2_measure(active, association, x, y, **params)

        # numnber of observations
        n_obs = (notna(x) & notna(y)).sum()

        # number of values taken by the features
        n_mod_x, n_mod_y = x.nunique(), y.nunique()
        min_n_mod = min(n_mod_x, n_mod_y)

        # Chi2 statistic
        chi2 = association.get("chi2_measure")

        # Cramer's V
        cramerv = sqrt(chi2 / n_obs / (min_n_mod - 1))

        # updating association
        association.update({"cramerv_measure": cramerv})

    return active, association


def tschuprowt_measure(
    active: bool, association: dict[str, Any], x: Series, y: Series, **params
) -> tuple[bool, dict[str, Any]]:
    """Tschuprow's T between two Series of qualitative features"""

    # check that previous steps where passed
    if active:
        # computing chi2
        if "chi2_measure" not in association:
            active, association = chi2_measure(active, association, x, y, **params)

        # numnber of observations
        n_obs = (notna(x) & notna(y)).sum()

        # number of values taken by the features
        n_mod_x, n_mod_y = x.nunique(), y.nunique()

        # Chi2 statistic
        chi2 = association.get("chi2_measure")

        # Tschuprow's T
        dof_mods = sqrt((n_mod_x - 1) * (n_mod_y - 1))
        tschuprowt = 0
        if dof_mods > 0:
            tschuprowt = sqrt(chi2 / n_obs / dof_mods)

        # updating association
        association.update({"tschuprowt_measure": tschuprowt})

    return active, association
