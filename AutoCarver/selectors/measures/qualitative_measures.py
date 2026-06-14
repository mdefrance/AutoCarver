"""Measures of association between a Qualitative feature and binary target."""

from math import sqrt

import pandas as pd
from scipy.stats import chi2_contingency

from AutoCarver.selectors.measures._vectorized import chi2_all
from AutoCarver.selectors.measures.base_measures import BaseMeasure
from AutoCarver.utils import extend_docstring

# X continue y continue distance correlation
# X binaire y continue kruskal y, x
# X multiclass y continue kruskal y, x

# X continue y binaire kruskal x, y
# X binaire y binaire cramerv/tschurpowt x, y
# X multiclass y binaire cramerv/tschurpowt x, y


# X continue y multiclass kruskal x, y
# X binaire y multiclass cramerv/tschurpowt x, y
# X multiclass y multiclass cramerv/tschurpowt x, y


class Chi2Measure(BaseMeasure):
    """Wrapper for `scipy.stats.chi2_contingency <https://docs.scipy.org/doc/scipy/reference/
    generated/scipy.stats.chi2_contingency.html>`_.
    Computes Chi2 statistic on the ``x`` by ``y`` `pandas.crosstab <https://pandas.pydata.org/docs/
    reference/api/pandas.crosstab.html>`_.
    """

    __name__ = "Chi2"
    is_x_qualitative = True
    is_y_qualitative = True

    @extend_docstring(BaseMeasure.compute_association)
    def compute_association(self, x: pd.Series, y: pd.Series) -> float:
        # computing crosstab between x and y
        xtab = pd.crosstab(x, y)

        # computing Chi2 statistic
        self.value = chi2_contingency(xtab)[0]
        return self.value

    def compute_all(self, X, y, features) -> dict[str, dict]:
        """Vectorized chi²-family statistic over all qualitative ``features``.

        Computes each feature's contingency table with ``bincount`` (instead of
        ``pd.crosstab`` + ``chi2_contingency``); subclasses turn the raw chi²
        into their normalized coefficient via :meth:`_stat`.
        """
        block = X[[feature.version for feature in features]]
        chi2, n_obs, n_mod_x, n_mod_y = chi2_all(block, y)
        return {
            feature.version: self._result(self._stat(chi2[i], n_obs[i], n_mod_x[i], n_mod_y[i]))
            for i, feature in enumerate(features)
        }

    def _stat(self, chi2: float, n_obs: float, n_mod_x: float, n_mod_y: float) -> float:
        """Maps raw chi² to the reported statistic. Base measure reports chi²."""
        return float(chi2)


class CramervMeasure(Chi2Measure):
    """Computes Carmér's V between a Qualitative feature and a binary target."""

    __name__ = "CramervMeasure"

    @extend_docstring(Chi2Measure.compute_association)
    def compute_association(self, x: pd.Series, y: pd.Series) -> float:
        # computing Chi2 if not provided
        chi2_value = super().compute_association(x, y)

        # number of non-missing observations
        n_obs = (pd.notna(x) & pd.notna(y)).sum()

        # number of values taken by the features
        n_mod_x, n_mod_y = x.nunique(), y.nunique()
        min_n_mod = min(n_mod_x, n_mod_y)

        # computing Cramér's V
        if min_n_mod > 1:
            self.value = sqrt(chi2_value / n_obs / (min_n_mod - 1))

        return self.value  # type: ignore

    def _stat(self, chi2: float, n_obs: float, n_mod_x: float, n_mod_y: float) -> float:
        min_n_mod = min(n_mod_x, n_mod_y)
        if min_n_mod > 1:
            return sqrt(chi2 / n_obs / (min_n_mod - 1))
        return float(chi2)


class TschuprowtMeasure(Chi2Measure):
    """Computes Tschuprow's T between a Qualitative feature and a binary target."""

    __name__ = "TschuprowtMeasure"

    @extend_docstring(Chi2Measure.compute_association)
    def compute_association(self, x: pd.Series, y: pd.Series) -> float:
        # computing Chi2 if not provided
        chi2_value = super().compute_association(x, y)

        # number of non-missing observations
        n_obs = (pd.notna(x) & pd.notna(y)).sum()

        # number of values taken by the features
        n_mod_x, n_mod_y = x.nunique(), y.nunique()

        # computing Tschuprow's T
        dof_mods = sqrt((n_mod_x - 1) * (n_mod_y - 1))
        self.value = 0
        if dof_mods > 0:
            self.value = sqrt(chi2_value / n_obs / dof_mods)
        return self.value

    def _stat(self, chi2: float, n_obs: float, n_mod_x: float, n_mod_y: float) -> float:
        dof_mods = sqrt((n_mod_x - 1) * (n_mod_y - 1))
        if dof_mods > 0:
            return sqrt(chi2 / n_obs / dof_mods)
        return 0.0
