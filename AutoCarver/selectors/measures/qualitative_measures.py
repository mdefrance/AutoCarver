""" Measures of association between a Qualitative feature and binary target.
"""

from math import sqrt

from pandas import Series, crosstab, notna
from scipy.stats import chi2_contingency

from ...utils import extend_docstring
from .base_measures import BaseMeasure

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
    def compute_association(self, x: Series, y: Series) -> float:
        # computing crosstab between x and y
        xtab = crosstab(x, y)

        # computing Chi2 statistic
        self.value = chi2_contingency(xtab)[0]
        return self.value


class CramervMeasure(Chi2Measure):
    """Computes Carmér's V between a Qualitative feature and a binary target."""

    __name__ = "CramervMeasure"

    @extend_docstring(Chi2Measure.compute_association)
    def compute_association(self, x: Series, y: Series) -> float:
        # computing Chi2 if not provided
        chi2_value = super().compute_association(x, y)

        # number of non-missing observations
        n_obs = (notna(x) & notna(y)).sum()

        # number of values taken by the features
        n_mod_x, n_mod_y = x.nunique(), y.nunique()
        min_n_mod = min(n_mod_x, n_mod_y)

        # computing Cramér's V
        if min_n_mod > 1:
            self.value = sqrt(chi2_value / n_obs / (min_n_mod - 1))

        return self.value


class TschuprowtMeasure(Chi2Measure):
    """Computes Tschuprow's T between a Qualitative feature and a binary target."""

    __name__ = "TschuprowtMeasure"

    @extend_docstring(Chi2Measure.compute_association)
    def compute_association(self, x: Series, y: Series) -> float:
        # computing Chi2 if not provided
        chi2_value = super().compute_association(x, y)

        # number of non-missing observations
        n_obs = (notna(x) & notna(y)).sum()

        # number of values taken by the features
        n_mod_x, n_mod_y = x.nunique(), y.nunique()

        # computing Tschuprow's T
        dof_mods = sqrt((n_mod_x - 1) * (n_mod_y - 1))
        self.value = 0
        if dof_mods > 0:
            self.value = sqrt(chi2_value / n_obs / dof_mods)
        return self.value
