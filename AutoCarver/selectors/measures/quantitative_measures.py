"""Measures of association between a Quantitative feature and binary target."""

import warnings
from math import sqrt

import numpy as np
import pandas as pd
from scipy.spatial.distance import correlation
from scipy.stats import kruskal, pearsonr, spearmanr
from statsmodels.formula.api import ols

from AutoCarver.selectors.measures._vectorized import kruskal_h, kruskal_h_reversed
from AutoCarver.selectors.measures.base_measures import AbsoluteMeasure, BaseMeasure, OutlierMeasure
from AutoCarver.utils import extend_docstring


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
    def compute_association(self, x: pd.Series, y: pd.Series) -> float:
        # reversing if requested
        if self.reversed:
            x, y = y, x

        # excluding rows missing in either x or the grouping y (in the reversed
        # case y is the qualitative feature and may carry NaN; dropping it here
        # avoids an empty NaN group that would void the whole statistic)
        nans = x.isnull() | x.isna() | y.isnull() | y.isna()

        # getting y values (NaN already excluded via the mask above)
        y_values = y[~nans].unique()

        # computing Kruskal-Wallis statistic
        self.value = np.nan
        if has_values(x, y, nans):
            try:
                kw = kruskal(*tuple(x[(~nans) & (y == y_value)] for y_value in y_values))
                if kw:
                    self.value = kw[0]
            # when all identic values
            except ValueError:
                pass
        return self.value

    def compute_all(self, X, y, features) -> dict[str, dict]:
        """Vectorized Kruskal-Wallis over all ``features`` at once.

        Picks the reversed kernel (each qualitative column defines the groups,
        continuous ``y`` is ranked) when :attr:`reversed` is set, else the
        shared-grouping kernel (quantitative columns ranked, ``y`` groups).
        Subclasses map the raw ``H`` to their effect size via :meth:`_effect`.
        """
        block = X[[feature.version for feature in features]]
        if self.reversed:
            h, n_obs, n_groups = kruskal_h_reversed(block, y)
        else:
            h, n_obs, n_groups = kruskal_h(block, y)
        return {
            feature.version: self._result(self._effect(h[i], n_obs[i], n_groups[i]))
            for i, feature in enumerate(features)
        }

    def _effect(self, h: float, n_obs: float, n_groups: float) -> float:
        """Maps raw ``H`` to the reported statistic. Base measure reports ``H``."""
        return float(h)


class KruskalEpsilonSquaredMeasure(KruskalMeasure):
    """Epsilon-squared effect size derived from Kruskal-Wallis' H statistic.

    Unlike the raw H statistic (which grows with the number of observations,
    making it unsuitable for comparing features of differing sample sizes),
    :math:`\\varepsilon^2 = H / (N - 1)` is bounded in :math:`[0, 1]`. It is to
    Kruskal-Wallis what Cramér's V is to Chi2 — a sample-size-normalized effect
    size meant for cross-feature ranking.
    """

    __name__ = "KruskalEpsilonSquaredMeasure"

    @extend_docstring(KruskalMeasure.compute_association)
    def compute_association(self, x: pd.Series, y: pd.Series) -> float:
        # computing Kruskal-Wallis' H (handles the reverse swap internally)
        h = super().compute_association(x, y)

        # mirroring the reverse swap to count the pooled observations
        if self.reversed:
            x, y = y, x
        n_obs = ((~(x.isnull() | x.isna())) & y.notna()).sum()

        # computing epsilon-squared
        self.value = np.nan
        if pd.notna(h) and n_obs > 1:
            self.value = float(h / (n_obs - 1))
        return self.value

    def _effect(self, h: float, n_obs: float, n_groups: float) -> float:
        if np.isnan(h) or n_obs <= 1:
            return np.nan
        return float(h / (n_obs - 1))


class KruskalEtaSquaredMeasure(KruskalMeasure):
    """Eta-squared effect size derived from Kruskal-Wallis' H statistic.

    :math:`\\eta^2 = (H - k + 1) / (N - k)`, where ``k`` is the number of groups
    and ``N`` the number of pooled observations. Like
    :class:`KruskalEpsilonSquaredMeasure` it removes the sample-size inflation of the
    raw H statistic, but it additionally corrects for ``k`` — useful in the
    reversed (regression) case where ``k`` is the feature's modality count and
    therefore varies across features. Clamped to :math:`[0, 1]`.
    """

    __name__ = "KruskalEtaSquaredMeasure"

    @extend_docstring(KruskalMeasure.compute_association)
    def compute_association(self, x: pd.Series, y: pd.Series) -> float:
        # computing Kruskal-Wallis' H (handles the reverse swap internally)
        h = super().compute_association(x, y)

        # mirroring the reverse swap to count observations and groups
        if self.reversed:
            x, y = y, x
        valid = (~(x.isnull() | x.isna())) & y.notna()
        n_obs = int(valid.sum())
        n_groups = y[valid].nunique()

        # computing eta-squared, clamped to a non-negative effect size
        self.value = np.nan
        if pd.notna(h) and n_obs - n_groups > 0:
            self.value = max(0.0, float((h - n_groups + 1) / (n_obs - n_groups)))
        return self.value

    def _effect(self, h: float, n_obs: float, n_groups: float) -> float:
        if np.isnan(h) or (n_obs - n_groups) <= 0:
            return np.nan
        return max(0.0, float((h - n_groups + 1) / (n_obs - n_groups)))


class RMeasure(BaseMeasure):
    """Square root of the coefficient of determination of linear regression model of
    a Quantitative feature by a Binary target."""

    __name__ = "RMeasure"
    is_x_quantitative = True
    is_y_qualitative = True
    is_y_binary = True

    @extend_docstring(BaseMeasure.compute_association)
    def compute_association(self, x: pd.Series, y: pd.Series) -> float:
        # # reversing if requested
        # if self.reversed:
        #     x, y = y, x

        # ckecking for nans
        nans = x.isnull() | x.isna()

        # checking values of y
        if y[~nans].nunique() != 2:
            raise ValueError(f"[{self}] Provided y is not binary")

        # a constant feature has no explained variance (centered_tss == 0): R is
        # undefined, return NaN before fitting to avoid a divide-by-zero in statsmodels
        if x[~nans].nunique() <= 1:
            self.value = np.nan
            return self.value

        # grouping feature and target
        ols_df = pd.DataFrame({"feature": x[~nans], "target": y[~nans]})

        # fitting regression of feature by target
        regression = ols("feature~C(target)", ols_df).fit()

        # computing R statistic
        self.value = sqrt(regression.rsquared) if regression.rsquared and regression.rsquared >= 0 else np.nan

        return self.value


def has_values(x: pd.Series, y: pd.Series, nans: pd.Series) -> bool:
    """Checks if x and y have values"""
    # only nan values
    if all(nans):
        return False
    # only one unique value
    if x[~nans].nunique() <= 1 or y[~nans].nunique() <= 1:
        return False
    return True


class CorrelationMeasure(AbsoluteMeasure):
    """Base for measures that batch via ``DataFrame.corrwith`` (Spearman/Pearson)."""

    _corr_method = "pearson"

    def compute_all(self, X, y, features) -> dict[str, dict]:
        """Vectorized correlation of every feature with ``y`` in one ``corrwith`` call."""
        block = X[[feature.version for feature in features]]
        # degenerate columns (constant, or <=1 non-NaN overlap) legitimately yield
        # NaN; silence the numpy/scipy RuntimeWarnings (divide, df<=0, ConstantInput)
        # rather than surface them to end users
        with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            corr = block.corrwith(y, method=self._corr_method)
        return {feature.version: self._result(corr[feature.version]) for feature in features}


class PearsonMeasure(CorrelationMeasure):
    """Pearson's linear correlation coefficient between a Quantitative feature and target."""

    __name__ = "PearsonMeasure"

    is_x_quantitative = True
    is_y_quantitative = True
    _corr_method = "pearson"

    @extend_docstring(BaseMeasure.compute_association)
    def compute_association(self, x: pd.Series, y: pd.Series) -> float:
        # ckecking for nans
        nans = x.isnull() | x.isna()

        # computing pearson's r
        self.value = np.nan
        if has_values(x, y, nans):
            r = pearsonr(x[~nans], y[~nans])
            if r:
                self.value = r[0]
        return self.value


class SpearmanMeasure(CorrelationMeasure):
    """Spearman's rank correlation coefficient between a Quantitative feature and target."""

    __name__ = "SpearmanMeasure"
    is_x_quantitative = True
    is_y_quantitative = True
    _corr_method = "spearman"

    @extend_docstring(BaseMeasure.compute_association)
    def compute_association(self, x: pd.Series, y: pd.Series) -> float:
        # ckecking for nans
        nans = x.isnull() | x.isna()
        # computing spearman's rho
        self.value = np.nan
        if has_values(x, y, nans):
            rho = spearmanr(x[~nans], y[~nans])
            if rho:
                self.value = rho[0]
        return self.value


class DistanceMeasure(AbsoluteMeasure):
    """Distance correlation between a Quantitative feature and target."""

    __name__ = "DistanceMeasure"
    is_x_quantitative = True
    is_y_quantitative = True

    @extend_docstring(BaseMeasure.compute_association)
    def compute_association(self, x: pd.Series, y: pd.Series) -> float:
        # ckecking for nans
        nans = x.isnull()

        # computing distance correlation
        self.value = np.nan
        if has_values(x, y, nans):
            self.value = correlation(x[~nans], y[~nans]) - 1
        return self.value


class ZscoreOutlierMeasure(OutlierMeasure):
    """Z-Score based outlier measure"""

    __name__ = "ZScore"

    @extend_docstring(OutlierMeasure.compute_association)
    def compute_association(self, x: pd.Series, y: pd.Series | None = None) -> float:
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
    def compute_association(self, x: pd.Series, y: pd.Series | None = None) -> float:
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
