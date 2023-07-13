"""Tools to select the best Quantitative and Qualitative features."""

from math import sqrt
from random import shuffle
from typing import Any, Callable, Dict, Tuple

from IPython.display import display_html
from numpy import inf, nan, ones, triu
from pandas import DataFrame, Series, crosstab, notna
from scipy.stats import chi2_contingency, kruskal
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .discretizers.utils.base_discretizers import GroupedList, GroupedListDiscretizer

# TODO: convert to groupedlistdiscretizer
# TODO: add parameter to shut down displayed info
class FeatureSelector(BaseEstimator, TransformerMixin):
    """A pipeline of measures to perform EDA and feature pre-selection

     - best features are the n_best of each measure
     - selected features are stored in FeatureSelector.best_features

    Parameters
    ----------
    features: list[str]
        Features on which to compute association.
    n_best, int:
        Number of features to be selected
    sample_size: float, default 1.
        Should be set between ]0, 1]
        Size of sampled list of features speeds up computation.
        By default, all features are used. For sample_size=0.5,
        FeatureSelector will search for the best features in
        features[:len(features)//2] and then in features[len(features)//2:]
    measures, list[Callable]: default list().
        List of association measures to be used.
        Implemented measures are:
            [Quantitative Features]
             - For association evaluation: `kruskal_measure`, `R_measure`
             - For outlier detection: `zscore_measure`, `iqr_measure`
            [Qualitative Features]
             - For correlation: `chi2_measure`, `cramerv_measure`, `tschuprowt_measure`
        Ranks features based on last measure of the list.
    filters, list[Callable]: default list().
        List of filters to be used.
        Implemented filters are:
            [Quantitative Features]
             - For linear correlation: `spearman_filter`, `pearson_filter`
             - For multicoloinearity: `vif_filter`
            [Qualitative Features]
             - For correlation: `cramerv_filter`, `tschuprowt_filter`

    Thresholds (to be passed as kwargs)
    ----------
    thresh_measure, float: default 0.
        Minimum association between target and features
        To be used with: `measure_filter`
    name_measure, str
        Measure to be used for minimum association filtering
        To be used with: `measure_filter`
    thresh_nan, float: default 1.
        Maximum percentage of NaNs in a feature
        To be used with: `nans_measure`
    thresh_mode, float: default 1.
        Maximum percentage of the mode of a feature
        To be used with: `mode_measure`
    thresh_outlier, float: default 1.
        Maximum percentage of Outliers in a feature
        To be used with: `iqr_measure`, `zscore_measure`
    thresh_corr, float: default 1.
        Maximum association between features
        To be used with: `spearman_filter`, `pearson_filter`, `cramerv_filter`, `tschuprowt_filter`
    thresh_vif, float: default inf
        Maximum VIF between features
        To be used with: `vif_filter`
    ascending, bool default False
        According to this measure:
         - True: Lower values of the measure are to be considered as more associated to the target
         - False: Higher values of the measure are to be considered as more associated to the target
    """

    def __init__(
        self,
        features: list[str],
        n_best: int,
        measures: list[Callable],
        *,
        filters: list[Callable] = None,
        sample_size: float = 1.0,
        values_orders: dict[str, GroupedList] = None,
        copy: bool = True,
        drop: bool = False,  # TODO
        verbose: bool = True,
        **params,
    ) -> None:
        """_summary_

        Parameters
        ----------
        features : list[str]
            _description_
        n_best : int
            _description_
        measures : list[Callable], optional
            _description_, by default list()
        filters : list[Callable], optional
            _description_, by default list()
        sample_size : float, optional
            _description_, by default 1.0
        copy : bool, optional
            _description_, by default True
        verbose : bool, optional
            _description_, by default True
        """
        print("Warning: not fully optimized for package versions greater than 4.")

        self.features = list(set(features))
        self.n_best = n_best
        assert n_best <= len(features), "Must set n_best <= len(features)"
        self.best_features = features[:]
        self.sample_size = sample_size

        self.measures = [dtype_measure, nans_measure, mode_measure] + measures[:]
        if filters is None:
            filters = []
        self.filters = [thresh_filter] + filters[:]
        self.sort_measures = [measure.__name__ for measure in measures[::-1]]

        # Values_orders from GroupedListDiscretizer
        if values_orders is None:
            values_orders = {}
        self.values_orders = {feature: GroupedList(value) for feature, value in values_orders.items()}

        self.drop = drop
        self.copy = copy
        self.verbose = verbose
        self.params = params

        self.associations = None
        self.filtered_associations = None

    def measure(self, x: Series, y: Series) -> Dict[str, Any]:
        """Measures association between x and y"""

        passed = True  # measures keep going only if previous basic tests are passed
        association = {}

        # iterating over each measure
        for measure in self.measures:
            passed, association = measure(passed, association, x, y, **self.params)

        return association

    def measure_apply(self, X: DataFrame, y: Series, features: list[str]) -> None:
        """Measures association between columns of X and y

        Parameters
        ----------
        ascending, bool default False
            According to this measure:
             - True: Lower values of the measure are to be considered as more associated to the target
             - False: Higher values of the measure are to be considered as more associated to the target
        """

        # applying association measure to each column
        self.associations = X[features].apply(self.measure, y=y, result_type="expand", axis=0).T

        # filtering non association measure (pct_zscore, pct_iqr...)
        asso_measures = [c for c in self.associations if "_measure" in c]
        self.sort_measures = [c for c in self.sort_measures if c in asso_measures]

        # sorting statistics if an association measure was provided
        self.associations = self.associations.sort_values(
            self.sort_measures, ascending=self.params.get("ascending", False)
        )

    def filter_apply(self, X: DataFrame, sort_measure: str) -> DataFrame:
        """Filters out too correlated features (least relevant first)

        Parameters
        ----------
        ascending, bool default False
            According to this measure:
             - True: Lower values of the measure are to be considered as more associated to the target
             - False: Higher values of the measure are to be considered as more associated to the target
        """

        # ordering features by sort_by
        self.filtered_associations = self.associations.sort_values(
            sort_measure, ascending=self.params.get("ascending", False)
        )

        # applying successive filters
        for filtering in self.filters:
            # ordered filtering
            self.filtered_associations = filtering(X, self.filtered_associations, **self.params)

    def display_stats(self, association: DataFrame, caption: str) -> None:
        """EDA of fitted associations"""

        # appllying style
        subset = [c for c in association if "pct_" in c or "_measure" in c or "_filter" in c]
        style = association.style.background_gradient(cmap="coolwarm", subset=subset)
        style = style.set_table_attributes("style='display:inline'")
        style = style.set_caption(caption)
        display_html(style._repr_html_(), raw=True)

    def fit_features(self, X: DataFrame, y: Series, features: list[str], n_best: int) -> list[str]:
        """Selects the n_best features amongst the specified ones"""

        # initial computation of all association measures
        self.measure_apply(X, y, features)

        # displaying association measure
        if self.verbose:
            self.display_stats(self.associations, "Raw association")

        # iterating over each sort_measures
        # useful when measures hints to specific associations
        ranks = []
        for n, sort_measure in enumerate(self.sort_measures):
            # filtering by sort_measure
            self.filter_apply(X, sort_measure)
            ranks += [list(self.filtered_associations.index)]

            # displaying filtered out association measure
            if n == 0 and self.verbose and len(self.filters) > 1:
                self.display_stats(self.filtered_associations, "Filtered association")

        # retrieving the n_best features per each ranking
        best_features = []
        if len(self.sort_measures) > 0:
            best_features = [feature for rank in ranks for feature in rank[:n_best]]
            best_features = list(set(best_features))  # deduplicating

        return best_features

    def fit(self, X: DataFrame, y: Series) -> None:
        """Selects the n_best features"""

        # splitting features in chunks
        if self.sample_size < 1:
            # shuffling features to get random samples of features
            shuffle(self.features)

            # number of features per sample
            chunks = int(len(self.features) // (1 / self.sample_size))

            # splitting feature list in samples
            feature_samples = [
                self.features[chunks * i : chunks * (i + 1)]
                for i in range(int(1 / self.sample_size) - 1)
            ]

            # adding last sample with all remaining features
            feature_samples += [self.features[chunks * (int(1 / self.sample_size) - 1) :]]

            # iterating over each feature samples
            best_features = []
            for features in feature_samples:
                # fitting association on features
                best_features += self.fit_features(X, y, features, int(self.n_best // 2))

        # splitting in chunks not requested
        else:
            best_features = self.features[:]

        # final selection with all best_features selected
        self.best_features = self.fit_features(X, y, best_features, self.n_best)

        # ordering best_features according to their rank
        self.best_features = [
            f for f in self.filtered_associations.index if f in self.best_features
        ]

        # removing feature from values_orders
        dropped_features = [f for f in self.associations.index if f not in self.best_features]
        for feature in dropped_features:
            if feature in self.values_orders:
                self.values_orders.pop(feature)

        return self

    def transform(self, X: DataFrame, y: Series = None) -> DataFrame:
        """Drops the non-selected columns from `features`.

        Parameters
        ----------
        X : DataFrame
            Contains columns named in `features`
        y : Series, optional
            Model target, by default None

        Returns
        -------
        DataFrame
            `X` without non-selected columns from `features`.
        """

        return X


# MEASURES
def nans_measure(
    active: bool,
    association: Dict[str, Any],
    x: Series,
    y: Series = None,
    **params,
) -> Tuple[bool, Dict[str, Any]]:
    """Measure of the percentage of NaNs

    Parameters
    ----------
    thresh_nan, float: default 1.
      Maximum percentage of NaNs in a feature
    """

    # whether or not tests where passed
    if active:
        nans = x.isnull()  # ckecking for nans
        pct_nan = nans.mean()  # Computing percentage of nans

        # updating association
        association.update({"pct_nan": pct_nan})

        # Excluding feature that have to many NaNs
        active = pct_nan < params.get("thresh_nan", 1.0)

    return active, association


def dtype_measure(
    active: bool,
    association: Dict[str, Any],
    x: Series,
    y: Series = None,
    **params,
) -> Tuple[bool, Dict[str, Any]]:
    """Gets dtype"""

    # updating association
    association.update({"dtype": x.dtype})

    return active, association


def mode_measure(
    active: bool,
    association: Dict[str, Any],
    x: Series,
    y: Series = None,
    **params,
) -> Tuple[bool, Dict[str, Any]]:
    """Measure of the percentage of the Mode

    Parameters
    ----------
    thresh_mode, float: default 1.
      Maximum percentage of the mode of a feature
    """

    # whether or not tests where passed
    if active:
        mode = x.mode(dropna=True).values[0]  # computing mode
        pct_mode = (x == mode).mean()  # Computing percentage of the mode

        # updating association
        association.update({"pct_mode": pct_mode, "mode": mode})

        # Excluding feature with too frequent modes
        active = pct_mode < params.get("thresh_mode", 1.0)

    return active, association


def kruskal_measure(
    active: bool, association: Dict[str, Any], x: Series, y: Series, **params
) -> Tuple[bool, Dict[str, Any]]:
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
    active: bool, association: Dict[str, Any], x: Series, y: Series, **params
) -> Tuple[bool, Dict[str, Any]]:
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
    association: Dict[str, Any],
    x: Series,
    y: Series = None,
    **params,
) -> Tuple[bool, Dict[str, Any]]:
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
    association: Dict[str, Any],
    x: Series,
    y: Series = None,
    **params,
) -> Tuple[bool, Dict[str, Any]]:
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


def chi2_measure(
    active: bool, association: Dict[str, Any], x: Series, y: Series, **params
) -> Tuple[bool, Dict[str, Any]]:
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
    active: bool, association: Dict[str, Any], x: Series, y: Series, **params
) -> Tuple[bool, Dict[str, Any]]:
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
    active: bool, association: Dict[str, Any], x: Series, y: Series, **params
) -> Tuple[bool, Dict[str, Any]]:
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


# FILTERS
def thresh_filter(X: DataFrame, ranks: DataFrame, **params) -> Dict[str, Any]:
    """Filters out missing association measure (did not pass a threshold)"""

    # drops rows with nans
    associations = ranks.dropna(axis=0)

    return associations


def measure_filter(X: DataFrame, ranks: DataFrame, **params) -> Dict[str, Any]:
    """Filters out specified measure's lower ranks than threshold

    Parameters
    ----------
    thresh_measure, float: default 0.
        Minimum association between target and features
        To be used with: `association_filter`
    name_measure, str
        Measure to be used for minimum association filtering
        To be used with: `association_filter`
    """

    associations = ranks.copy()

    # drops rows with nans
    if "name_measure" in params:
        associations = ranks[ranks[params.get("name_measure")] > params.get("thresh_measure", 0.0)]

    return associations


def quantitative_filter(
    X: DataFrame, ranks: DataFrame, corr_measure: str, **params
) -> Dict[str, Any]:
    """Computes max association between X and X (quantitative) excluding features
    that are correlated to a feature more associated with the target
    (defined by the ranks).

    Parameters
    ----------
    thresh_corr, float: default 1.
        Maximum association between features
    """

    # accessing the prefered order
    prefered_order = ranks.index

    # computing correlation between features
    X_corr = X[prefered_order].corr(corr_measure).abs()
    X_corr = X_corr.where(triu(ones(X_corr.shape), k=1).astype(bool))

    # initiating list of maximum association per feature
    associations = []

    # iterating over each feature by target association order
    for feature in prefered_order:
        # correlation with features more associated to the target
        corr_with_better_features = X_corr.loc[:feature, feature]

        # maximum correlation with a better feature
        corr_with, worst_corr = corr_with_better_features.agg(["idxmax", "max"])

        # dropping the feature if it was too correlated to a better feature
        if worst_corr > params.get("thresh_corr", 1):
            X_corr = X_corr.drop(feature, axis=0).drop(feature, axis=1)

        # kept feature: updating associations with this feature
        else:
            associations += [
                {
                    "feature": feature,
                    f"{corr_measure}_filter": worst_corr,
                    f"{corr_measure}_with": corr_with,
                }
            ]

    # formatting ouput to DataFrame
    associations = DataFrame(associations).set_index("feature")

    # applying filter on association
    associations = ranks.join(associations, how="right")

    return associations


def spearman_filter(X: DataFrame, ranks: DataFrame, **params) -> Dict[str, Any]:
    """Computes max Spearman between X and X (quantitative) excluding features
    that are correlated to a feature more associated with the target
    (defined by the ranks).

    Parameters
    ----------
    thresh_corr, float: default 1.
      Maximum association between features
    """

    # applying quantitative filter with spearman correlation
    return quantitative_filter(X, ranks, "spearman", **params)


def pearson_filter(X: DataFrame, ranks: DataFrame, **params) -> Dict[str, Any]:
    """Computes max Pearson between X and X (quantitative) excluding features
    that are correlated to a feature more associated with the target
    (defined by the ranks).

    Parameters
    ----------
    thresh_corr, float: default 1.
      Maximum association between features
    """

    # applying quantitative filter with spearman correlation
    return quantitative_filter(X, ranks, "pearson", **params)


def vif_filter(X: DataFrame, ranks: DataFrame, **params) -> Dict[str, Any]:
    """Computes Variance Inflation Factor (multicolinearity)

    Parameters
    ----------
    thresh_vif, float: default inf
      Maximum VIF between features
    """

    # accessing the prefered order
    prefered_order = ranks.index

    # initiating list association per feature
    associations = []

    # list of dropped features
    dropped = []

    # iterating over each column
    for i, feature in enumerate(prefered_order):
        # identifying remaining more associated features
        better_features = [f for f in prefered_order[: i + 1] if f not in dropped]

        X_vif = X[better_features]  # keeping only better features
        X_vif = X_vif.dropna(axis=0)  # dropping NaNs for OLS

        # computation of VIF
        vif = nan
        if len(better_features) > 1 and len(X_vif) > 0:
            vif = variance_inflation_factor(X_vif.values, len(better_features) - 1)

        # dropping the feature if it was too correlated to a better feature
        if vif > params.get("thresh_vif", inf) and notna(vif):
            dropped += [feature]

        # kept feature: updating associations with this feature
        else:
            associations += [{"feature": feature, "vif_filter": vif}]

    # formatting ouput to DataFrame
    associations = DataFrame(associations).set_index("feature")

    # applying filter on association
    associations = ranks.join(associations, how="right")

    return associations


def qualitative_worst_corr(
    X: DataFrame,
    feature: str,
    ranks: DataFrame,
    corr_measure: Callable,
    **params,
):
    """Computes maximum association between a feature and features
    more associated to the target (according to ranks)
    """

    # measure name
    measure_name = corr_measure.__name__
    measure = measure_name.replace("_measure", "")

    # initiating worst correlation
    worst_corr = {"feature": feature}

    # features more associated with target
    better_features = list(ranks.loc[:feature].index)[:-1]

    # iterating over each better feature
    for better_feature in better_features:
        # computing association with better feature
        _, association = corr_measure(
            True,
            {f"{measure}_with": better_feature},
            X[feature],
            X[better_feature],
            **params,
        )

        # updating association if it's greater than previous better features
        if association.get(measure_name) > worst_corr.get(measure_name, 0):
            # renaming association measure as filter
            association[f"{measure}_filter"] = association.pop(measure_name)

            # removing temporary measures
            association = {k: v for k, v in association.items() if "_measure" not in k}

            # updating worst known association
            worst_corr.update(association)

        # stopping measurements if association is greater than threshold
        if association.get(f"{measure}_filter") > params.get("thresh_corr", 1):
            ranks = ranks.drop(feature, axis=0)  # removing feature from ranks

            return ranks, None

    return ranks, worst_corr


def qualitative_filter(
    X: DataFrame, ranks: DataFrame, corr_measure: Callable, **params
) -> Dict[str, Any]:
    """Computes max association between X and X (qualitative) excluding features
    that are correlated to a feature more associated with the target
    (defined by the ranks).

    Parameters
    ----------
    thresh_corr, float: default 1.
        Maximum association between features
    """

    # accessing the prefered order
    prefered_order = ranks.index

    # initiating list of maximum association per feature
    associations = []

    # iterating over each feature by target association order
    for feature in prefered_order:
        # computing correlation with better features anf filtering out ranks
        ranks, worst_corr = qualitative_worst_corr(X, feature, ranks, corr_measure, **params)

        # updating associations
        if worst_corr:
            associations += [worst_corr]

    # formatting ouput to DataFrame
    associations = DataFrame(associations).set_index("feature")

    # applying filter on association
    associations = ranks.join(associations, how="right")

    return associations


def cramerv_filter(X: DataFrame, ranks: DataFrame, **params) -> Dict[str, Any]:
    """Computes max Cramer's V between X and X (qualitative) excluding features
    that are correlated to a feature more associated with the target
    (defined by the ranks).

    Parameters
    ----------
    thresh_corr, float: default 1.
        Maximum association between features
    """

    # applying quantitative filter with Cramer's V correlation
    return qualitative_filter(X, ranks, cramerv_measure, **params)


def tschuprowt_filter(X: DataFrame, ranks: DataFrame, **params) -> Dict[str, Any]:
    """Computes max Tschuprow's T between X and X (qualitative) excluding
     features that are correlated to a feature more associated with the target
    (defined by the ranks).

    Parameters
    ----------
    thresh_corr, float: default 1.
        Maximum association between features
    """

    # applying quantitative filter with Tschuprow's T correlation
    return qualitative_filter(X, ranks, tschuprowt_measure, **params)
