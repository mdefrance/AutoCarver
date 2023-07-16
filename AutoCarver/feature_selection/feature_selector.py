"""Tools to select the best Quantitative and Qualitative features."""

from random import shuffle
from typing import Any, Callable

from pandas import Series, DataFrame
from IPython.display import display_html
from sklearn.base import BaseEstimator, TransformerMixin

from ..discretizers import GroupedList
from .filters import thresh_filter
from .measures import dtype_measure, mode_measure, nans_measure


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
        self.values_orders = {
            feature: GroupedList(value) for feature, value in values_orders.items()
        }

        self.drop = drop
        self.copy = copy
        self.verbose = verbose
        self.params = params

        self.associations = None
        self.filtered_associations = None

    def measure(self, x: Series, y: Series) -> dict[str, Any]:
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
