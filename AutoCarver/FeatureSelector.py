from IPython.display import display_html
from math import sqrt
from numpy import triu, ones, nan, inf
from pandas import DataFrame, Series, notna
from scipy.stats import kruskal
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import List, Dict, Any, Callable


class FeatureSelector(BaseEstimator, TransformerMixin):
    """ A pipeline of measures to perform EDA and feature pre-selection
        
    Parameters
    ----------
    features: List[str]
        Features on which to compute association.
    n_best, int:
        Number of features to be selected
    measures, List[Callable]: default list().
        List of association measures to be used.
        Implemented measures are:
            - For association evaluation: `kruskal_measure`, `R_measure`
            - For outlier detection: `zscore_measure`, `iqr_measure`
        Ranks features based on last measure of the list.
    filters, List[Callable]: default list().
        List of filters to be used.
        Implemented filters are:
          - For linear correlation: `spearman_filter`, `pearson_filter`
          - For multicoloinearity: `vif_filter`

    params examples
    ---------------
    thresh_nan, float: default 1.
        Maximum percentage of NaNs in a feature
    thresh_mode, float: default 1.
        Maximum percentage of the mode of a feature
    thresh_outlier, float: default 1.
        Maximum percentage of Outliers in a feature
    thresh_corr, float: default 1.
        Maximum association between features
    thresh_vif, float: default inf
        Maximum VIF between features
    ascending, bool: default False
        Ascending of Descending sort by sort_measure
    """
    
    def __init__(self, features: List[str], n_best: int, measures: List[Callable]=list(), filters: List[Callable]=list(),
                 copy: bool=True, verbose: bool=True, **params) -> None:
        
        self.features = features[:]
        self.n_best = n_best
        assert n_best <= len(features), "Must set n_best <= len(features)"

        self.measures = [dtype_measure, nans_measure, mode_measure, zscore_measure] + measures[:]
        self.filters = [thresh_filter] + filters[:]
        self.sort_measures = [measure.__name__ for measure in measures[-1::]]

        self.copy = copy
        self.verbose = verbose
        self.params = params
    
    def measure(self, x: Series, y: Series) -> Dict[str, Any]:
        """ Measures association between x and y """
        
        passed = True  # measures keep going only if previous basic tests are passed
        association = {}

        # iterating over each measure
        for measure in self.measures:
            passed, association = measure(passed, association, x, y, **self.params)
            
        return association
    
    def measure_apply(self, X: DataFrame, y: Series) -> None:
        """ Measures association between columns of X and y

    Parameters
    ----------
    ascending, bool: default False
      Ascending of Descending sort by sort_measure
    """
        
        # applying association measure to each column
        self.associations = X[self.features].apply(self.measure, y=y, result_type='expand', axis=0).T
        self.associations = self.associations.sort_values(self.sort_measures[0], ascending=self.params.get('ascending', False))
    
    def filter_apply(self, X: DataFrame, sort_measure: str) -> DataFrame:
        """ Filters out too correlated features (least relevant first)

    Parameters
    ----------
    ascending, bool: default False
      Ascending of Descending sort by sort_measure
    """
        
        # ordering features by sort_by
        self.filtered_associations = self.associations.sort_values(sort_measure, ascending=self.params.get('ascending', False))

        # applying successive filters
        for filtering in self.filters:

            # ordered filtering
            self.filtered_associations = filtering(X, self.filtered_associations, **self.params)
    
    def display_stats(self, association: DataFrame, caption: str) -> None:
        """ EDA of fitted associations"""
        
        # appllying style 
        subset = [c for c in association if 'pct_' in c or '_measure' in c]
        style = association.style.background_gradient(cmap='coolwarm', subset=subset)
        style = style.set_table_attributes("style='display:inline'")
        style = style.set_caption(caption)
        display_html(style._repr_html_(), raw=True)
        
    
    def fit(self, X: DataFrame, y: Series) -> None:
        """ Selects the n_best features"""
        
        # initial computation of all association measures
        self.measure_apply(X, y)

        # displaying association measure
        if self.verbose:
            self.display_stats(self.associations, 'Raw association')
        
        # iterating over each sort_measures 
        # useful when measures hints to specific associations
        ranks = []
        for n, sort_measure in enumerate(self.sort_measures):
            
            # filtering by sort_measure
            self.filter_apply(X, sort_measure)
            ranks += [list(self.filtered_associations.index)]

            # displaying filtered out association measure
            if n == 0 and self.verbose:
                self.display_stats(self.filtered_associations, 'Filtered association')

        # retrieving the n_best features per ranking
        self.best_features = [feature for rank in ranks for feature in rank[:self.n_best]]
        self.best_features = list(set(self.best_features))  # deduplicating

        # displaying filtered out association measure
        # if n == 0 and self.verbose:
        #     self.display_stats(self.associations.reindex(self.best_features), 'Filtered association')

        return self

    def transform(self, X: DataFrame, y: Series=None) -> DataFrame:

        # copying dataset
        Xc = X
        if self.copy:
            Xc.copy()

        # filtering out unwanted features
        Xc = X.drop([c for c in self.features if c not in self.best_features], axis=1)

        return Xc


# MEASURES
def nans_measure(active: bool, association: Dict[str, Any], x: Series, y: Series=None, **params) -> Tuple[bool, Dict[str, Any]]:
    """ Measure of the percentage of NaNs

    Parameters
    ----------
    thresh_nan, float: default 1.
      Maximum percentage of NaNs in a feature
    """
    
    passed = True  # whether or not tests where passed
    
    nans = x.isnull()  # ckecking for nans
    pct_nan = nans.mean()   # Computing percentage of nans
    
    # updating association
    association.update({'pct_nan': pct_nan})
    
    # Excluding feature that have to many NaNs
    passed = pct_nan < params.get('thresh_nan', 1.)
    
    return passed, association

def dtype_measure(active: bool, association: Dict[str, Any], x: Series, y: Series=None, **params) -> Tuple[bool, Dict[str, Any]]:
    """ Gets dtype"""
    
    # updating association
    association.update({'dtype': x.dtype})
        
    return active, association

def mode_measure(active: bool, association: Dict[str, Any], x: Series, y: Series=None, **params) -> Tuple[bool, Dict[str, Any]]:
    """ Measure of the percentage of the Mode

    Parameters
    ----------
    thresh_mode, float: default 1.
      Maximum percentage of the mode of a feature
    """
    
    passed = True  # whether or not tests where passed
    
    mode = x.mode(dropna=True).values[0]  # computing mode
    pct_mode = (x == mode).mean()  # Computing percentage of the mode
    
    # updating association
    association.update({'pct_mode': pct_mode, 'mode': mode})
    
    # Excluding feature with too frequent modes
    passed = pct_mode < params.get('thresh_mode', 1.)
    
    return passed, association

def kruskal_measure(active: bool, association: Dict[str, Any], x: Series, y: Series, **params) -> Tuple[bool, Dict[str, Any]]:
    """ Kruskal-Wallis statistic between x (quantitative) and y (binary)"""
    
    # check that previous steps where passed
    if active:
    
        nans = x.isnull()  # ckecking for nans
        
        # computation of Kruskal-Wallis statistic
        kw = kruskal(x[(~nans) & (y == 0)], x[(~nans) & (y == 1)])
        
        # updating association
        if kw:
            association.update({'kruskal_measure': kw[0]})
        
    return active, association

def R_measure(active: bool, association: Dict[str, Any], x: Series, y: Series, **params) -> Tuple[bool, Dict[str, Any]]:
    """ R of the linear regression of x (quantitative) by y (binary)"""
    
    # check that previous steps where passed
    if active:
    
        nans = x.isnull()  # ckecking for nans

        # grouping feature and target
        ols_df = DataFrame({'feature': x[~nans], 'target': y[~nans]})
        
        # fitting regression of feature by target
        regression = ols('feature~C(target)', ols_df).fit()
        
        # updating association
        if regression.rsquared:
            association.update({'R_measure': sqrt(regression.rsquared)})
        
    return active, association


def zscore_measure(active: bool, association: Dict[str, Any], x: Series, y: Series=None, **params) -> Tuple[bool, Dict[str, Any]]:
    """ Computes outliers based on the z-score

    Parameters
    ----------
    thresh_outlier, float: default 1.
      Maximum percentage of Outliers in a feature
    """
    
    # check that previous steps where passed for computational optimization
    if active:
        
        mean = x.mean()  # mean of the feature
        std = x.std()  # standard deviation of the feature
        zscore = (x-mean) / std  # zscore per observation
        
        # checking for outliers
        outliers = abs(zscore) > 3
        pct_zscore = outliers.mean()
        
        # updating association
        association.update({
            'pct_zscore': pct_zscore,
            'min': x.min(),
            'max': x.max(),
            'mean': mean,
            'std': std
        })

        # Excluding feature with too frequent modes
        passed = pct_zscore < params.get('thresh_outlier', 1.)
        
    return active, association

def iqr_measure(active: bool, association: Dict[str, Any], x: Series, y: Series=None, **params) -> Tuple[bool, Dict[str, Any]]:
    """ Computes outliers based on the inter-quartile range

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
        association.update({
            'pct_iqr': pct_iqr,
            'q1': q1,
            'median': x.median(),
            'q3': q3
        })

        # Excluding feature with too frequent modes
        passed = pct_iqr < params.get('thresh_outlier', 1.)
        
    return active, association

    
# FILTERS        
def thresh_filter(X: DataFrame, ranks: DataFrame, **params) -> Dict[str, Any]:
    """ Filters out missing association measure (did not pass a threshold)"""
    
    # drops rows with nans
    associations = ranks.dropna(axis=0)
    
    return associations

def quantitative_filter(X: DataFrame, ranks: DataFrame, corr_measure: str, **params) -> Dict[str, Any]:
    """ Computes max association between X and X (quantitative) excluding features 
    that are correlated to a feature more associated with the target 
    (defined by the prefered_order).

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
        corr_with, worst_corr = corr_with_better_features.agg(['idxmax', 'max'])
        
        # dropping the feature if it was too correlated to a better feature
        if worst_corr > params.get('thresh_corr', 1.):
            X_corr = X_corr.drop(feature, axis=0).drop(feature, axis=1)
            
        # kept feature: updating associations with this feature
        else:
            associations += [{
                'feature': feature, 
                f'{corr_measure}_measure': worst_corr,
                f'{corr_measure}_with': corr_with
            }]
            
    # formatting ouput to DataFrame
    associations = DataFrame(associations).set_index('feature')
            
    # applying filter on association
    associations = ranks.join(associations, how='right')
    
    return associations 


def spearman_filter(X: DataFrame, ranks: DataFrame, **params) -> Dict[str, Any]:
    """ Computes max Spearman between X and X (quantitative) excluding features 
    that are correlated to a feature more associated with the target 
    (defined by the prefered_order).

    Parameters
    ----------
    thresh_corr, float: default 1.
      Maximum association between features
    """
            
    # applying quantitative filter with spearman correlation
    associations = quantitative_filter(X, ranks, 'spearman', **params)
    
    return associations

def pearson_filter(X: DataFrame, ranks: DataFrame, **params) -> Dict[str, Any]:
    """ Computes max Pearson between X and X (quantitative) excluding features 
    that are correlated to a feature more associated with the target 
    (defined by the prefered_order).

    Parameters
    ----------
    thresh_corr, float: default 1.
      Maximum association between features
    """
            
    # applying quantitative filter with spearman correlation
    associations = quantitative_filter(X, ranks, 'pearson', **params)
    
    return associations

def vif_filter(X: DataFrame, ranks: DataFrame, **params) -> Dict[str, Any]:
    """ Computes Variance Inflation Factor (multicolinearity)

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
        better_features = [f for f in prefered_order[:i+1] if f not in dropped]
        
        X_vif = X[better_features]  # keeping only better features
        X_vif = X_vif.dropna(axis=0)  # dropping NaNs for OLS
        
        # computation of VIF
        vif = nan
        if len(better_features) > 1 and len(X_vif) > 0:
            vif = variance_inflation_factor(X_vif.values, len(better_features)-1)
        
        # dropping the feature if it was too correlated to a better feature
        if vif > params.get('thresh_vif', inf) and notna(vif):
            dropped += [feature]
            
        # kept feature: updating associations with this feature
        else:
            associations += [{'feature': feature, 'vif_measure': vif}]
    
    # formatting ouput to DataFrame
    associations = DataFrame(associations).set_index('feature')
            
    # applying filter on association
    associations = ranks.join(associations, how='right')
    
    return associations
