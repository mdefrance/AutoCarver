from IPython.display import display_html
from math import sqrt
from numpy import triu, ones
from pandas import DataFrame, Series
from scipy.stats import kruskal
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor


class FeatureSelector():
    """ A pipeline of measures to perform EDA and feature pre-selection"""
    
    def __init__(self, measures: list=[], filters: list=[]):
        
        self.measures = [dtype_measure, nans_measure, mode_measure] + measures[:]
        self.filters = [thresh_filter] + filters[:]
        
        self.associations = []
        self.filtered_associations = []
    
    def measure(self, x: Series, y: Series, **params):
        """ Measures association between x and y
        
        Parameters
        ----------
        thresh_nan, float: default 1.
          Maximum percentage of NaNs in a feature
        thresh_mode, float: default 1.
          Maximum percentage of the mode of a feature
        thresh_outlier, float: default 1.
          Maximum percentage of Outliers in a feature
        """
        
        passed = True
        association = {}
        for measure in self.measures:
            passed, association = measure(passed, association, x, y, **params)
            
        return association
    
    def measure_apply(self, X: DataFrame, y: Series, **params):
        """ Measures association between columns of X and y
        
        Parameters
        ----------
        thresh_nan, float: default 1.
          Maximum percentage of NaNs in a feature
        thresh_mode, float: default 1.
          Maximum percentage of the mode of a feature
        thresh_outlier, float: default 1.
          Maximum percentage of Outliers in a feature
        """
        
        # applying association measure to each column
        self.associations = X.apply(self.measure, y=y, **params, result_type='expand', axis=0).T
        
        return self.associations
    
    def filter_apply(self, X: DataFrame, sort_measure: str, **params):
        """ Filters out to correlated features that are less relevant
        
        Parameters
        ----------
        thresh_corr, float: default 1.
          Maximum association between features
        ascending, bool: default False
          Ascending of Descending sort by sort_measure
        """
        
        # ordering features by sort_by
        self.filtered_associations = self.associations.sort_values(sort_measure, ascending=params.get('ascending', False))

        # applying successive filters
        for filtering in self.filters:

            # ordered filtering
            self.filtered_associations = filtering(X, self.filtered_associations, **params)
        
        return self.filtered_associations
    
    def display_stats(self, X: DataFrame, y: Series):
        """ Computes statistics for EDA with default params"""
        
        # initial computation of all association measures
        self.measure_apply(X, y)
        
        # filtering
        association = self.filter_apply(X, self.measures[-1].__name__)
            
        # appllying style 
        subset = [c for c in association if 'pct_' in c or '_measure' in c]
        association = association.style.background_gradient(cmap='coolwarm', subset=subset)
        display_html(association.set_table_attributes("style='display:inline'")._repr_html_(), raw=True)
        
    
    def select_features(self, X: DataFrame, y: Series, n_best: int, sort_measures: list, **params):
        """ Selects the n_best features
        
        Parameters
        ----------
        thresh_nan, float: default 1.
          Maximum percentage of NaNs in a feature
        thresh_mode, float: default 1.
          Maximum percentage of the mode of a feature
        thresh_outlier, float: default 1.
          Maximum percentage of Outliers in a feature
        thresh_corr, float: default 1.
          Maximum association between features
        ascending, bool: default False
          Ascending of Descending sort by sort_measure
        """
        
        # initial computation of all association measures
#         if len(self.associations) == 0:
        self.measure_apply(X, y, **params)
        
        # iterating over each sort_measures
        ranks = []
        for sort_measure in sort_measures:
            
            # filtering by sort_measure
            rank = list(self.filter_apply(X, sort_measure, **params).index)
            ranks += [rank]
        
        # retrieving the n_best features per ranking
        best_features = [feature for rank in ranks for feature in rank[:n_best]]
        best_features = list(set(best_features))  # deduplicating
        
        return best_features

    
# MEASURES
def nans_measure(active: bool, association: dict, x: Series, y: Series=None, **params):
    """ Measure of the percentage of NaNs"""
    
    passed = True  # whether or not tests where passed
    
    nans = x.isnull()  # ckecking for nans
    pct_nan = nans.mean()   # Computing percentage of nans
    
    # updating association
    association.update({'pct_nan': pct_nan})
    
    # Excluding feature that have to many NaNs
    passed = pct_nan < params.get('thresh_nan', 1.)
    
    return passed, association

def dtype_measure(active: bool, association: dict, x: Series, y: Series=None, **params):
    """ Gets dtype"""
    
    # updating association
    association.update({'dtype': x.dtype})
        
    return active, association

def mode_measure(active: bool, association: dict, x: Series, y: Series=None, **params):
    """ Measure of the percentage of the Mode"""
    
    passed = True  # whether or not tests where passed
    
    mode = x.mode(dropna=True).values[0]  # computing mode
    pct_mode = (x == mode).mean()  # Computing percentage of the mode
    
    # updating association
    association.update({'pct_mode': pct_mode, 'mode': mode})
    
    # Excluding feature with too frequent modes
    passed = pct_mode < params.get('thresh_mode', 1.)
    
    return passed, association

def kruskal_measure(active: bool, association: dict, x: Series, y: Series, **params):
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

def R_measure(active: bool, association: dict, x: Series, y: Series, **params):
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


def zscore_measure(active: bool, association: dict, x: Series, y: Series=None, **params):
    """ Computes outliers based on the z-score"""
    
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

def iqr_measure(active: bool, association: dict, x: Series, y: Series=None, **params):
    """ Computes outliers based on the inter-quartile range"""
    
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
def thresh_filter(X: DataFrame, ranks: DataFrame, **params):
    """ Filters out missing association measure (did not pass a threshold)"""
    
    # drops rows with nans
    associations = ranks.dropna(axis=0)
    
    return associations

def quantitative_filter(X: DataFrame, ranks: DataFrame, corr_measure: str, **params):
    """ Computes max association between X and X (quantitative) excluding features 
    that are correlated to a feature more associated with the target 
    (defined by the prefered_order)."""
    
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


def spearman_filter(X: DataFrame, ranks: DataFrame, **params):
    """ Computes max Spearman between X and X (quantitative) excluding features 
    that are correlated to a feature more associated with the target 
    (defined by the prefered_order)."""
            
    # applying quantitative filter with spearman correlation
    associations = quantitative_filter(X, ranks, 'spearman', **params)
    
    return associations

def pearson_filter(X: DataFrame, ranks: DataFrame, **params):
    """ Computes max Pearson between X and X (quantitative) excluding features 
    that are correlated to a feature more associated with the target 
    (defined by the prefered_order)."""
            
    # applying quantitative filter with spearman correlation
    associations = quantitative_filter(X, ranks, 'pearson', **params)
    
    return associations

def vif_filter(X: DataFrame, ranks: DataFrame, **params):
    """ Computes Variance Inflation Factor (multicolinearity)"""
    
    # accessing the prefered order
    prefered_order = ranks.index
    
    # initiating list of maximum association per feature
    associations = []
    
    
    # iterating over each column
    for i, feature in enumerate(X):
        vif = variance_inflation_factor(X.values, i)
        associations += [{'feature': feature, 'vif_measure': vif}]
    
    # formatting ouput to DataFrame
    associations = DataFrame(associations).set_index('feature')
            
    # applying filter on association
    associations = ranks.join(associations, how='right')
    
    return associations
