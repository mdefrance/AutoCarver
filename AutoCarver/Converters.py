from numpy import select, nan, tanh
from pandas import isna, notna, DataFrame, Series, to_datetime
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Tuple, Any


class StringConverter(BaseEstimator, TransformerMixin):
    """ Converts specified columns a DataFrame into str
    
     - Keeps NaN inplace
     - Converts floats of int to int

    Parameters
    ----------
    features: list, default []
        List of columns to be converted to string.
    """
    
    def __init__(self, features: List[str]=[], copy: bool=False) -> None:
    
        self.features = features[:]
        self.copy = copy
        
    def fit(self, X: DataFrame, y: Series=None):
        
        return self
        
    def transform(self, X: DataFrame, y: Series=None) -> DataFrame:
        
        # copying DataFrame if requested
        Xc = X
        if self.copy:
            Xc = X.copy()

        # storing nans
        nans = isna(Xc[self.features])

        # storing ints
        ints = Xc[self.features].applymap(lambda u: isinstance(u, float) and float.is_integer(u))
        
        # converting to string
        Xc[self.features] = Xc[self.features].astype(str)
        
        # converting to int-strings
        converted_ints = Xc[ints][self.features].applymap(lambda u: str(int(float(u))) if isinstance(u, str) else u)
        Xc[self.features] = select([ints], [converted_ints], default=Xc[self.features])
        
        # converting back to nan
        Xc[nans] = nan
        
        return Xc


class TimeDeltaConverter(BaseEstimator, TransformerMixin):
    """ Converts specified DateTime columns into TimeDeltas between themselves

     - Str Columns are converted to pandas datetime columns
     - New TimeDelta column names are stored in TimeDeltaConverter.delta_features

    Parameters
    ----------
    features: List[str]
        List of DateTime columns to be converted to string.
    nans: Any, default numpy.nan
        Date value to be considered as missing data.
    drop: bool, default True
        Whether or not to drop initial DateTime columns (specified in features).
    """
    
    def __init__(self, features: List[str], nans: Any=nan, copy: bool=False, drop: bool=True) -> None:
        
        self.features = features[:]
        self.copy = copy
        self.delta_features: List[str] = []
        self.nans = nans
        self.drop = drop
        
    def fit(self, X: DataFrame, y=None) -> None:
        
        # creating list of names of delta featuers
        for i, date1 in enumerate(self.features):
            for date2 in self.features[i+1:]:
                self.delta_features += [f'delta_{date1}_{date2}']
        
        return self
    
    def transform(self, X: DataFrame, y: Series=None) -> DataFrame:
        
        # copying dataset
        Xc = X
        if self.copy:
            Xc = X.copy()
            
        # converting back nans
        if notna(self.nans):
            Xc[self.features] = Xc[self.features].where(Xc[self.features]!=self.nans, nan)
        
        # converting to datetime
        Xc[self.features] = to_datetime(Xc[self.features].stack()).unstack()
        
        # iterating over each combination of dates
        for i, date1 in enumerate(self.features):
            for date2 in self.features[i+1:]:
                
                # computing timedelta of days
                Xc[f'delta_{date1}_{date2}'] = (Xc[date1] - Xc[date2]).dt.days
        
        # dropping date columns
        Xc = Xc.drop(self.features, axis=1)
        
        return Xc


class TanhNormalizer(BaseEstimator, TransformerMixin):
    """ Tanh Normalization that keeps data distribution and borns between 0 and 1

    Parameters
    ----------
    features: List[str]
        List of quantitative features to be normalized.
    """
    
    def __init__(self, features: List[str], copy: bool=False) -> None:
        
        self.features = features[:]
        self.copy = copy
        
    def fit(self, X: DataFrame, y=None) -> None:
        
        self.distribs = X[self.features].agg(['mean', 'std']).to_dict(orient='index')
        
        return self
    
    def transform(self, X: DataFrame, y: Series=None) -> DataFrame:
        
        # copying dataset
        Xc = X
        if self.copy:
            Xc = X.copy()
        
        # applying tanh normalization
        Xc[self.features] = (tanh(Xc[self.features].sub(self.distribs['mean']).divide(self.distribs['std']) * 0.01) + 1 ) / 2
        
        return Xc