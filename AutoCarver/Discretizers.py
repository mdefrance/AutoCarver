from .Converters import StringConverter
from IPython.display import display_html
from numpy import sort, nan, inf, float32, where, isin, argsort, array, select, append, quantile, linspace, argmin
from pandas import DataFrame, Series, isna, qcut, notna, unique
from pandas.api.types import is_numeric_dtype, is_string_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Any, Dict, List
from warnings import warn

def nan_unique(x: Series):
    """ Unique non-NaN values. """
    
    # unique values
    uniques = unique(x)
    
    # filtering out nans
    uniques = [u for u in uniques if notna(u)]
    
    return uniques

class GroupedList(list):
    
    def __init__(self, iterable=()) -> None:
        """ An ordered list that historizes its elements' merges."""

        # case 0: iterable is the contained dict
        if isinstance(iterable, dict):
            # TODO: check thaht keys are in list
            
            # récupération des valeurs de la liste (déjà ordonné)
            values = [key for key in iterable]

            # initialsiation de la liste
            super().__init__(values)

            # attribution des valeurs contenues
            self.contained = {k: v for k, v in iterable.items()}

            # adding key to itself if that's not the case
            for k in [k for k in values if k not in self.contained.get(k)]:
                self.contained.update({k: self.contained.get(k) + [k]})
        
        # case 1: copying a GroupedList
        elif hasattr(iterable, 'contained'):

            # initialsiation de la liste
            super().__init__(iterable)

            # copie des groupes
            self.contained = {k: v for k, v in iterable.contained.items()}
        
        # case 2: initiating GroupedList from a list
        elif isinstance(iterable, list):

            # initialsiation de la liste
            super().__init__(iterable)

            # création des groupes
            self.contained = {v: [v] for v in iterable}

    def group_list(self, to_discard: List[Any], to_keep: Any) -> None:
        """ Groups elements to_discard into values to_keep"""
        
        for discarded, kept in zip(to_discard, [to_keep] * len(to_discard)):
            self.group(discarded, kept)

    def group(self, discarded: Any, kept: Any) -> None:
        """ Groups the discarded value with the kept value"""

        # checking that those values are distinct
        if not is_equal(discarded, kept):

            # checking that those values exist in the list
            assert discarded in self, f"{discarded} not in list"
            assert kept in self, f"{kept} not in list"

            # accessing values contained in each value
            contained_discarded = self.contained.get(discarded)
            contained_kept = self.contained.get(kept)

            # updating contained dict
            self.contained.update({
                kept: contained_discarded + contained_kept,
                discarded: []
            })

            # removing discarded from the list
            self.remove(discarded)
        
        return self
        
    def append(self, new_value: Any) -> None:
        """ Appends a new_value to the GroupedList"""
        
        self += [new_value]
        
        self.contained.update({new_value: [new_value]})
        
        return self
        
    def update(self, new_value: Dict[Any, List[Any]]) -> None:
        """ Updates the GroupedList via a dict"""
        
        # adding keys to the order if they are new values
        for k in [c for c in new_value if c not in self]:
            self += new_value.keys()
        
        # updating contained accord to new_value
        self.contained.update(new_value)
        
        return self

    def sort(self) -> None:
        """ Sorts the values of the list and dict (if any, NaNs are last). """

        # str values
        keys_str = [key for key in self if isinstance(key, str)]

        # non-str values
        keys_float = [key for key in self if not isinstance(key, str)]

        # sorting and merging keys
        keys = list(sort(keys_str)) + list(sort(keys_float)) 

        # recreating an ordered GroupedList
        self = GroupedList({k: self.get(k) for k in keys})

        return self

    def sort_by(self, ordering: List[Any]) -> None:
        """ Sorts the values of the list and dict, if any, NaNs are the last. """

        # checking that all values are given an order
        assert all([o in self for o in ordering]), f"Unknown values in ordering: {', '.join([str(v) for v in ordering if v not in self])}"
        assert all([s in ordering for s in self]), f"Missing value from ordering: {', '.join([str(v) for v in self if v not in ordering])}"

        # ordering the contained
        self = GroupedList({k: self.get(k) for k in ordering})

        return self

    
    def remove(self, value: Any) -> None:
        
        super().remove(value)
        self.contained.pop(value)
    
    def pop(self, idx: int) -> None:
        
        value = self[idx]
        self.remove(value)
    
    def get(self, key: Any) -> List[Any]:
        """ returns list of values contained in key"""

        # default to fing an element
        found = self.contained.get(key)

        # copying with dictionnaries (not working with numpy.nan)
        # if isna(key):
            # found = [value for dict_key, value in self.contained.items() if is_equal(dict_key, key)][0]

        return found

    def get_group(self, value: Any) -> Any:
        """ returns the group containing the specified value """
        
        found = [key for key, values in self.contained.items() if any(is_equal(value, elt) for elt in values)]

        if any(found):
            return found[0]
        else:
            return value

    def values(self) -> List[Any]:
        """ returns all values contained in each group """

        known = [value for values in self.contained.values() for value in values]

        return known

    def contains(self, value: Any) -> bool:
        """ checks if a value if contained in any group """

        known_values = self.values()

        return any(is_equal(value, known) for known in known_values)

    def get_repr(self, char_limit: int=10) -> List[str]:
        """" Returns a representative list of strings of values of groups. """

        # initiating list of group representation
        repr: List[str] = []

        # iterating over each group
        for group in self:

            # accessing group's values
            values = self.get(group)

            if len(values) == 0:  # case 0: there are no value in this group
                pass
            
            elif len(values) == 1:  # case 1: there is only one value in this group
                repr += [values[0]]
            
            elif len(values) == 2:  # case 2: two values in this group
                repr += [f'{values[1]}'[:char_limit]+' and '+f'{values[0]}'[:char_limit]]
            
            elif len(values) > 2:  # case 3: more than two values in this group
                repr += [f'{values[-1]}'[:char_limit]+' to '+f'{values[0]}'[:char_limit]]
                
        return repr


class GroupedListDiscretizer(BaseEstimator, TransformerMixin):
    
    def __init__(
            self,
            values_orders: Dict[str, Any],
            *,
            copy: bool=False,
            output: type= float,
            str_nan: str=None,
            verbose: bool=False
        ) -> None:
        
        self.features = list(values_orders.keys())
        self.values_orders = {k: GroupedList(v) for k, v in values_orders.items()}
        self.copy = copy
        self.output = output
        self.verbose = verbose
        self.str_nan = str_nan
        
    def fit(self, X, y=None) -> None:

        return self
    
    def transform(self, X: DataFrame, y: Series=None) -> DataFrame:

        # copying dataframes
        Xc = X
        if self.copy:
            Xc = X.copy()

        # filling up nans with specified value
        if self.str_nan:
            Xc[self.features] = Xc[self.features].fillna(self.str_nan)

        # iterating over each feature
        for n, feature in enumerate(self.features):
            
            # verbose if requested
            if self.verbose: 
                print(f" - [GroupedListDiscretizer] Transform {feature} ({n+1}/{len(self.features)})")
            
            # bucketizing feature
            order = self.values_orders.get(feature)  # récupération des groupes
            to_discard = [order.get(group) for group in order]  # identification des valeur à regrouper
            to_input = [Xc[feature].isin(discarded) for discarded in to_discard]  # identifying main bucket value
            to_keep = [n if self.output == float else group for n, group in enumerate(order)]  # récupération du groupe dans lequel regrouper

            # case when there are no values
            if len(to_input)==0 & len(to_keep)==0:
                pass

            # grouping modalities
            else:
                Xc[feature] = select(to_input, to_keep, default=Xc[feature])

        # converting to float
        if self.output == float:
            Xc[self.features] = Xc[self.features].astype(float32)

        return Xc


class QualitativeDiscretizer(BaseEstimator, TransformerMixin):
    """ Automatic discretizing of categorical and categorical ordinal features.

    Modalities/values of features are grouped according to there respective orders:
     - [Qualitative features] order based on modality target rate.
     - [Qualitative ordinal features] user-specified order.

    TODO: pass ordinal_features/qualitati_features as parameters to be able to pass values_orders with other orders (ex: from chaineddiscretizer)

    Parameters
    ----------
    features: list
        Contains qualitative (categorical and categorical ordinal) features to be discretized.

    min_freq: int
        [Qualitative features] Minimal frequency of a modality.
         - NaNs are considered a specific modality but will not be grouped.
         - [Qualitative features] Less frequent modalities are grouped in the `__OTHER__` modality.
         - [Qualitative ordinal features] Less frequent modalities are grouped to the closest modality 
        (smallest frequency or closest target rate), between the superior and inferior values (specified
        in the `values_orders` dictionnary).
        Recommandation: `min_freq` should be set from 0.01 (preciser) to 0.05 (faster, increased stability).

    values_orders: dict, default {}
        [Qualitative ordinal features] dict of features values and list of orders of their values.
         - [Qualitative ordinal features] Less frequent modalities are grouped to the closest modality 
        (smallest frequency or closest target rate), between the superior and inferior values (described
        by the `values_orders`).
        Exemple: for an `age` feature, `values_orders` could be `{'age': ['0-18', '18-30', '30-50', '50+']}`.
    """
    
    def __init__(
            self,
            features: List[str],
            min_freq: float,
            *,
            values_orders: Dict[str, Any]={},
            copy: bool=False,
            verbose: bool=False
        ) -> None:
    
        self.features = features[:]
        self.values_orders = {k: GroupedList(v) for k, v in values_orders.items()}
        self.ordinal_features = [f for f in values_orders if f in features]  # ignores non qualitative features
        self.non_ordinal_features = [f for f in features if f not in self.ordinal_features]
        self.min_freq = min_freq
        self.pipe: List[BaseEstimator] = []
        self.copy = copy
        self.verbose = verbose
        
    def prepare_data(self, X: DataFrame, y: Series) -> DataFrame:
        """ Checking data for bucketization"""

        # copying dataframe
        Xc = X.copy()

        # checking for quantitative columns
        is_object = Xc[self.features].dtypes.apply(is_string_dtype)
        if not all(is_object):  # non qualitative features detected

            if self.verbose:
                print(f"""Non-string features: {', '.join(is_object[~is_object].index)}, will be converted using Converters.StringConverter.""")

            # converting specified features into qualitative features
            stringer = StringConverter(features=self.features)
            Xc = stringer.fit_transform(Xc)

            # append the string converter to the feature engineering pipeline
            self.pipe += [('StringConverter', stringer)]

        # checking for binary target
        y_values = unique(y)
        assert (0 in y_values) & (1 in y_values), "y must be a binary Series (int or float, not object)"
        assert len(y_values) == 2, "y must be a binary Series (int or float, not object)"

        # checking that all unique values in X are in values_orders
        uniques = Xc[self.features].apply(nan_unique)
        for feature in self.ordinal_features:
            missing = [val for val in uniques[feature] if val not in self.values_orders[feature]]
            assert len(missing)==0, f"The ordering for {', '.join(missing)} of feature '{feature}' must be specified in values_orders (str-only)."

        return Xc

    def fit(self, X: DataFrame, y: Series) -> None:
        """ Learning TRAIN distribution"""
        
        # checking data before bucketization
        Xc = self.prepare_data(X, y)

        # [Qualitative ordinal features] Grouping rare values into closest common one
        if len(self.ordinal_features) > 0:

            # discretizing
            ordinal_orders = {k: GroupedList(v) for k, v in self.values_orders.items() if k in self.ordinal_features}
            discretizer = ClosestDiscretizer(
                ordinal_orders, min_freq=self.min_freq, verbose=self.verbose
            )
            discretizer.fit(Xc, y)

            # storing results
            self.values_orders.update(discretizer.values_orders)  # adding orders of grouped features
            self.pipe += [('QualitativeClosestDiscretizer', discretizer)]  # adding discretizer to pipe

        # [Qualitative non-ordinal features] Grouping rare values into default_value '__OTHER__'
        if len(self.non_ordinal_features) > 0:

            # Grouping rare modalities
            discretizer = DefaultDiscretizer(self.non_ordinal_features, min_freq=self.min_freq, 
                                             values_orders=self.values_orders,
                                             verbose=self.verbose)
            discretizer.fit(Xc, y)

            # storing results
            self.values_orders.update(discretizer.values_orders)  # adding orders of grouped features
            self.pipe += [('DefaultDiscretizer', discretizer)]  # adding discretizer to pipe

        return self

    def transform(self, X: DataFrame, y: Series=None) -> DataFrame:
        """ Applying learned bucketization on TRAIN and/or TEST"""

        # copying dataframe if requested
        Xc = X
        if self.copy:
            Xc = X.copy()

        # iterating over each transformer
        for _, step in self.pipe:
            Xc = step.transform(Xc)

        return Xc

class QuantitativeDiscretizer(BaseEstimator, TransformerMixin):
    """ Automatic discretizing of continuous features.

    Modalities/values of features are grouped according to there respective orders:
     - [Quantitative features] real order of the values.

    Parameters
    ----------
    features: list
        Contains quantitative (continuous) features to be discretized.

    q: int, default None
        [Quantitative features] Number of quantiles to initialy cut the feature.
         - NaNs are considered a specific value but will not be grouped.
         - Values more frequent than `1/q` will be set as their own group and remaining frequency will be
        cut into proportionaly less quantiles (`q:=max(round(non_frequent * q), 1)`). 
        Exemple: if q=10 and the value numpy.nan represent 50 % of the observed values, non-NaNs will be 
        cut in q=5 quantiles.
        Recommandation: `q` should be set from 10 (faster) to 20 (preciser).

    """
    
    def __init__(
            self,
            features: List[str],
            q: int,
            *,
            values_orders: Dict[str, Any]={},
            copy: bool=False,
            verbose: bool=False
        ) -> None:
        
        self.features = features[:]
        self.values_orders = {k: GroupedList(v) for k, v in values_orders.items()}
        self.q = q
        self.pipe: List[BaseEstimator] = []
        self.copy = copy
        self.verbose = verbose
    
    def prepare_data(self, X: DataFrame, y: Series) -> DataFrame:
        """ Checking data for bucketization"""
        
        # checking for quantitative columns
        is_numeric = X[self.features].dtypes.apply(is_numeric_dtype)
        assert all(is_numeric), f"Non-numeric features: {', '.join(is_numeric[~is_numeric].index)}"
        
        # checking for binary target
        y_values = unique(y)
        assert (0 in y_values) & (1 in y_values), "y must be a binary Series (int or float, not object)"
        assert len(y_values) == 2, "y must be a binary Series (int or float, not object)"
        
        # copying dataframe
        Xc = X.copy()
        
        return Xc

    def fit(self, X: DataFrame, y: Series) -> None:
        """ Learning TRAIN distribution"""
        
        # checking data before bucketization
        Xc = self.prepare_data(X, y)

        # [Quantitative features] Grouping values into quantiles
        discretizer = QuantileDiscretizer(self.features, q=self.q, values_orders=self.values_orders, verbose=self.verbose)
        Xc = discretizer.fit_transform(Xc, y)

        # storing results
        self.values_orders.update(discretizer.values_orders)  # adding orders of grouped features
        self.pipe += [('QuantileDiscretizer', discretizer)]  # adding discretizer to pipe

        # [Quantitative features] Grouping rare quantiles into closest common one
        #  -> can exist because of overrepresented values (values more frequent than 1/q)
        # searching for features with rare quantiles: computing min frequency per feature
        frequencies = Xc[self.features].apply(lambda u: min_value_counts(u, self.values_orders[u.name]), axis=0)
        
        # minimal frequency of a quantile
        q_min_freq = 1 / self.q / 2
        
        # identifying features that have rare modalities
        has_rare = list(frequencies[frequencies <= q_min_freq].index)
        
        # Grouping rare modalities
        if len(has_rare) > 0:
            
            # Grouping only features with rare modalities 
            rare_values_orders = {feature: order for feature, order in self.values_orders.items() if feature in has_rare}
            discretizer = ClosestDiscretizer(rare_values_orders, min_freq=q_min_freq, verbose=self.verbose)
            discretizer.fit(Xc, y)

            # storing results
            self.values_orders.update(discretizer.values_orders)  # adding orders of grouped features
            self.pipe += [('QuantitativeClosestDiscretizer', discretizer)]  # adding discretizer to pipe

        return self

    def transform(self, X: DataFrame, y: Series=None) -> DataFrame:
        """ Applying learned bucketization on TRAIN and/or TEST"""

        # copying dataframe if requested
        Xc = X
        if self.copy:
            Xc = X.copy()

        # iterating over each transformer
        for _, step in self.pipe:
            Xc = step.transform(Xc)

        return Xc

class Discretizer(BaseEstimator, TransformerMixin):
    """ Automatic discretizing of continuous, categorical and categorical ordinal features.

    Modalities/values of features are grouped according to there respective orders:
     - [Qualitative features] order based on modality target rate.
     - [Qualitative ordinal features] user-specified order.
     - [Quantitative features] real order of the values.

    Parameters
    ----------
    quanti_features: list
        Contains quantitative (continuous) features to be discretized.

    quali_features: list
        Contains qualitative (categorical and categorical ordinal) features to be discretized.

    min_freq: int, default None
        [Qualitative features] Minimal frequency of a modality.
         - NaNs are considered a specific modality but will not be grouped.
         - [Qualitative features] Less frequent modalities are grouped in the `__OTHER__` modality.
         - [Qualitative ordinal features] Less frequent modalities are grouped to the closest modality 
        (smallest frequency or closest target rate), between the superior and inferior values (specified
        in the `values_orders` dictionnary).
        Recommandation: `min_freq` should be set from 0.01 (preciser) to 0.05 (faster, increased stability).

    q: int, default None
        [Quantitative features] Number of quantiles to initialy cut the feature.
         - NaNs are considered a specific value but will not be grouped.
         - Values more frequent than `1/q` will be set as their own group and remaining frequency will be
        cut into proportionaly less quantiles (`q:=max(round(non_frequent * q), 1)`). 
        Exemple: if q=10 and the value numpy.nan represent 50 % of the observed values, non-NaNs will be 
        cut in q=5 quantiles.
        Recommandation: `q` should be set from 10 (faster) to 20 (preciser).

    values_orders: dict, default {}
        [Qualitative ordinal features] dict of features values and list of orders of their values.
         - [Qualitative ordinal features] Less frequent modalities are grouped to the closest modality 
        (smallest frequency or closest target rate), between the superior and inferior values (described
        by the `values_orders`).
        Exemple: for an `age` feature, `values_orders` could be `{'age': ['0-18', '18-30', '30-50', '50+']}`.
    """

    def __init__(
            self,
            quanti_features: List[str],
            quali_features: List[str],
            min_freq: float,
            *,
            values_orders: Dict[str, Any]={},
            copy: bool=False,
            verbose: bool=False
        ) -> None:

        self.features = quanti_features[:] + quali_features[:]
        self.quanti_features = quanti_features[:]
        assert len(list(set(quanti_features))) == len(quanti_features), "Column duplicates in quanti_features"
        self.quali_features = quali_features[:]
        assert len(list(set(quali_features))) == len(quali_features), "Column duplicates in quali_features"
        self.values_orders = {k: GroupedList(v) for k, v in values_orders.items()}
        self.min_freq = min_freq
        self.q = int(1 / min_freq)  # number of quantiles
        self.pipe: List[BaseEstimator] = []
        self.copy = copy
        self.verbose = verbose

    def fit(self, X: DataFrame, y: Series) -> None:

        # [Qualitative features] Grouping qualitative features
        if len(self.quali_features) > 0:

            # verbose if requested
            if self.verbose:
                print("\n---\n[Discretizer] Fit Qualitative Features")

            # grouping qualitative features
            discretizer = QualitativeDiscretizer(
                self.quali_features, min_freq=self.min_freq,
                values_orders=self.values_orders, copy=self.copy,
                verbose=self.verbose
            )
            discretizer.fit(X, y)

            # storing results
            self.values_orders.update(discretizer.values_orders)  # adding orders of grouped features
            self.pipe += discretizer.pipe  # adding discretizer to pipe

        # [Quantitative features] Grouping quantitative features
        if len(self.quanti_features) > 0:

            # verbose if requested
            if self.verbose:
                print("\n---\n[Discretizer] Fit Quantitative Features")

            # grouping quantitative features
            discretizer = QuantitativeDiscretizer(
                self.quanti_features, q=self.q,
                values_orders=self.values_orders, copy=self.copy,
                verbose=self.verbose
            )
            discretizer.fit(X, y)

            # storing results
            self.values_orders.update(discretizer.values_orders)  # adding orders of grouped features
            self.pipe += discretizer.pipe  # adding discretizer to pipe

        return self

    def transform(self, X: DataFrame, y: Series=None) -> DataFrame:

        # verbose if requested
        if self.verbose:
            print("\n---\n[Discretizer] Transform Features")
        
        # copying dataframe if requested
        Xc = X
        if self.copy:
            Xc = X.copy()

        # iterating over each transformer
        for _, step in self.pipe:
            Xc = step.transform(Xc)

        return Xc



class ChainedDiscretizer(GroupedListDiscretizer):
    
    def __init__(
            self,
            features: List[str],
            min_freq: float,
            chained_orders: List[GroupedList],
            *,
            remove_unknown: bool=False,
            str_nan: str='__NAN__',
            copy: bool=False,
            verbose: bool=False
        ) -> None:       
        
        self.min_freq = min_freq
        self.features = features[:]
        self.chained_orders = [GroupedList(order) for order in chained_orders]
        self.copy = copy
        self.verbose = verbose

        # parameters to handle missing/unknown values
        self.remove_unknown = remove_unknown
        self.str_nan = str_nan
        
        # initiating features' values orders to all possible values
        self.known_values = list(set([v for o in self.chained_orders for v in o.values()]))
        self.values_orders = {f: GroupedList(self.known_values[:] + [self.str_nan]) for f in self.features}

    def fit(self, X: DataFrame, y: Series=None) -> None:
        
        # filling nans
        Xc = X[self.features].fillna(self.str_nan)
        
        # iterating over each feature
        for n, feature in enumerate(self.features):

            # verbose if requested
            if self.verbose:
                print(f" - [ChainedDiscretizer] Fit {feature} ({n+1}/{len(self.features)})")

            # computing frequencies of each modality
            frequencies = Xc[feature].value_counts(normalize=True)
            values, frequencies = frequencies.index, frequencies.values

            # checking for unknown values (values to present in an order of self.chained_orders)
            missing = [value for value in values if notna(value) and (value not in self.known_values)]

            # converting unknown values to NaN
            if self.remove_unknown & (len(missing) > 0):

                # alerting user
                print(f"Order for {feature} was not provided for values: {missing}, these values will be converted to '{self.str_nan}' (policy remove_unknown=True)")

                # adding missing valyes to the order
                order = self.values_orders.get(feature)
                order.update({self.str_nan: missing + order.get(self.str_nan)})

            # alerting user
            else:
                assert not len(missing) > 0, f"Order for {feature} needs to be provided for values: {missing}, otherwise set remove_unknown=True"

            # iterating over each specified orders
            for order in self.chained_orders:
                
                # values that are frequent enough
                to_keep = list(values[frequencies >= self.min_freq])

                # all values from the order 
                values_order = [o for v in order for o in order.get(v)]

                # values from the order to group (not frequent enough or absent)
                to_discard = [value for value in values_order if value not in to_keep]

                # values to group into discarded values
                value_to_group = [order.get_group(value) for value in to_discard]

                # values of the series to input
                df_to_input = [Xc[feature] == discarded for discarded in to_discard]  # identifying observation to input

                # inputing non frequent values
                Xc[feature] = select(df_to_input, value_to_group, default=Xc[feature])
                
                # historizing in the feature's order
                for discarded, kept in zip(to_discard, value_to_group):
                    self.values_orders.get(feature).group(discarded, kept)
                    
                # updating frequencies of each modality for the next ordering
                frequencies = Xc[feature].value_counts(dropna=False, normalize=True).drop(nan, errors='ignore')  # dropping nans to keep them anyways
                values, frequencies = frequencies.index, frequencies.values
        
        super().__init__(self.values_orders, str_nan=self.str_nan, copy=self.copy, output=str)
        super().fit(X, y)
            
        return self

class QuantileDiscretizer(BaseEstimator, TransformerMixin):
    
    def __init__(self, features: List[str], q: int, *, 
    	         values_orders: Dict[str, Any]={},
                 copy: bool=False, verbose: bool=False) -> None:
        """ Discretizes quantitative features into groups of q quantiles"""
        
        self.features = features[:]
        self.q = q
        self.values_orders = {k: GroupedList(v) for k, v in values_orders.items()}
        self.quantiles: Dict[str, Any] = {}
        self.copy = copy
        self.verbose = verbose

    def fit(self, X: DataFrame, y: Series=None) -> None:
        
        # computing quantiles for the feature
        self.quantiles = X[self.features].apply(find_quantiles, q=self.q, axis=0)

        # case when only one feature is discretized
        if len(self.features) == 1:
        	self.quantiles = {self.features[0]: list(self.quantiles.get(self.features[0]).values)}

        # building string of values to be displayed
        values: List[str] = []
        for n, feature in enumerate(self.features):

            # verbose
            if self.verbose:
                print(f" - [QuantileDiscretizer] Fit {feature} ({n+1}/{len(self.features)})")

            # quantiles as strings
            feature_quantiles = self.quantiles.get(feature)
            str_values = ['<= ' + q for q in format_list(feature_quantiles)]

            # case when there is only one value
            if len(feature_quantiles) == 0:
                str_values = ['non-nan']

            # last quantile is between the last value and inf
            else:
                str_values = str_values + [str_values[-1].replace('<= ', '>  ')]
            values += [str_values]
        
        # adding inf for the last quantiles
        self.quantiles = {f: q + [inf] for f, q in self.quantiles.items()}

        # creating the values orders based on quantiles
        self.values_orders.update({feature: GroupedList(str_values) for feature, str_values in zip(self.features, values)})

        return self
    
    def transform(self, X: DataFrame, y: Series=None) -> DataFrame:
        
        # copying dataset if requested
        Xc = X
        if self.copy:
            Xc = X.copy()
        
        # iterating over each feature
        for n, feature in enumerate(self.features):

            # verbose if requested
            if self.verbose:
                print(f" - [QuantileDiscretizer] Transform {feature} ({n+1}/{len(self.features)})")

            nans = isna(Xc[feature])  # keeping track of nans

            # grouping values inside quantiles
            to_input = [Xc[feature] <= q for q in self.quantiles.get(feature)]  # values that will be imputed
            values = [[v] * len(X) for v in self.values_orders.get(feature)]  # new values to imput
            Xc[feature] = select(to_input, values, default=Xc[feature])  # grouping modalities

            # adding back nans
            if any(nans):
                Xc.loc[nans, feature] = nan
        
        return Xc

class ClosestDiscretizer(BaseEstimator, TransformerMixin):
    
    def __init__(self, values_orders: Dict[str, Any], min_freq: float, *, default: str='worst', copy: bool=False, verbose: bool=False):
        """ Discretizes ordered qualitative features into groups more frequent than min_freq"""
        
        self.features = list(values_orders.keys())
        self.min_freq = min_freq
        self.values_orders = {k: GroupedList(v) for k, v in values_orders.items()}
        self.default = default[:]
        self.default_values: Dict[str, Any] = {}
        self.copy = copy
        self.verbose = verbose

    def fit(self, X: DataFrame, y: Series) -> None:
        
        # verbose
        if self.verbose:
            print(f" - [ClosestDiscretizer] Fit {', '.join(self.features)}")
        
        # grouping rare modalities for each feature
        common_modalities = X[self.features].apply(
            lambda u: find_common_modalities(u, y, self.min_freq, self.values_orders.get(u.name)), 
            axis=0, result_type='expand'
        ).T.to_dict()

        # updating the order per feature
        self.values_orders.update({f: common_modalities.get('order').get(f) for f in self.features})

        # defining the default value based on the strategy
        self.default_values = {f: common_modalities.get(self.default).get(f) for f in self.features}

        return self
    
    def transform(self, X: DataFrame, y: Series=None) -> DataFrame:
        
        # copying dataset if requested
        Xc = X
        if self.copy:
            Xc = X.copy()

        # iterating over each feature
        for n, feature in enumerate(self.features):

            # printing verbose if requested
            if self.verbose:
                print(f" - [ClosestDiscretizer] Transform {feature} ({n+1}/{len(self.features)})")

            # accessing feature's modalities' order
            order = self.values_orders.get(feature)

            # imputation des valeurs inconnues le cas échéant
            unknowns = [value for value in Xc[feature].unique() if not any(is_equal(value, known) for known in order.values())]
            unknowns = [value for value in unknowns if notna(value)]  # suppression des NaNs
            if any(unknowns):
                to_input = [Xc[feature] == unknown for unknown in unknowns]  # identification des valeurs à regrouper
                Xc[feature] = select(to_input, [self.default_values.get(feature)], default=Xc[feature])  # regroupement des valeurs
                warn(f"Unknown modalities provided for {feature}: {unknowns}")

            # grouping values inside groups of modalities
            to_discard = [order.get(group) for group in order]  # identification des valeur à regrouper
            to_input = [Xc[feature].isin(discarded) for discarded in to_discard]  # identification des valeurs à regrouper
            Xc[feature] = select(to_input, order, default=Xc[feature])  # regroupement des valeurs

        return Xc

def min_value_counts(x: Series, order: List[Any]) -> float:
    """ Minimum of modalities' frequencies. """

    # modality frequency
    values = x.value_counts(dropna=False, normalize=True)
    
    # ignoring NaNs
    values = values.drop(nan, errors='ignore')
    
    # adding missing modalities
    values = values.reindex(order).fillna(0)
    
    # minimal frequency 
    min_values = values.values.min()

    return min_values

def value_counts(x: Series, dropna: bool=False, normalize: bool=True) -> dict:
    """ Counts the values of each modality of a series into a dictionnary"""
    
    values = x.value_counts(dropna=dropna, normalize=normalize)
    
    return values.to_dict()

def target_rate(x: Series, y: Series, dropna: bool=True, ascending=True) -> dict:
    """ Target y rate per modality of x into a dictionnary"""
    
    rates = y.groupby(x, dropna=dropna).mean().sort_values(ascending=ascending)
    
    return rates.to_dict()

def nunique(x: Series, dropna=False) -> int:
    """ Computes number of unique modalities"""
    
    uniques = unique(x)
    n = len(uniques)
    
    # removing nans
    if dropna:
        if any(isna(uniques)):
            n -= 1
    
    return n

class DefaultDiscretizer(BaseEstimator, TransformerMixin):
    
    def __init__(
        self, features: List[str], min_freq: float, *, 
        values_orders: Dict[str, Any]={},
        default_value: str='__OTHER__',
        str_nan: str='__NAN__',
        copy: bool=False, verbose: bool=False) -> None:
        """ Groups a qualitative features' values less frequent than min_freq into a default_value"""
        
        self.features = features[:]
        self.min_freq = min_freq
        self.values_orders = {k: GroupedList(v) for k, v in values_orders.items()}
        self.default_value = default_value[:]
        self.str_nan = str_nan[:]
        self.copy = copy
        self.verbose = verbose

    def fit(self, X: DataFrame, y: Series) -> None:

        # filling up NaNs
        Xc = X[self.features].fillna(self.str_nan)

        # computing frequencies of each modality
        frequencies = Xc.apply(value_counts, normalize=True, axis=0)

        # computing ordered target rate per modality for ordering
        target_rates = Xc.apply(target_rate, y=y, ascending=True, axis=0)

        # attributing orders to each feature
        self.values_orders.update({f: GroupedList(list(target_rates[f])) for f in self.features})
        
        # number of unique modality per feature
        nuniques = Xc.apply(nunique, axis=0)
            
        # identifying modalities which are the most common
        self.to_keep: Dict[str, Any] = {}  # dict of features and corresponding kept modalities

        # iterating over each feature
        for feature in self.features:

            # checking for binary features
            if nuniques[feature] > 2:
                kept = [val for val, freq in frequencies[feature].items() if freq >= self.min_freq]

            # keeping all modalities of binary features
            else:
                kept = [val for val, freq in frequencies[feature].items()]

            self.to_keep.update({feature: kept})

        # grouping rare modalities 
        for n, feature in enumerate(self.features):

        	# printing verbose
            if self.verbose:
                print(f" - [DefaultDiscretizer] Fit {feature} ({n+1}/{len(self.features)})")

            # identifying values to discard (rare modalities)
            to_discard = [value for value in self.values_orders[feature] if value not in self.to_keep[feature]]

            # discarding rare modalities
            if len(to_discard) > 0:

                # adding default_value to possible values
                order = self.values_orders[feature]
                order.append(self.default_value)

                # grouping rare modalities into default_value
                order.group_list(to_discard, self.default_value)

                # computing target rate for default value and reordering values according to feature's target rate
                default_target_rate = y.loc[X[feature].isin(order.get(self.default_value))].mean()  # computing target rate for default value
                order_target_rate = [target_rates.get(feature).get(value) for value in order if value != self.default_value]
                default_position = next(n for n, trate in enumerate(order_target_rate + [inf]) if trate > default_target_rate)

                # updating the modalities' order                
                new_order = order[:-1][:default_position] + [self.default_value] + order[:-1][default_position:]  # getting rid of default value already in order
                order = order.sort_by(new_order)
                self.values_orders.update({feature: order})

        return self

    def transform(self, X: DataFrame, y: Series=None) -> DataFrame:

        # copying dataset if requested
        Xc = X
        if self.copy:
            Xc = X.copy()

        # filling up NaNs
        Xc[self.features] = Xc[self.features].fillna(self.str_nan)

        # grouping values inside groups of modalities
        for n, feature in enumerate(self.features):

            # verbose if requested
            if self.verbose: 
                print(f" - [DefaultDiscretizer] Transform {feature} ({n+1}/{len(self.features)})")

            # grouping modalities
            Xc[feature] = select([~Xc[feature].isin(self.to_keep[feature])], [self.default_value], default=Xc[feature])

        return Xc



def find_quantiles(df_feature: Series, q: int, len_df: int=None, quantiles: List[float]=None) -> List[float]:
    """ Découpage en quantile de la feature.
    
    Fonction récursive : on prend l'échantillon et on cherche des valeurs sur-représentées
    Si la valeur existe on la met dans une classe et on cherche à gauche et à droite de celle-ci, d'autres variables sur-représentées
    Une fois il n'y a plus de valeurs qui soient sur-représentées,
    on fait un découpage classique avec qcut en multipliant le nombre de classes souhaité par le pourcentage de valeurs restantes sur le total
    
    """

    # initialisation de la taille total du dataframe
    if len_df is None:
        len_df = len(df_feature)
    
    # initialisation de la liste des quantiles
    if quantiles is None:
        quantiles = []

    # calcul du nombre d'occurences de chaque valeur
    frequencies = df_feature.value_counts(dropna=False, normalize=False).drop(nan, errors='ignore') / len_df # dropping nans to keep them anyways
    values, frequencies = array(frequencies.index), array(frequencies.values)
    
    # cas 1 : il n'y a pas d'observation dans le dataframe
    if len(df_feature) == 0:
        
        return quantiles
    
    # cas 2 : il y a des valeurs manquantes NaN
    elif any(isna(df_feature)):
        
        return find_quantiles(df_feature[notna(df_feature)], q, len_df=len_df, quantiles=quantiles)
        
    # cas 2 : il n'y a que des valeurs dans le dataframe (pas de NaN)
    else:
        
        # cas 1 : il existe une valeur sur-représentée
        if any(frequencies > 1 / q):

            # identification de la valeur sur-représentée
            frequent_value = values[frequencies.argmax()]
            
            # ajout de la valeur fréquente à la liste des quantiles
            quantiles += [frequent_value]

            # calcul des quantiles pour les parties inférieures et supérieures
            quantiles_inf = find_quantiles(df_feature[df_feature < frequent_value], q, len_df=len_df)
            quantiles_sup = find_quantiles(df_feature[df_feature > frequent_value], q, len_df=len_df)
            
            return quantiles_inf + quantiles + quantiles_sup

        # cas 2 : il n'existe pas de valeur sur-représentée
        else:
            
            # nouveau nombre de quantile en prenant en compte les classes déjà constituées
            new_q = max(round(len(df_feature) / len_df * q), 1)
            
            # calcul des quantiles sur le dataframe
            if new_q > 1:
                quantiles += list(quantile(df_feature.values, linspace(0, 1, new_q + 1)[1:-1], interpolation='lower'))

            # case when there are no enough observations to compute quantiles
            else:
                quantiles += [max(df_feature.values)]

            return quantiles

def is_equal(a: Any, b: Any) -> bool:
    """ checks if a and b are equal (NaN insensitive)"""
    
    # default equality
    equal = a == b
    
    # Case where a and b are NaNs
    if isna(a) and isna(b):
        equal = True
    
    return equal

def find_common_modalities(df_feature: Series, y: Series, min_freq: float, order: GroupedList, len_df: int=None) -> Dict[str, Any]:
    """ Découpage en modalités de fréquence minimal (Cas des variables ordonnées).
    
    Fonction récursive : on prend l'échantillon et on cherche des valeurs sur-représentées
    Si la valeur existe on la met dans une classe et on cherche à gauche et à droite de celle-ci, d'autres variables sur-représentées
    Une fois il n'y a plus de valeurs qui soient sur-représentées,
    on fait un découpage classique avec qcut en multipliant le nombre de classes souhaité par le pourcentage de valeurs restantes sur le total
    
    """

    # initialisation de la taille totale du dataframe
    if len_df is None:
        len_df = len(df_feature)
    
    # conversion en GroupedList si ce n'est pas le cas 
    order = GroupedList(order)
    
    # cas 1 : il y a des valeurs manquantes NaN
    if any(isna(df_feature)):
        
        return find_common_modalities(df_feature[notna(df_feature)], y[notna(df_feature)], min_freq, order, len_df)


    # cas 2 : il n'y a que des valeurs dans le dataframe (pas de NaN)
    else:
        
        # computing frequencies and target rate of each modality
        init_frequencies = df_feature.value_counts(dropna=False, normalize=False) / len_df
        init_values, init_frequencies = init_frequencies.index, init_frequencies.values
        
        # ordering
        frequencies = [init_frequencies[init_values == value][0] if any(init_values == value) else 0 for value in order]  # sort selon l'ordre des modalités
        values = [init_values[init_values == value][0] if any(init_values == value) else value for value in order]  # sort selon l'ordre des modalités
        target_rate = y.groupby(df_feature).sum().reindex(order) / frequencies / len_df # target rate per modality
        underrepresented = [value for value, frequency in zip(values, frequencies) if frequency < min_freq]  # valeur peu fréquentes

        # cas 1 : il existe une valeur sous-représentée
        while any(underrepresented) & (len(frequencies) > 1):

            # identification de la valeur sous-représentée
            discarded_idx = argmin(frequencies)
            discarded = values[discarded_idx]

            # identification de la modalité la plus proche (volume et taux de défaut)
            kept = find_closest_modality(discarded, discarded_idx, list(zip(order, frequencies, target_rate)), min_freq)

            # removing the value from the initial ordering
            order.group(discarded, kept)
            
            # ordering
            frequencies = [init_frequencies[init_values == value][0] if any(init_values == value) else 0 for value in order]  # sort selon l'ordre des modalités
            values = [init_values[init_values == value][0] if any(init_values == value) else value for value in order]  # sort selon l'ordre des modalités
            target_rate = y.groupby(df_feature).sum().reindex(order) / frequencies / len_df # target rate per modality
            underrepresented = [value for value, frequency in zip(values, frequencies) if frequency < min_freq]  # valeur peu fréquentes

        worst, best = target_rate.idxmin(), target_rate.idxmax()

        # cas 2 : il n'existe pas de valeur sous-représentée
        return {'order': order, 'worst': worst, 'best': best}

def find_closest_modality(value, idx: int, freq_target: Series, min_freq: float) -> int:
    """HELPER Finds the closest modality in terms of frequency and target rate"""

    # case 1: lowest modality
    if idx == 0:
        closest_modality = 1
        
    # case 2: biggest modality
    elif idx == len(freq_target) - 1:
        closest_modality = len(freq_target) - 2
    
    # case 3: not the lowwest nor the biggest modality
    else:
        # previous modality's volume and target rate
        previous_value, previous_volume, previous_target = freq_target[idx - 1]
        
        # current modality's volume and target rate
        _, volume, target = freq_target[idx]
        
        # next modality's volume and target rate
        next_value, next_volume, next_target = freq_target[idx + 1]
        
        # regroupement par volumétrie (préféré s'il n'y a pas beaucoup de volume)
        # case 1: la modalité suivante est inférieure à min_freq 
        if next_volume < min_freq <= previous_volume:
            closest_modality = idx + 1
            
        # case 2: la modalité précédante est inférieure à min_freq 
        elif previous_volume < min_freq <= next_volume:
            closest_modality = idx - 1
            
        # case 3: les deux modalités sont inférieures à min_freq 
        elif (previous_volume < min_freq) & (next_volume < min_freq):
            
            # cas 1: la modalité précédante est inférieure à la modalité suivante
            if previous_volume < next_volume:
                closest_modality = idx - 1
            
            # cas 2: la modalité suivante est inférieure à la modalité précédante
            elif next_volume < previous_volume:
                closest_modality = idx + 1
            
            # cas 3: les deux modalités ont la même fréquence
            else:
                
                # cas1: la modalité précédante est plus prorche en taux de cible
                if abs(previous_target - target) <= abs(next_target - target):
                    closest_modality = idx - 1
                    
                # cas2: la modalité précédante est plus prorche en taux de cible
                else:
                    closest_modality = idx + 1
                
        # case 4: les deux modalités sont supérieures à min_freq 
        else:
                
            # cas1: la modalité précédante est plus prorche en taux de cible
            if abs(previous_target - target) <= abs(next_target - target):
                closest_modality = idx - 1

            # cas2: la modalité précédante est plus prorche en taux de cible
            else:
                closest_modality = idx + 1
    
    # récupération de la valeur associée
    closest_value = freq_target[closest_modality][0]
    
    return closest_value


def format_list(a_list: List[float]) -> List[str]:
    """ Formats a list of floats to a list of unique rounded strings of floats"""

    # finding the closest power of thousands for each element
    closest_powers = [next((k for k in range(-3, 4) if abs(elt) / 100 // 1000**(k) < 10)) for elt in a_list]

    # rounding elements to the closest power of thousands
    rounded_to_powers = [elt / 1000**(k) for elt, k in zip(a_list, closest_powers)]

    # computing optimal decimal per unique power of thousands
    optimal_decimals: Dict[str, int] = {}
    for power in unique(closest_powers):  # iterating over each power of thousands found

        # filtering on the specific power of thousands
        sub_array = array([elt for elt, p in zip(rounded_to_powers, closest_powers) if power == p])

        # number of distinct values
        n_uniques = sub_array.shape[0]

        # computing the first rounding decimal that allows for distinction of each values when rounded
        # by default None (no rounding)
        optimal_decimal = next((k for k in range(1, 10) if len(unique(sub_array.round(k))) == n_uniques), None)

        # storing in the dict    
        optimal_decimals.update({
            power: optimal_decimal
        })

    # rounding according to each optimal_decimal
    rounded_list: List[float] = []
    for elt, power in zip(rounded_to_powers, closest_powers):

        # rounding each element
        rounded = elt  # by default None (no rounding)
        optimal_decimal = optimal_decimals.get(power)
        if optimal_decimal:  # checking that the optimal decimal exists
            rounded = round(elt, optimal_decimal)

        # adding the rounded number to the list
        rounded_list += [rounded]

    # dict of suffixes per power of thousands
    suffixes = {
        -3: 'n', -2: 'mi', -1: 'm', 0: '', 1: 'K', 2: 'M', 3: 'B'
    }

    # adding the suffixes
    formatted_list = [f'{elt: 3.3f}{suffixes[power]}' for elt, power in zip(rounded_list, closest_powers)]
    
    # keeping zeros
    formatted_list = [rounded if raw != 0 else f'{raw: 3.3f}' for rounded, raw in zip(formatted_list, a_list)]
    
    return formatted_list
