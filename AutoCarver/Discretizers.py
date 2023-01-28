from IPython.display import display_html
from numpy import sort, nan, inf, float16, where, isin, argsort, array, select, append, quantile, linspace, argmin
from pandas import DataFrame, Series, isna, qcut, notna, unique
from sklearn.base import BaseEstimator, TransformerMixin
from warnings import warn


class GroupedList(list):
    
    def __init__(self, iterable=()):
        """ An ordered list that historizes its elements' merges."""
        
        # case 0: iterable est le dictionnaire contained
        if isinstance(iterable, dict):
            
            # récupération des valeurs de la liste (déjà ordonné)
            values = [key for key, _ in iterable.items()]

            # initialsiation de la liste
            super().__init__(values)

            # attribution des valeurs contenues
            self.contained = {k: v for k, v in iterable.items()}
        
        # case 1: s'il ne s'agit pas déjà d'une liste groupée -> création des groupes
        elif isinstance(iterable, GroupedList):

            # initialsiation de la liste
            super().__init__(iterable)

            # copie des groupes
            self.contained = {k: v for k, v in iterable.contained.items()}
        
        # case 2: il s'agit d'une GroupedList -> copie des groupes
        else:

            # initialsiation de la liste
            super().__init__(iterable)

            # création des groupes
            self.contained = {v: [v] for v in iterable}

    def group_list(self, to_discard: list, to_keep: str):
        """ Groups elements to_discard into values to_keep"""

        for discarded, kept in zip(to_discard, [to_keep] * len(to_discard)):
            self.group(discarded, kept)

    def group(self, discarded, kept):
        """ Groups the discarded value with the kept value"""

        if not is_equal(discarded, kept):

            assert discarded in self, f"{discarded} not in list"
            assert kept in self, f"{kept} not in list"

            contained_discarded = self.contained.get(discarded)
            contained_kept = self.contained.get(kept)

            self.contained.update({
                kept: contained_discarded + contained_kept,
                discarded: []
            })

            self.remove(discarded)
        
        return self
        
    def append(self, new_value):
        """ Appends a new_value to the GroupedList"""
        
        self += [new_value]
        
        self.contained.update({new_value: [new_value]})
        
        return self

    def sort(self):
        """ Sorts the values of the list and dict, if any, NaNs are the last. """

        # str values
        keys_str = [key for key in self if isinstance(key, str)]

        # non-str values
        keys_float = [key for key in self if not isinstance(key, str)]

        # sorting and merging keys
        keys = list(sort(keys_str)) + list(sort(keys_float)) 

        # recreating an ordered GroupedList
        self = GroupedList({key: self.get(key) for key in keys})

        return self

    def sort_by(self, ordering):
        """ Sorts the values of the list and dict, if any, NaNs are the last. """

        # recreating an ordered GroupedList
        self = GroupedList({k: self.get(k) for k in ordering})

        return self

    
    def remove(self, value):
        
        super().remove(value)
        self.contained.pop(value)
    
    def pop(self, idx):
        
        value = self[idx]
        self.remove(value)
    
    def get(self, key):
        """ returns list of values contained in key"""

        # default to fing an element
        found = self.contained.get(key)

        # copying with dictionnaries (not working with numpy.nan)
        if isna(key):
            found = [value for dict_key, value in self.contained.items() if is_equal(dict_key, key)][0]

        return found

    def get_group(self, value):
        """ returns the group containing the specified value """
        
        found = [key for key, values in self.contained.items() if any(is_equal(value, elt) for elt in values)]

        if any(found):
            return found[0]

    def values(self):
        """ returns all values contained in each group """

        known = [value for values in self.contained.values() for value in values]

        return known

    def contains(self, value):
        """ checks if a value if contained in any group """

        known_values = self.values()

        return any(is_equal(value, known) for known in known_values)


class Discretizer(BaseEstimator, TransformerMixin):
    """ Automatic discretizing of continuous, categorical and categorical ordinal features.

    Modalities/values of features are grouped according to there respective orders:
     - [Qualitative features] order based on modality target rate.
     - [Qualitative ordinal features] user-specified order.
     - [Quantitative features] real order of the values.

    Parameters
    ----------
    quanti_features: list, default []
        Contains quantitative (continuous) features to be discretized.

    quali_features: list, default []
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
    
    def __init__(self, quanti_features: list=[], quali_features: list=[], q: int=None, min_freq: float=None, values_orders: dict={}, copy: bool=False, verbose: bool=False):
    
        
        self.features = quanti_features[:] + quali_features[:]
        self.quanti_features = quanti_features[:]
        self.quali_features = quali_features[:]
        self.values_orders = {k: GroupedList(v) for k, v in values_orders.items()}
        self.min_freq = min_freq
        self.q = q
        self.pipe = []
        self.copy = copy
        self.verbose = verbose

    def fit(self, X, y):
        
        # copie des features
        Xc = X.copy()

        # [Qualitative ordinal features] Grouping rare values into closest common one
        if any(self.values_orders.keys()):
            if self.verbose: print("\n---\n[Discretizer] Fitting Qualitative Ordinal Features")

            discretizer = ClosestDiscretizer(self.values_orders, min_freq=self.min_freq, verbose=self.verbose)
            discretizer.fit(Xc, y)

            # storing results
            self.values_orders.update(discretizer.values_orders)  # adding orders of grouped features
            self.pipe += [('QualiClosestDiscretizer', discretizer)]  # adding discretizer to pipe

        # [Qualitative non-ordinal features] Grouping rare values into default_value '__OTHER__'
        non_ordinal = [f for f in self.quali_features if f not in self.values_orders.keys()]
        if any(non_ordinal):
            if self.verbose: print("\n---\n[Discretizer] Fitting Qualitative Non-Ordinal Features")

            discretizer = DefaultDiscretizer(non_ordinal, min_freq=self.min_freq, verbose=self.verbose)
            discretizer.fit(Xc, y)

            # storing results
            self.values_orders.update(discretizer.values_orders)  # adding orders of grouped features
            self.pipe += [('DefaultDiscretizer', discretizer)]  # adding discretizer to pipe

        # [Quantitative features] Grouping values into quantiles
        if any(self.quanti_features):
            if self.verbose: print("\n---\n[Discretizer] Fitting Quantitative Features")

            discretizer = QuantileDiscretizer(self.quanti_features, q=self.q, verbose=self.verbose)
            Xc = discretizer.fit_transform(Xc)

            # storing results
            self.values_orders.update(discretizer.values_orders)  # adding orders of grouped features
            self.pipe += [('QuantileDiscretizer', discretizer)]  # adding discretizer to pipe

            # [Quantitative features] Grouping rare quantiles into closest common one 
            #  -> can exist because of overrepresented values (values more frequent than 1/q)
            frequencies = Xc[self.quanti_features].apply(lambda u: u.value_counts(dropna=False, normalize=True).drop(nan, errors='ignore').reindex(self.values_orders.get(u.name)).fillna(0).values.min(), axis=0)  # computing min frequency per quantitative feature
            q_min_freq = 1 / self.q / 2
            has_rare = list(frequencies[frequencies <= q_min_freq].index)
            if any(has_rare):
                if self.verbose: print("\n---\n[Discretizer] Fitting Quantitative Features (that have rare values)")

                rare_values_orders = {feature: order for feature, order in self.values_orders.items() if feature in has_rare}
                discretizer = ClosestDiscretizer(rare_values_orders, min_freq=q_min_freq, verbose=self.verbose)
                discretizer.fit(Xc, y)

                # storing results
                self.values_orders.update(discretizer.values_orders)  # adding orders of grouped features
                self.pipe += [('QuantiClosestDiscretizer', discretizer)]  # adding discretizer to pipe

        return self

    def transform(self, X, y=None):
        if self.verbose: print("\n---\n[Discretizer] Discretizing Features")
        
        # copie des features
        Xc = X
        if self.copy:
            Xc = X.copy()

        # iterating over each transformer
        for _, step in self.pipe:
            Xc = step.transform(Xc)

        return Xc


class GroupedListDiscretizer(BaseEstimator, TransformerMixin):
    
    def __init__(self, values_orders: dict, *, copy: bool=False, output: type= float, verbose: bool=False):
        
        self.features = list(values_orders.keys())
        self.values_orders = {feature: GroupedList(order) for feature, order in values_orders.items()}
        self.copy = copy
        self.output = output
        self.verbose = verbose
        
    def fit(self, X, y=None):

        return self
    
    def transform(self, X, y=None):

        # copying dataframes
        Xc = X
        if self.copy:
            Xc = X.copy()

        # iterating over each feature
        for n, feature in enumerate(self.features):
            if self.verbose: print(f" - [GroupedListDiscretizer] Discretizing {feature} ({n+1}/{len(self.features)})")
            
            order = self.values_orders.get(feature)  # récupération des groupes
            to_discard = [order.get(group) for group in order]  # identification des valeur à regrouper
            to_input = [Xc[feature].isin(discarded) for discarded in to_discard]  # identification des observations à regrouper
            to_keep = [n if self.output == float else group for n, group in enumerate(order)]  # récupération du groupe dans lequel regrouper
            arr_feature = select(to_input, to_keep, default=Xc[feature])  # grouping modalities
            Xc[feature] = arr_feature  # storing grouped feature

        # converting to float
        if self.output == float:
            Xc[self.features] = Xc[self.features].astype(float16)

        return Xc

class ChainedDiscretizer(GroupedListDiscretizer):
    
    def __init__(self, features: list, min_freq: float, chained_orders: list, *, copy: bool=False, verbose: bool=False):       
        
        self.min_freq = min_freq
        self.features = features[:]
        self.chained_orders = [GroupedList(order) for order in chained_orders]
        self.pipe = []
        self.copy = copy
        self.verbose = verbose
        
        # initiating features' values orders to all possible values
        self.known_values = list(set([value for group in self.chained_orders for value in group.values()]))
        self.values_orders = {f: GroupedList(self.known_values[:]) for f in self.features}

    def fit(self, X, y=None):
        
        # copying dataframe
        Xc = X[self.features].copy()
        
        # iterating over each feature
        for n, feature in enumerate(self.features):
            if self.verbose: print(f" - [ChainedDiscretizer] Fitting {feature} ({n+1}/{len(self.features)})")

            # computing frequencies of each modality
            frequencies = Xc[feature].value_counts(dropna=False, normalize=True).drop(nan, errors='ignore')  # dropping nans to keep them anyways
            values, frequencies = frequencies.index, frequencies.values

            # checking for unknown values (values to present in an order of self.chained_orders)
            missing = [value for value in values if notna(value) and (value not in self.known_values)]
            assert not any(missing), f"Order needs to be provided for values: {missing}"

            # iterating over each specified orders
            for order in self.chained_orders:
                
                # identifying modalities which rarest values
                to_keep = values[frequencies >= self.min_freq]

                # grouping rare modalities
                to_discard = [[value for value in order.get(group) if (not value in to_keep)] for group in order]  # identifying rare values
                to_input = [Xc[feature].isin(discarded) for discarded in to_discard]  # identifying observation to input
                arr_feature = select(to_input, to_input, default=Xc[feature])  # regroupement des naf peu fréquents
                Xc[feature] = arr_feature  # storing grouped feature
                
                # historizing in the feature's order
                for discarded, kept in zip(to_discard, order):
                    self.values_orders.get(feature).group_list(discarded, kept)  # historisation dans l'order
                    
                # updating frequencies of each modality for the next ordering
                frequencies = Xc[feature].value_counts(dropna=False, normalize=True).drop(nan, errors='ignore')  # dropping nans to keep them anyways
                values, frequencies = frequencies.index, frequencies.values
        
        super().__init__(self.values_orders, copy=self.copy, output=str)
        super().fit(X, y)
            
        return self

class QuantileDiscretizer(BaseEstimator, TransformerMixin):
    
    def __init__(self, features: list, q: int, copy: bool=False, verbose: bool=False):
        """ Discretizes quantitative features into groups of q quantiles"""
        
        self.features = features[:]
        self.q = q
        self.values_orders = dict()
        self.quantiles = None
        self.copy = copy
        self.verbose = verbose

    def fit(self, X, y=None):
        
        # computing quantiles for the feature
        self.quantiles = X[self.features].apply(find_quantiles, q=self.q, axis=0)

        # case when only one feature is discretized
        if len(self.features) == 1:
        	self.quantiles = {self.features[0]: list(self.quantiles.get(self.features[0]).values)}

        # building string of values to be displayed
        values = []
        for n, feature in enumerate(self.features):
            if self.verbose: print(f" - [QuantileDiscretizer] Fitting {feature} ({n+1}/{len(self.features)})")

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
    
    def transform(self, X, y=None):
        
        # copying dataset if requested
        Xc = X
        if self.copy:
            Xc = X.copy()
        
        # iterating over each feature
        for n, feature in enumerate(self.features):
            if self.verbose: print(f" - [QuantileDiscretizer] Discretizing {feature} ({n+1}/{len(self.features)})")

            arr_feature = Xc[feature].values  # initial feature value
            nans = isna(arr_feature)  # keeping track of nans

            # grouping values inside quantiles
            to_input = [arr_feature <= q for q in self.quantiles.get(feature)]  # values that will be imputed
            values = [[v] * len(X) for v in self.values_orders.get(feature)]  # new values to imput
            arr_feature = select(to_input, values, default=arr_feature)  # grouping modalities

            # adding back nans
            if any(nans):
                Xc.loc[nans, feature] = nan

            Xc[feature] = arr_feature  # storing grouped feature
        
        return Xc

class ClosestDiscretizer(BaseEstimator, TransformerMixin):
    
    def __init__(self, values_orders: dict, min_freq: float, *, default: str='worst', copy: bool=False, verbose: bool=False):
        """ Discretizes ordered qualitative features into groups more frequent than min_freq"""
        
        self.features = list(values_orders.keys())
        self.min_freq = min_freq
        self.values_orders = {k: GroupedList(v) for k, v in values_orders.items()}
        self.default = default[:]
        self.default_values = {}
        self.copy = copy
        self.verbose = verbose

    def fit(self, X, y):
        
        # grouping rare modalities for each feature
        common_modalities = X[self.features].apply(
            lambda u: find_common_modalities(u, y, self.min_freq, self.values_orders.get(u.name)), 
            axis=0, result_type='expand'
        ).T

        # updating the order per feature
        self.values_orders = {f: common_modalities.get(f).get('order') for f in self.features}

        # defining the default value based on the strategy
        self.default_values = {f: common_modalities.get(f).get(self.default) for f in self.features}

        return self
    
    def transform(self, X, y=None):
        
        # copying dataset if requested
        X_c = X
        if self.copy:
            X_c = X.copy()

        # iterating over each feature
        for n, feature in enumerate(self.features):
            if self.verbose: print(f" - [ClosestDiscretizer] Discretizing {feature} ({n+1}/{len(self.features)})")

            # accessing feature's modalities' order
            order = self.values_orders.get(feature)

            # imputation des valeurs inconnues le cas échéant
            unknowns = [value for value in X_c[feature].unique() if not any(is_equal(value, known) for known in order.values())]
            unknowns = [value for value in unknowns if notna(value)]  # suppression des NaNs
            if any(unknowns):
                to_input = [X_c[feature] == unknown for unknown in unknowns]  # identification des valeurs à regrouper
                arr_feature = select(to_input, [self.default_values.get(feature)], default=X_c[feature])  # regroupement des valeurs
                X_c[feature] = arr_feature  # storing grouped feature
                warn(f"Unknown modalities provided for {feature}: {unknowns}")

            # grouping values inside groups of modalities
            to_discard = [order.get(group) for group in order]  # identification des valeur à regrouper
            to_input = [X_c[feature].isin(discarded) for discarded in to_discard]  # identification des valeurs à regrouper
            arr_feature = select(to_input, order, default=X_c[feature])  # regroupement des valeurs
            X_c[feature] = arr_feature  # storing grouped feature

        return X_c


class DefaultDiscretizer(BaseEstimator, TransformerMixin):
    
    def __init__(self, features: list, min_freq: float, *, default_value: str='__OTHER__', copy: bool=False, verbose: bool=False):
        """ Groups a qualitative features' values less frequent than min_freq into a default_value"""
        
        self.features = features[:]
        self.min_freq = min_freq
        self.values_orders = {}
        self.default_value = default_value[:]
        self.copy = copy
        self.verbose = verbose

    def fit(self, X, y):

        # computing frequencies of each modality
        frequencies = X[self.features].apply(lambda u: u.value_counts(dropna=False, normalize=True).to_dict(), axis=0).to_dict()

        # computing target rate per modality for ordering
        target_rates = X[self.features].apply(lambda u: y.groupby(u, dropna=True).mean().sort_values().to_dict(), axis=0).to_dict()

        # attributing orders to each feature
        self.values_orders = {feature: GroupedList([value for value, _ in target_rates.get(feature).items()]) for feature in self.features}

        # identifying modalities which are the most common
        self.to_keep = {feature: [value for value, frequency in frequencies.get(feature).items() if (frequency >= self.min_freq) & notna(value)] + [nan] for feature in self.features}

        # grouping rare modalities 
        for n, feature in enumerate(self.features):
            if self.verbose: print(f" - [DefaultDiscretizer] Fitting {feature} ({n+1}/{len(self.features)})")
            
            # identifying values to discard (rare modalities)
            to_discard = [value for value in self.values_orders.get(feature) if value not in self.to_keep.get(feature)]
            
            # discarding rare modalities
            if any(to_discard):
                
                # adding default_value to possible values
                order = self.values_orders.get(feature)
                order.append(self.default_value)
                
                # grouping rare modalities into default_value
                order.group_list(to_discard, self.default_value)
                
                # computing target rate for default value and reordering values according to feature's target rate
                default_target_rate = y.loc[X[feature].isin(order.get(self.default_value))].mean()  # computing target rate for default value
                order_target_rate = [target_rates.get(feature).get(value) for value in order if value != self.default_value]
                default_position = next(n for n, trate in enumerate(order_target_rate + [inf]) if trate > default_target_rate)
                new_order = order[:-1][:default_position] + [self.default_value] + order[:-1][default_position:]  # getting rid of default value already in order
                order = order.sort_by(new_order)
                self.values_orders.update({feature: order})

        return self

    def transform(self, X, y=None):

        # copying dataset if requested
        X_c = X
        if self.copy:
            X_c = X.copy()

        # grouping values inside groups of modalities
        for n, feature in enumerate(self.features):
            if self.verbose: print(f" - [DefaultDiscretizer] Discretizing {feature} ({n+1}/{len(self.features)})")

            # grouping modalities
            arr_feature = select([~X_c[feature].isin(self.to_keep.get(feature))], [self.default_value], default=X_c[feature])
            X_c[feature] = arr_feature  # storing grouped feature

        return X_c

def find_quantiles(df_feature: Series, q: int, len_df: int=None, quantiles: list=None):
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

def is_equal(a, b):
    """ checks if a and b are equal (NaN insensitive)"""
    
    # default equality
    equal = a == b
    
    # Case where a and b are NaNs
    if isna(a) and isna(b):
        equal = True
    
    return equal

def find_common_modalities(df_feature: Series, y: Series, min_freq: float, order: GroupedList, len_df: int=None):
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
        init_frequencies = df_feature.value_counts(dropna=False, normalize=True).drop(nan, errors='ignore')  # dropping nans to keep them anyways
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

def find_closest_modality(value, idx, freq_target, min_freq):
    """HELPER Finds the closesd modality in terms of frequency and target rate"""

    # cas 1: il s'agit de la modalité la plus petite
    if idx == 0:
        closest_modality = 1
        
    # cas 2: il s'agit de la modalité la plus grande
    elif idx == len(freq_target) - 1:
        closest_modality = len(freq_target) - 2
    
    # cas 3: il s'agit d'un cas intermédiaire
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


def format_list(a_list: list):
    """ Formats a list of floats to a list of unique rounded strings of floats"""

    # finding the closest power of thsousands for each element
    closest_powers = [next((k for k in range(-3, 4) if abs(elt) // 1000**(k) < 10)) for elt in a_list]

    # rounding elements to the closest power of thousands
    rounded_to_powers = [elt / 1000**(k) for elt, k in zip(a_list, closest_powers)]

    # computing optimal decimal per unique power of thousands
    optimal_decimals = {}
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
    rounded_list = []
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
    formatted_list = [str(elt) + suffixes.get(power) for elt, power in zip(rounded_list, closest_powers)]

    return formatted_list