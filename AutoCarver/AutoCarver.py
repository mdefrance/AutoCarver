from IPython.display import display_html
from numpy import sort, nan, inf, float32, where, isin, argsort, array, select, append, quantile, linspace, argmin, sqrt
from pandas import DataFrame, Series, isna, qcut, notna, unique, concat, crosstab
from scipy.stats import chi2_contingency
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm.notebook import tqdm
from warnings import warn
from Discretizers import GroupedList, GroupedListDiscretizer, is_equal


class AutoCarver(GroupedListDiscretizer):
    """ Automatic carving of continuous, categorical and categorical ordinal features 
    that maximizes association with a binary target.

    Modalities/values of features are carved/regrouped according to a computed specific
    order defined based on their types:
     - [Qualitative features] grouped based on modality target rate.
     - [Qualitative ordinal features] grouped based on specified modality order.
     - [Quantitative features] grouped based on the order of their values.
    Uses Tschurpow's T to find the optimal carving (regrouping) of modalities/values
    of features.

    Parameters
    ----------
    values_orders: dict, default {}
        Dictionnary of features and list of their respective values' order.
        Exemple: for an `age` feature, `values_orders` could be `{'age': ['0-18', '18-30', '30-50', '50+']}`.

    sort_by: str, default 'tschuprowt'
        Association measure used to find the optimal group modality combination.
        Implemented: ['cramerv', 'tschuprowt']

    sample_size: float, default 0.01
        Sample size used for stratified sampling per feature modalities by target rate. 
        Recommandation: `sample_size` should be set from 0.01 (faster, large dataset) to 0.2 (preciser, small dataset).

    max_n_mod: int, default 5
        Maximum number of modalities for the carved features (excluding `numpy.nan`).
         - All possible combinations of less than `max_n_mod` groups of modalities will be tested. 
        Recommandation: `max_n_mod` should be set from 4 (faster) to 6 (preciser).

    keep_nans: bool, default False
        Whether or not to group `numpy.nan` to other modalities/values.

    test_sample_size: float, default 1.0
        Sample size used for stratified sampling per feature modalities by target rate on test sample. 
        Recommandation: `test_sample_size` should be set from 0.1 (faster, large dataset) to 1.0 (preciser, small dataset).

    
    Examples
    ----------

    from AutoCarver import AutoCarver
    from Discretizers import Discretizer
    from sklearn.pipeline import Pipeline

    # defining training and testing sets
    X_train, y_train = train_set, train_set[target]
    X_test, y_test = test_set, test_set[target]

    # specifying features to be carved
    selected_quanti = ['amount', 'distance', 'length', 'height']
    selected_quali = ['age', 'type', 'grade', 'city']

    # specifying orders of categorical ordinal features
    values_orders = {
        'age': ['0-18', '18-30', '30-50', '50+'],
        'grade': ['A', 'B', 'C', 'D', 'J', 'K', 'NN']
    }

    # pre-processing of features into categorical ordinal features
    discretizer = Discretizer(selected_quanti, selected_quali, min_freq=0.02, q=20, values_orders=values_orders)
    X_train = discretizer.fit_transform(X_train, y_train)
    X_test = discretizer.transform(X_test)

    # storing Discretizer
    pipe = [('Discretizer', discretizer)]

    # updating features' values orders (at this step every features are qualitative ordinal)
    values_orders = discretizer.values_orders

    # intiating AutoCarver
    auto_carver = AutoCarver(values_orders, sort_by='cramerv', max_n_mod=5, sample_size=0.01)

    # fitting on training sample 
    # a test sample can be specified to evaluate carving robustness
    X_train = auto_carver.fit_transform(X_train, y_train, X_test, y_test)

    # applying transformation on test sample
    X_test = auto_carver.transform(X_test)

    # identifying non stable/robust features
    print(auto_carver.non_viable_features)

    # storing fitted GroupedListDiscretizer in a sklearn.pipeline.Pipeline
    pipe += [('AutoCarver', auto_carver.discretizer)]
    pipe = Pipeline(pipe)

    # applying pipe to a validation set or in production
    X_val = pipe.transform(X_val)

    """
    
    def __init__(self, values_orders: dict={}, *, sort_by: str='tschuprowt', sample_size: float=0.01,
                 copy: bool=False, max_n_mod: int=5, keep_nans: bool=False, test_sample_size: float=None, 
                 verbose: bool=True):
        """ Découpage automatique des variables : min_freq pour variables qualitatives, q pour celle continue"""
        
        self.features = list(values_orders.keys())
        self.values_orders = {k: GroupedList(v) for k, v in values_orders.items()}
        self.target = None  # le nom de la target est récupéré lors du fit
        self.discretizer = None
        self.non_viable_features = []  # liste des features pas stable entre TRAIN et TEST
        self.max_n_mod = max_n_mod  # nombre maximal de modalités
        self.keep_nans = keep_nans  # whether or not to group NaNs with other modalities
        self.sort_by = sort_by  # association measure used find the best groups
        self.copy = copy
        self.verbose = verbose
        self.sample_size = sample_size
        self.test_sample_size = test_sample_size

    def fit(self, X: DataFrame, y: Series, X_test: DataFrame=None, y_test: Series=None):
 
        # copying dataframes
        Xc = X
        X_testc = X_test if X_test is not None else DataFrame(columns=X.columns)
        if self.copy:
            Xc = X.copy()
            X_testc = X_test.copy() if X_test is not None else DataFrame(columns=X.columns)

        # récupération du nom de la target
        self.target = y.name

        # computing test sample size if not provided
        if self.test_sample_size is None:
            self.test_sample_size = min(self.sample_size / (len(X_test) / len(X)), 1)

        # découpage automatique de chacune des variables
        for n, feature in enumerate(self.features):
            if self.verbose: print(f"\n---\n[AutoCarver] Fitting {feature} ({n+1}/{len(self.features)})")

            # sampling data for faster computation
            Xc = X[[feature, self.target]].groupby([feature, self.target], group_keys=False, dropna=False).apply(lambda x: x.sample(frac=self.sample_size))
            yc = Xc[self.target]
            if self.test_sample_size < 1:
                Xtestc = X_test[[feature, self.target]].groupby([feature, self.target], group_keys=False, dropna=False).apply(lambda x: x.sample(frac=self.test_sample_size))
            else:
            	Xtestc = X_test[[feature, self.target]]
            ytestc = Xtestc[self.target]

            # printing the group statistics and determining default ordering
            if self.verbose:
                print(f'\n - {feature} Initial distribution')
                self.display_target_rate(Xc, Xtestc, feature)

            # getting best combination
            best_groups = self.get_best_combination(feature, Xc, yc, Xtestc, ytestc)

            # feature can not be carved robustly
            if not bool(best_groups):
                self.non_viable_features += [feature]
                warn("Following feature is not stable on test set: {}".format(feature))

            # feature can be carved robustly
            elif self.verbose:
                print(f'\n - {feature} Fitted distribution')
                self.display_target_rate(
                	DataFrame({feature: best_groups.get('x_cut'), self.target: yc}), 
                	DataFrame({feature: best_groups.get('x_test_cut'), self.target: ytestc}), feature)

        # discretizing features based on each feature's values_order
        super().__init__(self.values_orders, copy=self.copy, output=float)
        super().fit(Xc, yc)

        return self

    def get_best_combination(self, feature: str, X: DataFrame, y: Series, X_test: DataFrame, y_test: Series):

        # getting all possible combinations for the feature without NaNS
        n_values = X[feature].nunique(dropna=True)
        combinations = get_all_combinations(n_values, self.max_n_mod)
        combinations = [[self.values_orders.get(feature)[int(c[0]): int(c[1]) + 1] for c in combi] for combi in combinations]  # getting real feature values

        # measuring association with target for each combination and testing for stability on TEST
        if self.verbose: print(" - Grouping Modalities:")
        best_groups = best_combination(combinations, self.sort_by, feature, X, y, X_test, y_test, self.verbose, keep_nans=False)

        # storing grouped modalities in values_orders
        order = self.values_orders.get(feature)
        if bool(best_groups):
            combination = best_groups.get('combination')
            for group in combination:
                order.group_list(group[1:], [group[0]] * len(group[1:]))
            X = DataFrame({feature: best_groups.get('x_cut')})
            X_test = DataFrame({feature: best_groups.get('x_test_cut')})

        # testing adding NaNs to built groups
        if any(isna(X[feature])) & (not self.keep_nans):
            
            if self.verbose: print(" - Grouping NaNs:")

            # measuring association with target for each combination and testing for stability on TEST
            combinations = get_all_nan_combinations(order)
            best_groups = best_combination(combinations, self.sort_by, feature, X, y, X_test, y_test, self.verbose, keep_nans=True)

        return best_groups

    def display_target_rate(self, X, X_test, feature):
        """ Pretty display of frequency and target rate per modality on the same line. """

        known_order = self.values_orders.get(feature)
        stat_cols = ['target_rate', 'frequency']

        # target rate and frequency on TRAIN
        train_stats = DataFrame(X.groupby(feature, dropna=False)[self.target].mean())
        train_stats = train_stats.join(X[feature].value_counts(normalize=True, dropna=False))
        train_stats.columns = stat_cols
    
        # target rate and frequency on TEST
        test_stats = DataFrame(X_test.groupby(feature, dropna=False)[self.target].mean())
        test_stats = test_stats.join(X_test[feature].value_counts(normalize=True, dropna=False))
        test_stats.columns = stat_cols

        # default ordering based on observed target rate (on TRAIN)
        train_order = list(train_stats.sort_values('target_rate').index)
        test_order = list(test_stats.sort_values('target_rate').index)

        # keeping numpy.nan at the end of the order (default ordering)
        if any(X[feature].isna()):
            train_order = [c for c in train_order if notna(c)]
            train_order += [nan]
            test_order = [c for c in test_order if notna(c)]
            test_order += [nan]

        # ordering by target rate
        train_stats = train_stats.reindex(train_order)
        test_stats = test_stats.reindex(test_order)

        # ordering statistics if an order already exists
        if known_order is not None:

            # known modalities
            new_train_order = known_order[:]

            # adding unexpected modalities (only present in train_order)
            new_train_order += [elt for elt in train_order if not any(is_equal(elt, t) for t in new_train_order)]

            # ordering TRAIN
            train_stats = train_stats.reindex(new_train_order)

            # adding start and end modalities
            renamed_train_order = [f'{known_order.get(c)[-1]} to {known_order.get(c)[0]}' if c in known_order and len(known_order.get(c)) > 1 else c for c in new_train_order]
            train_stats.index = renamed_train_order

            # ordering the TEST statistics based on known order
            new_test_order = new_train_order[:]

            # adding unexpected modalities (only present in test_order)
            new_test_order += [elt for elt in test_order if not any(is_equal(elt, t) for t in new_test_order)]

            # ordering TEST
            test_stats = test_stats.reindex(new_test_order)

            # adding start and end modalities
            renamed_test_order = [f'{known_order.get(c)[-1]} to {known_order.get(c)[0]}' if c in known_order and len(known_order.get(c)) > 1 else c for c in new_test_order]
            test_stats.index = renamed_test_order

        # Displaying feature level stats
        styler = concat([train_stats, test_stats], ignore_index=True).style.background_gradient(cmap='coolwarm')  # unifying colors for both stats

        train_style = train_stats.style.use(styler.export()).set_table_attributes("style='display:inline'").set_caption('Train:')
        test_style = test_stats.style.use(styler.export()).set_table_attributes("style='display:inline'").set_caption('Test:')
        display_html(train_style._repr_html_() + ' ' + test_style._repr_html_(), raw=True)
    
        return train_order

def best_combination(combinations: list, sort_by: str, feature: str, X: DataFrame, y: Series, X_test: DataFrame, y_test: Series,
	                 verbose: bool, keep_nans: bool):
    """ Finds the best combination of groups of feature's values:
     - Most associated combination on train sample 
     - Stable target rate of combination on test sample.
    """

    # initiating list of associations between cut feature and target 
    associations = []
    df_feature, arr_feature = X[feature], X[feature].values
    
    # iterating over each combination
    for combination in tqdm(combinations, disable=not verbose):

        # grouping modalities according to the combination on train sample
        to_input = [df_feature.isin(values) for values in combination]
        value_input = [combi[0] for combi in combination]
        x_cut = select(to_input, value_input, default=arr_feature)
        
        # measuring association with the target
        association = association_quali_y(x_cut, y.values, keep_nans)
        
        # storing results
        association.update({'combination': combination, 'x_cut': x_cut})
        associations += [association]

    # checking that some combinations were provided
    if any(associations):

        # sort according to association measure
        associations = DataFrame(associations)
        associations.sort_values(sort_by, inplace=True, ascending=False)
        associations = associations.to_dict(orient='records')
    
        # evaluating stability on test sample
        combination = test_stability(associations, feature, X, y, X_test, y_test, verbose)
    
        return combination

def test_stability(associations: list, feature: str, X: DataFrame, y: Series, X_test: DataFrame, y_test: Series, verbose: bool):

    # iterating over each combination
    for association in tqdm(associations, disable=not verbose):
        
        # retrieving used combination
        combination = association.get('combination')

        # retrieving groupend feature on train sample
        x_cut = association.get('x_cut')

        # grouping modalities according to the combination on test sample
        if len(X_test) > 0:
            to_input = [X_test[feature].isin(values) for values in combination]
            value_input = [combi[0] for combi in combination]
            x_test_cut = select(to_input, value_input, default=X_test[feature])

        # same groups in TRAIN and TEST
        viability = all([e in unique(x_cut) for e in unique(x_test_cut) if notna(e)])
        viability = viability and all([e in unique(x_test_cut) for e in unique(x_cut) if notna(e)])
        # same target rate order in TRAIN and TEST
        train_target_rate = y.groupby(x_cut, dropna=True).mean().sort_values()
        test_target_rate = y_test.groupby(x_test_cut, dropna=True).mean().sort_values()
        viability = viability and all(train_target_rate.index == test_target_rate.index)
        
        if viability:
            association.update({'x_test_cut': x_test_cut})

            return association

def get_all_combinations(q: int, max_n_mod: int=None):
    
    # max number of classes
    if max_n_mod is None:
    	max_n_mod = q

    # all possible combinations
    combinations = list()
    for n_class in range(2, max_n_mod + 1):
        combinations += get_combinations(n_class, q)

    return combinations

def get_all_nan_combinations(order: list):
    """ all possible combinations of modalities with numpy.nan"""

    # initiating combination with NaN as a specific modality
    combinations = [[[c] for c in order] + [[nan]]]

    # adding combination in which non nans are grouped but not NAN
    for n in range(len(order) - 1):

        # grouping NaNs with an existing group
        new_combination = [order[n],  order[n+1]]

        # adding other groups unchanged
        pre = [[o] for o in order[:n]]
        nex = [[o] for o in order[n+2:]]
        combinations += [pre + [new_combination] + nex + [[nan]]]

    # iterating over each groups
    for n in range(len(order)):

        # grouping NaNs with an existing group
        new_combination = [order[n], nan]

        # adding other groups unchanged
        pre = [[o] for o in order[:n]]
        nex = [[o] for o in order[n+1:]]
        combinations += [pre + [new_combination] + nex]

    return combinations


def consecutive_combinations(n_remaining, start, end, grp_for_this_step=[]):
    """HELPER finds all consecutive combinations between start and end.    """

    # Import de la liste globale
    global __li_of_res__
    
    ### cas d'arrêt de la fonction récursive : quand il ne reste plus de nouvelle classe à créer ###
    if n_remaining==0:
        
        ### On ajoute le dernier quantile restant au découpage ###
        grp_for_this_step += [(start, end)]

        ### Ajout du découpage réalisé au groupe des solutions ###
        __li_of_res__ += [grp_for_this_step]

    
    ### Hors du cas d'arrêt ####
    else:
        
        ### Parcours de toutes les valeurs possibles de fin pour le i-ème groupe du groupement à x quantiles ###
        for i in range(start, end):
            
            ### Ajout de récursivité ###
            consecutive_combinations(n_remaining-1, i+1, end, grp_for_this_step + [(start, i)])
            
def get_combinations(n_class, q):
    """HELPER recupération des combinaisons possibles de q quantiles pour n_class."""
    
    globals()['__li_of_res__'] = []

    consecutive_combinations(n_class-1, 0, q-1)
    
    combinations = [list(map(lambda u: (str(u[0]).zfill(len(str(q-1))), str(u[1]).zfill(len(str(q-1)))), l)) for l in __li_of_res__]
    
    return combinations

def association_quali_y(x: array, y: array, keep_nans: bool):
    """ Computes measures of association between feature x and feature2. """
    
    # whether or not to keep nans
    if keep_nans:
        yc = y.astype(str)
        xc = x.astype(str)
    else:
        yc = y[notna(x)].astype(str)
        xc = x[notna(x)].astype(str)

    # numnber of observations
    n_obs = len(xc)

    # number of values taken by the features
    n_mod_x, n_mod_y = len(unique(xc)), len(unique(yc))

    # frequency across target and feature values
    crossed = crosstab(xc, yc)

    # Chi2 statistic
    chi2 = chi2_contingency(crossed)[0]

    # Cramer's V
    cramerv = sqrt(chi2 / n_obs)

    # Tschuprow's T
    n_mods = sqrt((n_mod_x - 1) * (n_mod_y - 1))
    tschuprowt = 0
    if n_mods > 0:
        tschuprowt = sqrt(chi2 / n_obs / n_mods)
    
    # recupération des résultats
    results = {'cramerv': cramerv, 'tschuprowt': tschuprowt}
    
    return results
