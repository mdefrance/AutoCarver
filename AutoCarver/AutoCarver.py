from .Discretizers import GroupedList, GroupedListDiscretizer, is_equal
from IPython.display import display_html
from numpy import sort, nan, inf, float32, where, isin, argsort, array, select, append, quantile, linspace, argmin, sqrt, random
from pandas import DataFrame, Series, isna, qcut, notna, unique, concat, crosstab
from scipy.stats import chi2_contingency
from tqdm.notebook import tqdm


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

    max_n_mod: int, default 5
        Maximum number of modalities for the carved features (excluding `numpy.nan`).
         - All possible combinations of less than `max_n_mod` groups of modalities will be tested. 
        Recommandation: `max_n_mod` should be set from 4 (faster) to 6 (preciser).

    keep_nans: bool, default False
        Whether or not to group `numpy.nan` to other modalities/values.

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
    pipe += [('AutoCarver', auto_carver)]
    pipe = Pipeline(pipe)

    # applying pipe to a validation set or in production
    X_val = pipe.transform(X_val)

    """
    
    def __init__(self, values_orders: dict={}, *, sort_by: str='tschuprowt',
                 copy: bool=False, max_n_mod: int=5, keep_nans: bool=False,
                 verbose: bool=True):
        
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

    def fit(self, X: DataFrame, y: Series, X_test: DataFrame=None, y_test: Series=None):
 
        # copying dataframes
        Xc = X
        Xtestc = X_test if X_test is not None else DataFrame(columns=X.columns)
        if self.copy:
            Xc = X.copy()
            Xtestc = X_test.copy() if X_test is not None else DataFrame(columns=X.columns)

        # récupération du nom de la target
        self.target = y.name

        # découpage automatique de chacune des variables
        for n, feature in tqdm(enumerate(self.features), total=len(self.features), disable=self.verbose):

            # printing the group statistics and determining default ordering
            if self.verbose: print(f"\n---\n[AutoCarver] Fitting {feature} ({n+1}/{len(self.features)})")

            # getting best combination
            best_groups = self.get_best_combination(feature, X, y, X_test, y_test)

            # feature can not be carved robustly
            if not bool(best_groups):
                self.non_viable_features += [feature]
                print(f"Feature {feature} is not viable")

        # discretizing features based on each feature's values_order
        super().__init__(self.values_orders, copy=self.copy, output=float)
        super().fit(X, y)

        return self

    def get_best_combination(self, feature: str, X: DataFrame, y: Series, X_test: DataFrame, y_test: Series):

        # getting all possible combinations for the feature without NaNS
        combinations = get_all_combinations(self.values_orders.get(feature), X[feature].nunique(dropna=True), self.max_n_mod)
        combinations = [GroupedList({group[0]: group for group in combination}) for combination in combinations]

        # keeping NaNs as a specific modality
        nan_val = '__NAN__'  # value to be imputed as NaN
        df_feature = X[feature].fillna(nan_val)
        df_feature_test = X_test[feature].fillna(nan_val)

        # computing initial crosstabs
        xtab = crosstab(df_feature, y)
        xtab_test = crosstab(df_feature_test, y_test)

        # keeping track of nans if there are any
        xtab.index = [idx if idx != nan_val else nan for idx in xtab.index]
        xtab_test.index = [idx if idx != nan_val else nan for idx in xtab_test.index]

        # printing the group statistics
        if self.verbose:
            print(f'\n - Initial distribution of {feature}')
            self.display_target_rate(feature, xtab, xtab_test)

        # measuring association with target for each combination and testing for stability on TEST
        best_groups = best_combination(combinations, self.sort_by, xtab[notna(xtab.index)], 
                                       xtab_test[notna(xtab_test.index)])

        # update of the values_orders grouped modalities in values_orders
        if best_groups:

            self.update_values_orders(feature, best_groups['combination'])
            # update of the crosstabs
            xtab = xtab.groupby(list(map(best_groups['combination'].get_group, xtab.index)), dropna=False).sum()
            xtab_test = xtab_test.groupby(list(map(best_groups['combination'].get_group, xtab_test.index)), dropna=False).sum()

        # testing adding NaNs to built groups
        order = self.values_orders.get(feature)
        if any(isna(X[feature])) & (not self.keep_nans) & (len(order) <= self.max_n_mod):

            # getting all possible combinations for the feature with NaNS
            combinations = get_all_nan_combinations(order)
            combinations = [GroupedList({group[0]: group for group in combination}) for combination in combinations]

            # adding nans at the end of the order
            order.append(nan)

            # measuring association with target for each combination and testing for stability on TEST
            best_groups = best_combination(combinations, self.sort_by, xtab, xtab_test)

            # update of the values_orders grouped modalities in values_orders
            if best_groups:

                self.update_values_orders(feature, best_groups['combination'])
                # update of the crosstabs
                xtab = xtab.groupby(list(map(best_groups['combination'].get_group, xtab.index)), dropna=False).sum()
                xtab_test = xtab_test.groupby(list(map(best_groups['combination'].get_group, xtab_test.index)), dropna=False).sum()

        # printing the new group statistics
        if self.verbose and best_groups:

            print(f'\n - Fitted distribution of {feature}')
            self.display_target_rate(feature, xtab, xtab_test)

        return best_groups

    def update_values_orders(self, feature: str, best_groups: GroupedList):
        """ Updates the values_orders according to the best_groups"""

        # accessing current order for specified feature
        order = self.values_orders.get(feature)

        # grouping for each group of the combination
        for kept, discarded in best_groups.contained.items():
            order.group_list(discarded, kept)

    def display_target_rate(self, feature: str, xtab: DataFrame, xtab_test: DataFrame):
        """ Pretty display of frequency and target rate per modality on the same line. """

        known_order = self.values_orders.get(feature)
        stat_cols = ['target_rate', 'frequency']

        # target rate and frequency on TRAIN
        train_stats = DataFrame({
            'target_rate': xtab[1].divide(xtab[0]),  # target rate of each modality
            'frequency': xtab[1].add(xtab[0]) / xtab.sum().sum()  # frequency of each modality
        })
    
        # target rate and frequency on TEST
        test_stats = DataFrame({
            'target_rate': xtab_test[1].divide(xtab_test[0]),  # target rate of each modality
            'frequency': xtab_test[1].add(xtab_test[0]) / xtab_test.sum().sum()  # frequency of each modality
        })

        # default ordering based on observed target rate (on TRAIN)
        train_order = list(train_stats.sort_values('target_rate').index)
        test_order = list(test_stats.sort_values('target_rate').index)

        # keeping numpy.nan at the end of the order (default ordering)
        if any(isna(xtab.index)):
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

def best_combination(combinations: list, sort_by: str, xtab: DataFrame, xtab_test: DataFrame):
    """ Finds the best combination of groups of feature's values:
     - Most associated combination on train sample 
     - Stable target rate of combination on test sample.
    """

    # initiating list of associations between cut feature and target 
    associations = []

    # iterating over each combination
    for combination in combinations:
        
        # converting combination to a grouped list
        association = {'combination': combination}

        # grouping modalities in the initial crosstab
        groups = list(map(combination.get_group, xtab.index))
        combi_xtab = xtab.groupby(groups, dropna=False).sum()
        association.update({'combi_xtab': combi_xtab})

        # measuring association with the target
        association.update(association_xtab(combi_xtab))
        
        # storing results
        associations += [association]

    # sort according to association measure
    if any(combinations):
        associations = DataFrame(associations)
        associations.sort_values(sort_by, inplace=True, ascending=False)
        associations = associations.to_dict(orient='records')

    # testing associations
    for association in associations:
        
        # needed parameters
        combination, combi_xtab = association['combination'], association['combi_xtab']

        # grouping modalities in the initial crosstab
        combi_xtab_test = xtab_test.groupby(list(map(combination.get_group, xtab_test.index)), dropna=False).sum()

        # checking that all non-nan groups are in TRAIN and TEST
        unq_x =  [v for v in unique(combi_xtab.index) if v != notna(v)]
        unq_xtest = [v for v in unique(combi_xtab_test.index) if v != notna(v)]
        viability = all([e in unq_x for e in unq_xtest])
        viability = viability and all([e in unq_xtest for e in unq_x])

        # same target rate order in TRAIN and TEST
        train_target_rate = combi_xtab[1].divide(combi_xtab[0]).sort_values()
        test_target_rate = combi_xtab_test[1].divide(combi_xtab_test[0]).sort_values()
        viability = viability and all(train_target_rate.index == test_target_rate.index)

        # checking that some combinations were provided
        if viability:

        	association.update({'combi_xtab_test': combi_xtab_test})

        	return association


def get_all_combinations(values: list, q: int, max_n_mod: int=None):

    # max number of classes
    if max_n_mod is None:
    	max_n_mod = q

    # all possible combinations
    combinations = list()
    for n_class in range(2, max_n_mod + 1):
        combinations += get_combinations(n_class, q)

    combinations = [[values[int(c[0]): int(c[1]) + 1] for c in combi] for combi in combinations]  # getting real feature values

    return combinations

def get_all_nan_combinations(order: list):
    """ all possible combinations of modalities with numpy.nan"""

    # computing all non-NaN combinations
    if len(order) > 1:
        combinations = get_all_combinations(order, len(order))
    else:
        combinations = [[order]]

    # iterating over each combinations of non-NaNs
    new_combinations = []
    for combi in combinations:

         # NaNs not attributed to a group (own modality)
        new_combinations += [combi + [[nan]]] 

        # NaNs attributed to a group of non NaNs
        for n in range(len(combi)):

            # grouping NaNs with an existing group
            new_combination = [combi[n] + [nan]]

            # adding other groups unchanged
            pre = [o for o in combi[:n]]
            nex = [o for o in combi[n+1:]]
            new_combinations += [pre + new_combination + nex]

    return new_combinations


def consecutive_combinations(n_remaining: int, start: int, end: int, grp_for_this_step: list=None):
    """HELPER finds all consecutive combinations between start and end.    """

    # Import de la liste globale
    global __li_of_res__

    # initiating group
    if not grp_for_this_step:
    	grp_for_this_step = []
    
    # stopping when there are non more remaining classes
    if n_remaining == 0:
        
        ### On ajoute le dernier quantile restant au découpage ###
        grp_for_this_step += [(start, end)]

        ### Ajout du découpage réalisé au groupe des solutions ###
        __li_of_res__ += [grp_for_this_step]

    
    # adding all possible combinations of each possible range
    else:
        
        ### Parcours de toutes les valeurs possibles de fin pour le i-ème groupe du groupement à x quantiles ###
        for i in range(start, end):

            consecutive_combinations(n_remaining-1, i+1, end, grp_for_this_step + [(start, i)])
            
def get_combinations(n_class, q):
    """HELPER recupération des combinaisons possibles de q quantiles pour n_class."""
    
    globals()['__li_of_res__'] = []

    consecutive_combinations(n_class-1, 0, q-1)
    
    combinations = [list(map(lambda u: (str(u[0]).zfill(len(str(q-1))), str(u[1]).zfill(len(str(q-1)))), l)) for l in __li_of_res__]
    
    return combinations

def association_xtab(xtab: DataFrame):
    """ Computes measures of association between feature x and feature2. """

    # numnber of observations
    n_obs = xtab.sum().sum()

    # number of values taken by the features
    n_mod_x, n_mod_y = len(xtab.index), len(xtab.columns)

    # Chi2 statistic
    chi2 = chi2_contingency(xtab)[0]

    # Cramer's V
    cramerv = sqrt(chi2 / n_obs)

    # Tschuprow's T
    n_mods = sqrt((n_mod_x - 1) * (n_mod_y - 1))
    tschuprowt = 0
    if n_mods > 0:
        tschuprowt = sqrt(chi2 / n_obs / n_mods)

    results = {'cramerv': cramerv, 'tschuprowt': tschuprowt}
    
    return results
