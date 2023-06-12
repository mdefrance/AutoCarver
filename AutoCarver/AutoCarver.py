from .Discretizers import GroupedList, GroupedListDiscretizer, is_equal
from IPython.display import display_html
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.pyplot import subplots
from matplotlib.ticker import PercentFormatter
from numpy import sqrt
from pandas import DataFrame, Series, notna, unique, crosstab
from pandas.api.types import is_string_dtype
from scipy.stats import chi2_contingency
from seaborn import color_palette, despine
from tqdm.notebook import tqdm
from typing import Any, Dict, List, Tuple


class AutoCarver(GroupedListDiscretizer):
    """ Automatic carving of continuous, categorical and categorical ordinal
    features that maximizes association with a binary target.

    Modalities/values of features are carved/regrouped according to a computed
    specific order defined based on their types:
     - [Qualitative features] grouped based on modality target rate.
     - [Qualitative ordinal features] grouped based on specified modality order
     - [Quantitative features] grouped based on the order of their values.
    Uses Tschurpow's T to find the optimal carving (regrouping) of modalities/
    values of features.

    Parameters
    ----------
    values_orders: dict, default {}
        Dictionnary of features and list of their respective values' order.
        Exemple: for an `age` feature, `values_orders` could be
        `{'age': ['0-18', '18-30', '30-50', '50+']}`.

    sort_by: str, default 'tschuprowt'
        Association measure used to find the optimal group modality combination
        Implemented: ['cramerv', 'tschuprowt']

    max_n_mod: int, default 5
        Maximum number of modalities for the carved features (excluding `nan`).
         - All possible combinations of less than `max_n_mod` groups of
           modalities will be tested.
        Recommandation: `max_n_mod` should be set from 4 (faster) to 6
        (preciser).

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
    discretizer = Discretizer(
        selected_quanti, selected_quali, min_freq=0.02, q=20,
        values_orders=values_orders
    )
    X_train = discretizer.fit_transform(X_train, y_train)
    X_test = discretizer.transform(X_test)

    # storing Discretizer
    pipe = [('Discretizer', discretizer)]

    # updating features' values orders (every features are qualitative ordinal)
    values_orders = discretizer.values_orders

    # intiating AutoCarver
    auto_carver = AutoCarver(
        values_orders, sort_by='cramerv', max_n_mod=5, sample_size=0.01)

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
   
    def __init__(
            self,
            values_orders: Dict[str, Any],
            *,
            max_n_mod: int=5,
            sort_by: str='tschuprowt',
            str_nan: str='__NAN__',
            dropna: bool=True,
            copy: bool=False,
            verbose: bool=True
        ) -> None:
        
        self.features = list(values_orders.keys())
        self.values_orders = {k: GroupedList(v) for k, v in values_orders.items()}
        self.non_viable_features: List[str] = []  # list of features non viable features
        self.max_n_mod = max_n_mod  # maximum number of modality per feature
        self.dropna = dropna  # whether or not to group NaNs with other modalities
        
        # association measure used to find the best groups
        measures = ['tschuprowt', 'cramerv']
        assert sort_by in measures, f"""{sort_by} not yet implemented. Choose
                                        from: {', '.join(measures)}."""
        self.sort_by = sort_by

        self.copy = copy
        self.verbose = verbose
        self.str_nan = str_nan
    
    def prepare_data(
            self,
            X: DataFrame,
            y: Series,
            X_test: DataFrame=None,
            y_test: Series=None
        ) -> Tuple[DataFrame, Series, DataFrame, Series]:
        """ Checks validity of provided data"""
        
        # preparing train sample
        # checking for binary target
        y_values = unique(y)
        assert (0 in y_values) & (1 in y_values), "y must be a binary Series (int or float, not object)"
        assert len(y_values) == 2, "y must be a binary Series (int or float, not object)"
        
        # checking for quantitative columns
        is_object = X[self.features].dtypes.apply(is_string_dtype)
        assert all(is_object), f"Non-string features in X: {', '.join(is_object[~is_object].index)}, consider using Discretizer."
        
        # Copying DataFrame if requested
        Xc = X
        if self.copy:
            Xc = X.copy()
        
        # preparing test sample
        # checking for binary target
        if y_test is not None:
            assert X_test is not None, "y_test was provided but X_test is missing"
            y_values = unique(y_test)
            assert (0 in y_values) & (1 in y_values), "y_test must be a binary Series (int or float, not object)"
            assert len(y_values) == 2, "y_test must be a binary Series (int or float, not object)"
        
        # Copying DataFrame if requested
        Xtestc = X_test
        if X_test is not None:
            assert y_test is not None, "X_test was provided but y_test is missing"
            if self.copy:
                Xtestc = X_test.copy()
        
            # checking for quantitative columns
            is_object = X_test[self.features].dtypes.apply(is_string_dtype)
            assert all(is_object), f"Non-string features in X_test: {', '.join(is_object[~is_object].index)}, consider using Discretizer."
            
        return Xc, y, Xtestc, y_test
    
    def fit(
            self,
            X: DataFrame,
            y: Series,
            X_test: DataFrame=None,
            y_test: Series=None
        ) -> None:

        # preparing datasets and checking for wrong values
        Xc, y, Xtestc, y_test = self.prepare_data(X, y, X_test, y_test)
        
        # automatic carving of each feature
        for n, feature in tqdm(enumerate(self.features), total=len(self.features), disable=self.verbose):

            # printing the group statistics and determining default ordering
            if self.verbose:
                print(f"\n---\n[AutoCarver] Fit {feature} ({n+1}/{len(self.features)})")

            # getting best combination
            best_groups = self.get_best_combination(feature, Xc, y, Xtestc, y_test)

            # feature can not be carved robustly
            if not bool(best_groups):
                self.non_viable_features += [feature]  # adding it to list of non viable features

        # discretizing features based on each feature's values_order
        super().__init__(self.values_orders, str_nan=self.str_nan, copy=self.copy, output=float)
        super().fit(X, y)

        return self


    def get_best_combination(
            self,
            feature: str,
            X: DataFrame,
            y: Series,
            X_test: DataFrame=None,
            y_test: Series=None
        ) -> Dict[str, Any]:
        """ Carves a feature"""

        # computing crosstabs
        # crosstab on TRAIN
        xtab = nan_crosstab(X[feature], y, self.str_nan)

        # crosstab on TEST
        xtab_test = None
        if X_test is not None:
            xtab_test = nan_crosstab(X_test[feature], y_test, self.str_nan)

        # printing the group statistics
        if self.verbose:
            self.display_xtabs(feature, 'Raw', xtab, xtab_test)

        # measuring association with target for each combination and testing for stability on TEST
        best_groups = best_combination(self.values_orders.get(feature), self.max_n_mod, self.sort_by, xtab, xtab_test, dropna=True, str_nan=self.str_nan)

        # update of the values_orders grouped modalities in values_orders
        if best_groups:
            xtab, xtab_test = self.update_order(feature, best_groups['combination'], xtab, xtab_test)

        # testing adding NaNs to built groups
        if (self.str_nan in xtab.index) & self.dropna & (self.str_nan not in self.values_orders.get(feature)):

            # measuring association with target for each combination and testing for stability on TEST
            best_groups = best_combination(self.values_orders.get(feature), self.max_n_mod, self.sort_by, xtab, xtab_test, dropna=False, str_nan=self.str_nan)

            # adding NaN to the order
            self.insert_nan(feature, xtab)

            # update of the values_orders grouped modalities in values_orders
            if best_groups:
                xtab, xtab_test = self.update_order(feature, best_groups['combination'], xtab, xtab_test)

        # printing the new group statistics
        if self.verbose and best_groups:
            self.display_xtabs(feature, 'Fitted', xtab, xtab_test)

        return best_groups

    def insert_nan(
            self,
            feature: str,
            xtab: DataFrame
        ) -> GroupedList:
        """ Inserts NaNs in the order"""

        # accessing order for specified feature
        order = self.values_orders.get(feature)

        # adding nans at the end of the order
        if self.str_nan not in order:
            order = order.append(self.str_nan)

            # updating values_orders
            self.values_orders.update({feature: order})

    def update_order(
            self,
            feature: str,
            best_groups: GroupedList,
            xtab: DataFrame,
            xtab_test: DataFrame=None
        ) -> Tuple[DataFrame, DataFrame]:
        """ Updates the values_orders and xtabs according to the best_groups"""

        # updating values_orders with best_combination 
        self.values_orders.update({feature: best_groups})

        # update of the TRAIN crosstab
        best_combi = list(map(best_groups.get_group, xtab.index))
        xtab = xtab.groupby(best_combi, dropna=False).sum()

        # update of the TEST crosstab
        if xtab_test is not None:
            best_combi = list(map(best_groups.get_group, xtab_test.index))
            xtab_test = xtab_test.groupby(best_combi, dropna=False).sum()
        
        return xtab, xtab_test
    
    def display_xtabs(
            self,
            feature: str,
            caption: str,
            xtab: DataFrame,
            xtab_test: DataFrame=None
        ) -> None:
        """ Pretty display of frequency and target rate per modality on the same line. """
        
        # known_order per feature
        known_order = self.values_orders.get(feature)
        
        # target rate and frequency on TRAIN
        train_stats = stats_xtab(xtab, known_order)
    
        # target rate and frequency on TEST
        if xtab_test is not None:
            test_stats = stats_xtab(xtab_test, train_stats.index, train_stats.labels)
            test_stats = test_stats.set_index('labels')  # setting labels as indices
        
        # setting TRAIN labels as indices
        train_stats = train_stats.set_index('labels')

        # Displaying TRAIN modality level stats
        train_style = train_stats.style.background_gradient(cmap='coolwarm')  # color scaling
        train_style = train_style.set_table_attributes("style='display:inline'")  # printing in notebook
        train_style = train_style.set_caption(f'{caption} distribution on X:')  # title
        html = train_style._repr_html_()
        
        # adding TEST modality level stats
        if xtab_test is not None:
            test_style = test_stats.style.background_gradient(cmap='coolwarm')  # color scaling
            test_style = test_style.set_table_attributes("style='display:inline'")  # printing in notebook
            test_style = test_style.set_caption(f'{caption} distribution on X_test:')  # title
            html += ' ' + test_style._repr_html_()

        # displaying html of colored DataFrame
        display_html(html, raw=True)

def groupedlist_combination(
        combination: List[List[Any]],
        order: GroupedList
    ) -> GroupedList:
    """ Converts a list of combination to a GroupedList"""
    
    order_copy = GroupedList(order)
    for combi in combination:
        order_copy.group_list(combi, combi[0])
    
    return order_copy

def stats_xtab(
        xtab: DataFrame,
        known_order: List[Any]=None,
        known_labels: List[Any]=None
    ) -> DataFrame:
    """ Computes column (target) rate per row (modality) and row frequency"""
    
    # target rate and frequency statistics per modality
    stats = DataFrame({
        # target rate per modality
        'target_rate': xtab[1].divide(xtab.sum(axis=1)),
        # frequency per modality
        'frequency': xtab.sum(axis=1) / xtab.sum().sum()
    })
    
    # sorting statistics
    # case 0: default ordering based on observed target rate
    if known_order is None:
        order = list(stats.sort_values('target_rate', ascending=True).index)
    
    # case 1: a known_order was provided
    else:
        order = known_order[:]
      
    # modalities' labels
    # case 0: default labels
    if known_labels is None:
        
        # accessing string representation of the GroupedList
        if isinstance(known_order, GroupedList):
            labels = known_order.get_repr()

        # labels are the default order
        else:
            labels = order[:]
            
    # case 1: known_labels were provided
    else:
        labels = known_labels[:]
            
            
    # keeping values missing from the order at the end
    unknown_modality = [mod for mod in xtab.index if mod not in order]
    for mod in unknown_modality:
        order = [c for c in order if not is_equal(c, mod)] + [mod]
        labels = [c for c in labels if not is_equal(c, mod)] + [mod]
    
    # sorting statistics
    stats = stats.reindex(order, fill_value=0)
    stats['labels'] = labels
    
    return stats

def apply_combination(
        xtab: DataFrame,
        combination: GroupedList
    ) -> Dict[str, Any]:
    """ applies a modality combination to a crosstab """

    # initiating association dict
    association = {'combination': combination}

    # grouping modalities in the initial crosstab
    groups = list(map(combination.get_group, xtab.index))
    combi_xtab = xtab.groupby(groups, dropna=False).sum()
    association.update({'combi_xtab': combi_xtab})

    # measuring association with the target
    association.update(association_xtab(combi_xtab))

    return association

def best_combination(
        order: GroupedList,
        max_n_mod: int,
        sort_by: str,
        xtab_train: DataFrame,
        xtab_dev: DataFrame=None,
        dropna: bool=True,
        str_nan: str=None
    ) -> Dict[str, Any]:
    """ Finds the best combination of groups of feature's values:
     - Most associated combination on train sample 
     - Stable target rate of combination on test sample.
    """
    
    # copying crosstabs
    xtab = xtab_train
    xtab_test = xtab_dev
    
    # removing nans if requested
    if dropna:
        
        # crosstab on TRAIN
        xtab = xtab_train[xtab_train.index != str_nan]  # filtering out nans

        # crosstab on TEST
        if xtab_test is not None:
            xtab_test = xtab_dev[xtab_dev.index != str_nan]  # filtering out nans
    
        # getting all possible combinations for the feature without NaNS
        combinations = get_all_combinations(order, max_n_mod, raw=False)
    
    # keeping nans as a modality
    else:
    
        # getting all possible combinations for the feature with NaNS
        combinations = get_all_nan_combinations(order, str_nan, max_n_mod)

    # computing association measure per combination 
    associations = [apply_combination(xtab, combi) for combi in combinations]

    # sort according to association measure
    if len(combinations) > 0:
        associations = DataFrame(associations)
        associations.sort_values(sort_by, inplace=True, ascending=False)
        associations = associations.to_dict(orient='records')

    # testing associations
    # case 0: no test set was provided
    if xtab_test is None and len(associations) > 0:
        return associations[0]
    
    # case 1: testing viability on provided TEST sample
    else:
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


def get_all_combinations(
        values: GroupedList,
        max_n_mod: int=None,
        raw: bool=False
    ) -> List[GroupedList]:
    """ Returns all possible triangular combinations"""

    # maximum number of classes
    q = len(values)

    # desired max number of classes
    if max_n_mod is None:
        max_n_mod = q

    # all possible combinations
    combinations = list()
    for n_class in range(2, max_n_mod + 1):
        combinations += get_combinations(n_class, q)

    # getting real feature values
    combinations = [[values[int(c[0]): int(c[1]) + 1] for c in combi] for combi in combinations]
    
    # converting back to GroupedList
    if not raw:
        combinations = [groupedlist_combination(combination, values) for combination in combinations]

    return combinations

def get_all_nan_combinations(
        order: GroupedList,
        str_nan: str,
        max_n_mod: int
    ) -> List[GroupedList]:
    """ all possible combinations of modalities with numpy.nan"""
    
    # computing all non-NaN combinations
    # case 0: several modalities -> several combinations
    if len(order) > 1:
        combinations = get_all_combinations(order, max_n_mod-1, raw=True)
    # case 1: unique or no modality -> two combinations
    else:
        combinations = []

    # iterating over each combinations of non-NaNs
    new_combinations = []
    for combi in combinations:

         # NaNs not attributed to a group (own modality)
        new_combinations += [combi + [[str_nan]]] 

        # NaNs attributed to a group of non NaNs
        for n in range(len(combi)):

            # grouping NaNs with an existing group
            new_combination = [combi[n] + [str_nan]]

            # adding other groups unchanged
            pre = [o for o in combi[:n]]
            nex = [o for o in combi[n+1:]]
            new_combinations += [pre + new_combination + nex]
            
    # adding NaN to order
    if str_nan not in order:
        order = order.append(str_nan)
    
    # converting back to GroupedList
    new_combinations = [groupedlist_combination(combination, order) for combination in new_combinations] 

    return new_combinations


def consecutive_combinations(
        n_remaining: int,
        start: int,
        end: int,
        grp_for_this_step: list=None
    ) -> None:
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
            
def get_combinations(n_class: int, q: int) -> List[List[str]]:
    """HELPER recupération des combinaisons possibles de q quantiles pour n_class."""
    
    globals()['__li_of_res__'] = []

    consecutive_combinations(n_class-1, 0, q-1)
    
    combinations = [list(map(lambda u: (str(u[0]).zfill(len(str(q-1))), str(u[1]).zfill(len(str(q-1)))), l)) for l in __li_of_res__]
    
    return combinations

def association_xtab(xtab: DataFrame) -> Dict[str, float]:
    """ Computes measures of association between feature x and feature2. """

    # numnber of observations
    n_obs = xtab.sum().sum()

    # number of values taken by the features
    n_mod_x, n_mod_y = len(xtab.index), len(xtab.columns)
    min_n_mod = min(n_mod_x, n_mod_y)

    # Chi2 statistic
    chi2 = chi2_contingency(xtab)[0]

    # Cramer's V
    cramerv = 0
    if min_n_mod > 1:
        cramerv = sqrt(chi2 / n_obs / (min_n_mod - 1))

    # Tschuprow's T
    dof_mods = sqrt((n_mod_x - 1) * (n_mod_y - 1))
    tschuprowt = 0
    if dof_mods > 0:
        tschuprowt = sqrt(chi2 / n_obs / dof_mods)

    results = {'cramerv': cramerv, 'tschuprowt': tschuprowt}
    
    return results

def nan_crosstab(
        x: Series,
        y: Series,
        str_nan: str='__NAN__'
    ):
    """ Crosstab that keeps nans as a specific value"""
    
    # keeping NaNs as a specific modality
    x_filled = x.fillna(str_nan)  # filling NaNs

    # computing initial crosstabs
    xtab = crosstab(x_filled, y)
    
    return xtab


def plot_stats(stats: DataFrame) -> Tuple[Figure, Axes]:
    """ Barplot of the volume and target rate"""
    
    x = [0] + [elt for e in stats['frequency'].cumsum()[:-1] for elt in [e] * 2] + [1]
    y2 = [elt for e in list(stats['target_rate']) for elt in [e]*2]
    s = list(stats.index)
    scaled_y2 = [(y-min(y2)) / (max(y2) - min(y2)) for y in y2]
    c = color_palette("coolwarm", as_cmap=True)(scaled_y2)

    fig, ax = subplots()

    for i in range(len(stats)):
        k = i*2
        ax.fill_between(x[k: k+2], [0, 0], y2[k: k+2], color=c[k])
        ax.text(sum(x[k: k+2]) / 2, y2[k], s[i], rotation=90, ha='center', va='bottom')

    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))    
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))    
    ax.set_xlabel('Volume')
    ax.set_ylabel('Target rate')
    ax.set_xlim(0, 1)
    ax.set_ylim(0)
    despine()
    
    return fig, ax
