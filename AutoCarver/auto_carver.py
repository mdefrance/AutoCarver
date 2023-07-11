"""Tool to build optimized buckets out of Quantitative and Qualitative features
for a binary classification model.
"""

from typing import Any, Dict, List, Tuple

from .discretizers.utils.base_discretizers import GroupedList, GroupedListDiscretizer, is_equal
from .discretizers.discretizers import Discretizer
from IPython.display import display_html  # TODO: remove this
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.pyplot import subplots
from matplotlib.ticker import PercentFormatter
from numpy import sqrt
from pandas import DataFrame, Series, crosstab, notna, unique
from pandas.api.types import is_string_dtype
from scipy.stats import chi2_contingency
from seaborn import color_palette, despine
from tqdm.notebook import tqdm

# TODO: issue with ranking of quantitative values? -> displayed not in an ascending way
# TODO: add parameter to shut down displayed info
class AutoCarver(GroupedListDiscretizer):
    """Automatic carving of continuous, categorical and categorical ordinal
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
        quantitative_features: List[str],
        qualitative_features: List[str],
        min_freq: float,
        *,
        ordinal_features: List[str] = None,
        values_orders: Dict[str, GroupedList] = None,
        max_n_mod: int = 5,
        sort_by: str = "tschuprowt",
        str_nan: str = "__NAN__",
        str_default: str = '__OTHER__',
        dropna: bool = True,
        copy: bool = False,
        verbose: bool = True,
    ) -> None:

        # copying quantitative features and checking for duplicates
        self.quantitative_features = quantitative_features[:]
        assert len(list(set(quantitative_features))) == len(
            quantitative_features
        ), "Column duplicates in quantitative_features"

        # copying qualitative features and checking for duplicates
        self.qualitative_features = qualitative_features[:]
        assert len(list(set(qualitative_features))) == len(
            qualitative_features
        ), "Column duplicates in qualitative_features"
        
        # copying ordinal features and checking for duplicates
        if ordinal_features is None:
            ordinal_features = []
        self.ordinal_features = ordinal_features[:]
        assert len(list(set(ordinal_features))) == len(
            ordinal_features
        ), "Column duplicates in ordinal_features"

        # initiating values_orders
        if values_orders is None:
            values_orders = {}
        self.values_orders = {k: GroupedList(v) for k, v in values_orders.items()}

        # list of all features
        self.features = list(set(quantitative_features + qualitative_features + ordinal_features))

        # minimum frequency per base bucket
        self.min_freq = min_freq
        # maximum number of modality per feature
        self.max_n_mod = max_n_mod
        # whether or not to group NaNs with other modalities
        self.dropna = dropna

        # association measure used to find the best groups
        measures = ["tschuprowt", "cramerv"]
        assert (
            sort_by in measures
        ), f"""{sort_by} not yet implemented. Choose
                                        from: {', '.join(measures)}."""
        self.sort_by = sort_by

        self.copy = copy
        self.verbose = verbose
        self.str_nan = str_nan
        self.str_default = str_default

    def prepare_data(
        self,
        X: DataFrame,
        y: Series,
        X_test: DataFrame = None,
        y_test: Series = None,
    ) -> Tuple[DataFrame, DataFrame]:
        """Checks validity of provided data"""

        # preparing train sample
        # checking for binary target
        y_values = unique(y)
        assert (0 in y_values) & (
            1 in y_values
        ), "y must be a binary Series (int or float, not object)"
        assert len(y_values) == 2, "y must be a binary Series (int or float, not object)"

        # Copying DataFrame if requested
        x_copy = X
        if self.copy:
            x_copy = X.copy()

        # preparing test sample
        # checking for binary target
        if y_test is not None:
            assert X_test is not None, "y_test was provided but X_test is missing"
            y_values = unique(y_test)
            assert (0 in y_values) & (
                1 in y_values
            ), "y_test must be a binary Series (int or float, not object)"
            assert len(y_values) == 2, "y_test must be a binary Series (int or float, not object)"

        # Copying DataFrame if requested
        x_test_copy = X_test
        if X_test is not None:
            assert y_test is not None, "X_test was provided but y_test is missing"
            if self.copy:
                x_test_copy = X_test.copy()

            # checking for quantitative columns
            is_object = X_test[self.features].dtypes.apply(is_string_dtype)
            assert all(
                is_object
            ), f"Non-string features in X_test: {', '.join(is_object[~is_object].index)}, consider using Discretizer."

        return x_copy, x_test_copy

    def fit(
        self,
        X: DataFrame,
        y: Series,
        X_test: DataFrame = None,
        y_test: Series = None,
    ) -> None:
        """_summary_

        Parameters
        ----------
        X : DataFrame
            _description_
        y : Series
            _description_
        X_test : DataFrame, optional
            _description_, by default None
        y_test : Series, optional
            _description_, by default None
        """
        # preparing datasets and checking for wrong values
        x_copy, x_test_copy = self.prepare_data(X, y, X_test, y_test)

        # discretizing all features
        discretizer = Discretizer(
            quantitative_features = self.quantitative_features,
            qualitative_features = self.qualitative_features,
            min_freq = self.min_freq,
            ordinal_features = self.ordinal_features,
            values_orders = self.values_orders,
            str_nan = self.str_nan,
            str_default = self.str_default,
            copy = False,
            verbose = self.verbose,
        )
        x_copy = discretizer.fit_transform(x_copy, y)
        if x_test_copy is not None:
            x_test_copy = discretizer.transform(x_test_copy, y_test)

        # updating values_orders according to base bucketization
        self.values_orders.update(discretizer.values_orders)

        # optimal butcketization/carving of each feature
        # if not self.verbose:
        # best_groups = X.apply(self.get_best_combination(feature, X, y, X_test=X_test, y_test=y_test))
        for n, feature in enumerate(self.features):
            if self.verbose:  # verbose if requested
                print(f"\n---\n[AutoCarver] Fit {feature} ({n+1}/{len(self.features)})")

            # getting best combination
            self.get_best_combination(feature, x_copy, y, X_test=x_test_copy, y_test=y_test)

        # discretizing features based on each feature's values_order
        super().__init__(
            features=self.features,
            values_orders=self.values_orders,
            copy=self.copy,
            input_dtypes=discretizer.input_dtypes,
            output_dtype='float',
            str_nan=self.str_nan,
            verbose=self.verbose,
        )
        super().fit(X, y)

        return self

    def get_best_combination(
        self,
        feature: str,
        X: DataFrame,
        y: Series,
        *,
        X_test: DataFrame = None,
        y_test: Series = None,
    ) -> Dict[str, Any]:
        """Carves a feature

        Parameters
        ----------
        feature : str
            Column name of the feature to be bucketized
        X : DataFrame
            Contains a column named `feature`
        y : Series
            Model target
        X_test : DataFrame, optional
            Used for robustess evaluation. Contains a column named `feature`, by default None
        y_test : Series, optional
            Used for robustess evaluation. Model target, by default None

        Returns
        -------
        Dict[str, Any]
            _description_
        """

        # computing crosstabs
        # crosstab on TRAIN
        xtab = nan_crosstab(X[feature], y, self.str_nan)

        # crosstab on TEST
        xtab_test = None
        if X_test is not None:
            xtab_test = nan_crosstab(X_test[feature], y_test, self.str_nan)

        if self.verbose:  # printing the group statistics
            self.display_xtabs(feature, "Raw", xtab, xtab_test)

        # measuring association with target for each combination and testing for stability on TEST
        best_groups = None
        if xtab.shape[0] > 2:  # checking that there are modalities
            best_groups = best_combination(
                self.values_orders[feature],
                self.max_n_mod,
                self.sort_by,
                xtab,
                xtab_dev=xtab_test,
                dropna=True,
                str_nan=self.str_nan,
            )

        # update of the values_orders grouped modalities in values_orders
        if best_groups:
            xtab, xtab_test = self.update_order(
                feature, best_groups["combination"], xtab, xtab_test
            )

        # testing adding NaNs to built groups
        if (
            (self.str_nan in xtab.index)
            & self.dropna
            & (self.str_nan not in self.values_orders.get(feature))
        ):
            # measuring association with target for each combination and testing for stability on TEST
            best_groups = best_combination(
                self.values_orders.get(feature),
                self.max_n_mod,
                self.sort_by,
                xtab,
                xtab_dev=xtab_test,
                dropna=False,
                str_nan=self.str_nan,
            )

            # adding NaN to the order
            self.insert_nan(feature)

            # update of the values_orders grouped modalities in values_orders
            if best_groups:
                xtab, xtab_test = self.update_order(
                    feature, best_groups["combination"], xtab, xtab_test
                )

        # printing the new group statistics
        if self.verbose and best_groups:
            self.display_xtabs(feature, "Fitted", xtab, xtab_test)

        return best_groups

    def insert_nan(self, feature: str) -> GroupedList:
        """Inserts NaNs in the order"""

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
        xtab_test: DataFrame = None,
    ) -> Tuple[DataFrame, DataFrame]:
        """Updates the values_orders and xtabs according to the best_groups"""

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
        xtab_test: DataFrame = None,
    ) -> None:
        """Pretty display of frequency and target rate per modality on the same line."""

        # known_order per feature
        known_order = self.values_orders[feature]

        # target rate and frequency on TRAIN
        train_stats = stats_xtab(xtab, known_order)

        # target rate and frequency on TEST
        if xtab_test is not None:
            test_stats = stats_xtab(xtab_test, train_stats.index, train_stats.labels)
            test_stats = test_stats.set_index("labels")  # setting labels as indices

        # setting TRAIN labels as indices
        train_stats = train_stats.set_index("labels")

        # Displaying TRAIN modality level stats
        train_style = train_stats.style.background_gradient(cmap="coolwarm")  # color scaling
        train_style = train_style.set_table_attributes(
            "style='display:inline'"
        )  # printing in notebook
        train_style = train_style.set_caption(f"{caption} distribution on X:")  # title
        html = train_style._repr_html_()

        # adding TEST modality level stats
        if xtab_test is not None:
            test_style = test_stats.style.background_gradient(cmap="coolwarm")  # color scaling
            test_style = test_style.set_table_attributes(
                "style='display:inline'"
            )  # printing in notebook
            test_style = test_style.set_caption(f"{caption} distribution on X_test:")  # title
            html += " " + test_style._repr_html_()

        # displaying html of colored DataFrame
        display_html(html, raw=True)


def groupedlist_combination(combination: List[List[Any]], order: GroupedList) -> GroupedList:
    """Converts a list of combination to a GroupedList"""

    order_copy = GroupedList(order)
    for combi in combination:
        order_copy.group_list(combi, combi[0])

    return order_copy


def stats_xtab(
    xtab: DataFrame,
    known_order: List[Any] = None,
    known_labels: List[Any] = None,
) -> DataFrame:
    """Computes column (target) rate per row (modality) and row frequency"""

    # target rate and frequency statistics per modality
    stats = DataFrame(
        {
            # target rate per modality
            "target_rate": xtab[1].divide(xtab.sum(axis=1)),
            # frequency per modality
            "frequency": xtab.sum(axis=1) / xtab.sum().sum(),
        }
    )

    # sorting statistics
    # case 0: default ordering based on observed target rate
    if known_order is None:
        order = list(stats.sort_values("target_rate", ascending=True).index)

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
    stats["labels"] = labels

    return stats


def apply_combination(xtab: DataFrame, combination: GroupedList) -> Dict[str, Any]:
    """Applies a modality combination to a crosstab

    Parameters
    ----------
    xtab : DataFrame
        Crosstab
    combination : GroupedList
        Combination of index to apply to the crosstab

    Returns
    -------
    Dict[str, Any]
        _description_
    """

    # initiating association dict
    association = {"combination": combination}

    # grouping modalities in the initial crosstab
    groups = list(map(combination.get_group, xtab.index))
    combi_xtab = xtab.groupby(groups, dropna=False).sum()
    association.update({"combi_xtab": combi_xtab})

    # measuring association with the target
    association.update(association_xtab(combi_xtab))

    return association


def best_combination(
    order: GroupedList,
    max_n_mod: int,
    sort_by: str,
    xtab_train: DataFrame,
    *,
    xtab_dev: DataFrame = None,
    dropna: bool = True,
    str_nan: str = '__NAN__',
) -> Dict[str, Any]:
    """Finds the best combination of groups of feature's values:
    - Most associated combination on train sample
    - Stable target rate of combination on test sample.

    Parameters
    ----------
    order : GroupedList
        _description_
    max_n_mod : int
        _description_
    sort_by : str
        _description_
    xtab_train : DataFrame
        _description_
    xtab_dev : DataFrame, optional
        _description_, by default None
    dropna : bool, optional
        _description_, by default True
    str_nan : str, optional
        _description_, by default '__NAN__'

    Returns
    -------
    Dict[str, Any]
        _description_
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
        associations = associations.to_dict(orient="records")

    # testing associations
    # case 0: no test set was provided
    if xtab_test is None and len(associations) > 0:
        return associations[0]

    # case 1: testing viability on provided TEST sample
    for association in associations:
        # needed parameters
        combination, combi_xtab = (
            association["combination"],
            association["combi_xtab"],
        )
        
        # TODO: replace by base_discretizer.series_groupy_order
        # grouping modalities in the initial crosstab
        combi_xtab_test = xtab_test.groupby(
            list(map(combination.get_group, xtab_test.index)), dropna=False
        ).sum()

        # checking that all non-nan groups are in TRAIN and TEST
        unq_x = [v for v in unique(combi_xtab.index) if v != notna(v)]
        unq_xtest = [v for v in unique(combi_xtab_test.index) if v != notna(v)]
        viability = all(e in unq_x for e in unq_xtest)
        viability = viability and all(e in unq_xtest for e in unq_x)

        # same target rate order in TRAIN and TEST
        train_target_rate = combi_xtab[1].divide(combi_xtab[0]).sort_values()
        test_target_rate = combi_xtab_test[1].divide(combi_xtab_test[0]).sort_values()
        viability = viability and all(train_target_rate.index == test_target_rate.index)

        # checking that some combinations were provided
        if viability:
            association.update({"combi_xtab_test": combi_xtab_test})

            return association


def get_all_combinations(
    values: GroupedList, max_n_mod: int = None, raw: bool = False
) -> List[GroupedList]:
    """Returns all possible triangular combinations"""

    # maximum number of classes
    q = len(values)

    # desired max number of classes
    if max_n_mod is None:
        max_n_mod = q

    # all possible combinations
    combinations = []
    for n_class in range(2, max_n_mod + 1):
        combinations += get_combinations(n_class, q)

    # getting real feature values
    combinations = [[values[int(c[0]) : int(c[1]) + 1] for c in combi] for combi in combinations]

    # converting back to GroupedList
    if not raw:
        combinations = [
            groupedlist_combination(combination, values) for combination in combinations
        ]

    return combinations


def get_all_nan_combinations(order: GroupedList, str_nan: str, max_n_mod: int) -> List[GroupedList]:
    """all possible combinations of modalities with numpy.nan"""

    # computing all non-NaN combinations
    # case 0: several modalities -> several combinations
    if len(order) > 1:
        combinations = get_all_combinations(order, max_n_mod - 1, raw=True)
    # case 1: unique or no modality -> two combinations
    else:
        combinations = []

    # iterating over each combinations of non-NaNs
    new_combinations = []
    for combi in combinations:
        # NaNs not attributed to a group (own modality)
        new_combinations += [combi + [[str_nan]]]

        # NaNs attributed to a group of non NaNs
        for n, combi_elt in enumerate(combi):
            # grouping NaNs with an existing group
            new_combination = [combi_elt + [str_nan]]

            # adding other groups unchanged
            pre = list(combi[:n])
            nex = list(combi[n + 1 :])
            new_combinations += [pre + new_combination + nex]

    # adding NaN to order
    if str_nan not in order:
        order = order.append(str_nan)

    # converting back to GroupedList
    new_combinations = [
        groupedlist_combination(combination, order) for combination in new_combinations
    ]

    return new_combinations

def combinations_at_index(start_idx, order, nb_remaining_groups, min_group_size=1):
    """Gets all possible combinations of sizes up to the last element of a list"""
    
    # iterating over each possible length of groups
    for size in range(min_group_size, len(order) + 1):
        next_idx = start_idx + size  # index from which to start the next group
        
        # checking that next index is not off the order list
        if next_idx < len(order) + 1:
            # checking that there are remaining groups or that it is the last group
            if (nb_remaining_groups > 1) | (next_idx == len(order)):
                combination = list(order[start_idx : next_idx])
                yield (combination, next_idx, nb_remaining_groups - 1)

def consecutive_combinations(raw_order, max_group_size, min_group_size = 1, nb_remaining_group = None, current_combination = None, next_index = None, all_combinations = None):
    """Computes all possible combinations of values of order up to max_group_size."""

    # initiating recursive attributes
    if current_combination is None:
        current_combination = []
    if next_index is None:
        next_index = 0
    if nb_remaining_group is None:
        nb_remaining_group = max_group_size
    if all_combinations is None:
        all_combinations = []
    
    # getting combinations for next index
    next_combinations = [elt for elt in combinations_at_index(next_index, raw_order, nb_remaining_group, min_group_size)]
    
    # stop case: no next_combinations possible -> adding to all_combinations
    if len(next_combinations) == 0 and min_group_size < len(current_combination) <= max_group_size:
        # saving up combination
        all_combinations += [current_combination]
        
        # resetting remaining number of groups
        nb_remaining_group = max_group_size
        

    # otherwise: adding all next_combinations to the current_combination
    for combination, next_index, current_nb_remaining_group in next_combinations:
        
        # going a rank further in the raw_xtab 
        consecutive_combinations(
            raw_order,
            max_group_size,
            min_group_size=min_group_size,
            nb_remaining_group=current_nb_remaining_group,
            current_combination=current_combination + [combination],
            next_index=next_index,
            all_combinations=all_combinations,
        )
    
    return all_combinations







def plot_stats(stats: DataFrame) -> Tuple[Figure, Axes]:
    """Barplot of the volume and target rate"""

    x = [0] + [elt for e in stats["frequency"].cumsum()[:-1] for elt in [e] * 2] + [1]
    y2 = [elt for e in list(stats["target_rate"]) for elt in [e] * 2]
    s = list(stats.index)
    scaled_y2 = [(y - min(y2)) / (max(y2) - min(y2)) for y in y2]
    c = color_palette("coolwarm", as_cmap=True)(scaled_y2)

    fig, ax = subplots()

    for i in range(len(stats)):
        k = i * 2
        ax.fill_between(x[k : k + 2], [0, 0], y2[k : k + 2], color=c[k])
        ax.text(
            sum(x[k : k + 2]) / 2,
            y2[k],
            s[i],
            rotation=90,
            ha="center",
            va="bottom",
        )

    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set_xlabel("Volume")
    ax.set_ylabel("Target rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0)
    despine()

    return fig, ax
