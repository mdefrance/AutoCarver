"""Tool to build optimized buckets out of Quantitative and Qualitative features
for a binary classification model.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
from IPython.display import display_html  # TODO: remove this
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.pyplot import subplots
from matplotlib.ticker import PercentFormatter
from pandas import DataFrame, Series, crosstab, unique
from scipy.stats import chi2_contingency
from seaborn import color_palette, despine
from tqdm import tqdm

from .discretizers.discretizers import Discretizer
from .discretizers.utils.base_discretizers import (
    GroupedList,
    GroupedListDiscretizer,
    convert_to_labels,
    convert_to_values,
    is_equal,
)


# TODO: display tables
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
        min_carved_freq: float = 0,  # TODO: update this parameter so that it is set according to frequency rather than number of groups
        sort_by: str = "tschuprowt",
        str_nan: str = "__NAN__",
        str_default: str = "__OTHER__",
        output_dtype: str = 'float',
        dropna: bool = True,
        copy: bool = False,
        verbose: bool = True,
    ) -> None:
        """_summary_

        Parameters
        ----------
        quantitative_features : List[str]
            _description_
        qualitative_features : List[str]
            _description_
        min_freq : float
            _description_
        ordinal_features : List[str], optional
            _description_, by default None
        values_orders : Dict[str, GroupedList], optional
            _description_, by default None
        max_n_mod : int, optional
            _description_, by default 5
        min_carved_freq : float, optional
            _description_, by default 0
        str_nan : str, optional
            _description_, by default "__NAN__"
        str_default : str, optional
            _description_, by default "__OTHER__"
        output_dtype : str, optional
            _description_, by default 'float'
        dropna : bool, optional
            _description_, by default True
        copy : bool, optional
            _description_, by default False
        verbose : bool, optional
            _description_, by default True
        """
        # Lists of features
        self.features = list(set(quantitative_features + qualitative_features + ordinal_features))
        if ordinal_features is None:
            ordinal_features = []
        self.ordinal_features = list(set(ordinal_features))

        # initializing input_dtypes
        self.input_dtypes = {feature: "str" for feature in qualitative_features + ordinal_features}
        self.input_dtypes.update({feature: "float" for feature in quantitative_features})

        # Initiating GroupedListDiscretizer
        super().__init__(
            features=self.features,
            values_orders=values_orders,
            input_dtypes=self.input_dtypes,
            output_dtype=output_dtype,
            str_nan=str_nan,
            dropna=dropna,
            copy=copy,
        )

        # class specific attributes
        self.min_freq = min_freq  # minimum frequency per base bucket
        self.max_n_mod = max_n_mod  # maximum number of modality per feature
        self.dropna = dropna  # whether or not to group NaNs with other modalities
        self.verbose = verbose
        self.str_default = str_default
        self.min_carved_freq = min_carved_freq
        self.min_group_size = 1
        measures = ["tschuprowt", "cramerv"]  # association measure used to find the best groups
        assert (
            sort_by in measures
        ), f"""Measure '{sort_by}' not yet implemented. Choose from: {str(measures)}."""
        self.sort_by = sort_by

    def prepare_data(
        self,
        X: DataFrame,
        y: Series,
        X_test: DataFrame = None,
        y_test: Series = None,
    ) -> Tuple[DataFrame, DataFrame]:
        """Checks validity of provided data

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

        Returns
        -------
        Tuple[DataFrame, DataFrame]
            Copies of (X, X_test)
        """
        # Checking for binary target and copying X
        x_copy = super().prepare_data(X, y)
        x_test_copy = super().prepare_data(X_test, y_test)

        return x_copy, x_test_copy

    def remove_feature(self, feature: str) -> None:
        """Removes a feature from all instances 

        Parameters
        ----------
        feature : str
            Column name
        """        
        if feature in self.features:
            super().remove_feature(feature)
            if feature in self.ordinal_features:
                self.ordinal_features.remove(feature)

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
            quantitative_features=self.quantitative_features,
            qualitative_features=self.qualitative_features,
            min_freq=self.min_freq,
            ordinal_features=self.ordinal_features,
            values_orders=self.values_orders,
            str_nan=self.str_nan,
            str_default=self.str_default,
            copy=False,
            verbose=self.verbose,
        )
        x_copy = discretizer.fit_transform(x_copy, y)
        if x_test_copy is not None:
            x_test_copy = discretizer.transform(x_test_copy, y_test)
        self.input_dtypes.update(discretizer.input_dtypes)  # saving data types

        # updating values_orders according to base bucketization
        self.values_orders.update(discretizer.values_orders)

        # removing dropped features
        for feature in self.features:
            if feature not in discretizer.values_orders:
                self.remove_feature(feature)

        # converting potential quantiles into there respective labels
        labels_orders = convert_to_labels(
            features=self.features,
            quantitative_features=self.quantitative_features,
            values_orders=self.values_orders,
            str_nan=self.str_nan,
            dropna=False,
        )

        # computing crosstabs for each feature on train/test
        xtabs = get_xtabs(self.features, x_copy, y, labels_orders)
        xtabs_test = get_xtabs(self.features, x_test_copy, y_test, labels_orders)

        # optimal butcketization/carving of each feature
        for n, feature in enumerate(self.features):
            if self.verbose:  # verbose if requested
                print(f"\n---\n[AutoCarver] Fit {feature} ({n+1}/{len(self.features)})")

            # getting xtabs on train/test
            xtab = xtabs[feature]
            xtab_test = xtabs_test[feature]
            if self.verbose:  # verbose if requested
                print(xtab)

            # ordering
            order = labels_orders[feature]

            # getting best combination
            best_combination = self.get_best_combination(order, xtab, xtab_test=xtab_test)

            # checking that a suitable combination has been found
            if best_combination is not None:
                order, xtab, xtab_test = best_combination
                if self.verbose:  # verbose if requested
                    print(xtab)

                # updating label_orders
                labels_orders.update({feature: order})
            
            # no suitable combination has been found -> removing feature
            else:
                print(f"No robust combination for feature '{feature}' could be found. It will be ignored. You might have to increase the size of your test sample (test sample not representative of test sample for this feature) or you should consider dropping this features.")
                self.remove_feature(feature)
                if feature in labels_orders:
                    labels_orders.pop(feature)


        # converting potential labels into there respective values (quantiles)
        self.values_orders.update(
            convert_to_values(
                features=self.features,
                quantitative_features=self.quantitative_features,
                values_orders=self.values_orders,
                label_orders=labels_orders,
                str_nan=self.str_nan,
            )
        )

        # TODO pretty displaying

        # discretizing features based on each feature's values_order
        super().fit(X, y)

        return self

    def get_best_combination(
        self,
        order: GroupedList,
        xtab: DataFrame,
        *,
        xtab_test: DataFrame = None,
    ) -> Tuple[GroupedList, DataFrame, DataFrame]:
        # raw ordering
        raw_order = GroupedList(order)
        if self.str_nan in raw_order:
            raw_order.remove(self.str_nan)

        # filtering out nans if requested from train/test crosstabs
        raw_xtab = filter_nan_xtab(xtab, self.str_nan)
        raw_xtab_test = filter_nan_xtab(xtab_test, self.str_nan)

        # checking for non-nan values
        best_association = None
        if raw_xtab.shape[0] > 1:
            # all possible consecutive combinations
            combinations = consecutive_combinations(
                raw_order, self.max_n_mod, min_group_size=self.min_group_size
            )

            # getting most associated combination
            best_association = get_best_association(
                raw_xtab,
                combinations,
                sort_by=self.sort_by,
                xtab_test=raw_xtab_test,
                verbose=self.verbose,
            )

            # applying best_combination to order and xtabs
            if best_association is not None:
                order = order_apply_combination(order, best_association["combination"])
                xtab = xtab_apply_order(xtab, order)
                xtab_test = xtab_apply_order(xtab_test, order)

            # grouping NaNs if requested to drop them (dropna=True)
            if self.dropna and self.str_nan in order and best_association is not None:
                # raw ordering without nans
                raw_order = GroupedList(order)
                raw_order.remove(self.str_nan)

                # all possible consecutive combinations
                combinations = consecutive_combinations(
                    raw_order, self.max_n_mod, min_group_size=self.min_group_size
                )

                # adding combinations with NaNs
                nan_combinations = add_nan_in_combinations(
                    combinations, self.str_nan, self.max_n_mod
                )

                # getting most associated combination
                best_association = get_best_association(
                    xtab,
                    nan_combinations,
                    sort_by=self.sort_by,
                    xtab_test=xtab_test,
                    verbose=self.verbose,
                )

                # applying best_combination to order and xtab
                if best_association is not None:
                    order = order_apply_combination(order, best_association["combination"])
                    xtab = xtab_apply_order(xtab, order)
                    xtab_test = xtab_apply_order(xtab_test, order)

        # checking that a suitable combination has been found
        if best_association is not None:
            return order, xtab, xtab_test

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


def filter_nan_xtab(xtab: DataFrame, str_nan: str) -> DataFrame:
    """Filters out nans from the crosstab"""

    # cehcking for values in crosstab
    filtered_xtab = None
    if xtab is not None:
        # filtering out nans if requested from train crosstab
        filtered_xtab = xtab.copy()
        if str_nan in xtab.index:
            filtered_xtab = xtab.drop(str_nan, axis=0)

    return filtered_xtab


def get_xtabs(
    features: List[str], X: DataFrame, y: Series, labels_orders: Dict[str, GroupedList]
) -> Dict[str, DataFrame]:
    """Computes crosstabs for specified features and ensures that the crosstab is ordered according to the known labels"""

    # checking for empty datasets
    xtabs = {feature: None for feature in features}
    if X is not None:
        # crosstab for each feature
        for feature in features:
            # computing crosstab with str_nan
            xtab = crosstab(X[feature], y)

            # reordering according to known_order
            xtab = xtab.reindex(labels_orders[feature])  # TODO: fill nans for x_test?

            # storing results
            xtabs.update({feature: xtab})

    return xtabs


def association_xtab(xtab: DataFrame, n_obs, n_mod_y) -> Dict[str, float]:
    """Computes measures of association between feature x and feature2."""

    # number of values taken by the features
    n_mod_x = xtab.shape[0]

    # Chi2 statistic
    chi2 = chi2_contingency(xtab)[0]

    # Cramer's V
    cramerv = np.sqrt(chi2 / n_obs / (n_mod_y - 1))

    # Tschuprow's T
    tschuprowt = np.sqrt(chi2 / n_obs / np.sqrt((n_mod_x - 1) * (n_mod_y - 1)))

    return {"cramerv": cramerv, "tschuprowt": tschuprowt}


def vectorized_groupby_sum(xtab: DataFrame, groupby: List[str]):
    """Groups a crosstab by groupby and sums column values by groups"""

    # all indices that may be duplicated
    index_values = np.array(groupby)

    # all unique indices deduplicated
    unique_indices = np.unique(index_values)

    # initiating summed up array with zeros
    summed_values = np.zeros((len(unique_indices), len(xtab.columns)))

    # for each unique_index found in index_values sums xtab.Values at corresponding position in summed_values
    np.add.at(summed_values, np.searchsorted(unique_indices, index_values), xtab.values)

    # converting back to dataframe
    grouped_xtab = DataFrame(summed_values, index=unique_indices, columns=xtab.columns)

    return grouped_xtab


def combinations_at_index(start_idx, order, nb_remaining_groups, min_group_size=1):
    """Gets all possible combinations of sizes up to the last element of a list"""

    # iterating over each possible length of groups
    for size in range(min_group_size, len(order) + 1):
        next_idx = start_idx + size  # index from which to start the next group

        # checking that next index is not off the order list
        if next_idx < len(order) + 1:
            # checking that there are remaining groups or that it is the last group
            if (nb_remaining_groups > 1) | (next_idx == len(order)):
                combination = list(order[start_idx:next_idx])
                yield (combination, next_idx, nb_remaining_groups - 1)


def consecutive_combinations(
    raw_order,
    max_group_size,
    min_group_size=1,
    nb_remaining_group=None,
    current_combination=None,
    next_index=None,
    all_combinations=None,
):
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
    next_combinations = [
        elt
        for elt in combinations_at_index(next_index, raw_order, nb_remaining_group, min_group_size)
    ]

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


def xtab_target_rate(xtab: DataFrame) -> DataFrame:
    """Computes target rate per row for a binary target (column) in a crosstab"""

    return xtab[1].divide(xtab[0]).sort_values()


def get_best_association(
    xtab: DataFrame,
    combinations: List[List[str]],
    sort_by: str,
    xtab_test: DataFrame = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Computes associations of the xtab for each combination"""

    # values to groupby indices with
    indices_to_groupby = [
        [value for values in ([group[0]] * len(group) for group in combination) for value in values]
        for combination in combinations
    ]

    # grouping xtab by its indices
    grouped_xtabs = [
        vectorized_groupby_sum(xtab, index_to_groupby)
        for index_to_groupby in tqdm(
            indices_to_groupby, disable=not verbose, desc="Grouping modalities   "
        )
    ]

    # computing associations for each xtabs
    n_obs = xtab.sum().sum()  # number of observation
    n_mod_y = len(xtab.columns)  # number of modalities and minimum number of modalities
    associations_xtab = [
        association_xtab(grouped_xtab, n_obs, n_mod_y)
        for grouped_xtab in tqdm(grouped_xtabs, disable=not verbose, desc="Computing associations")
    ]

    # adding corresponding combination to the association
    for combination, index_to_groupby, association, grouped_xtab in zip(
        combinations, indices_to_groupby, associations_xtab, grouped_xtabs
    ):
        association.update(
            {"combination": combination, "index_to_groupby": index_to_groupby, "xtab": grouped_xtab}
        )

    # sorting associations according to specified metric
    associations_xtab = (
        DataFrame(associations_xtab).sort_values(sort_by, ascending=False).to_dict(orient="records")
    )

    # case 0: no test sample provided -> not testing for robustness
    if xtab_test is None:
        return associations_xtab[0]

    # case 1: testing viability on provided test sample
    for association in tqdm(associations_xtab, disable=not verbose, desc="Testing robustness    "):
        # needed parameters
        index_to_groupby, xtab = (
            association["index_to_groupby"],
            association["xtab"],
        )

        # grouping rows of the test crosstab
        grouped_xtab_test = vectorized_groupby_sum(xtab_test, index_to_groupby)

        # computing target rate ranks per value
        train_ranks = xtab_target_rate(xtab).index
        test_ranks = xtab_target_rate(grouped_xtab_test).index

        # viable on test sample: grouped values have the same ranks in train/test
        if all(train_ranks == test_ranks):
            return association


def add_nan_in_combinations(
    combinations: List[List[str]], str_nan: str, max_n_mod: int
) -> List[List[str]]:
    """Adds nan to each possible group and a last group only with nan if the max_n_mod is not reached by the combination"""

    # iterating over each combination
    nan_combinations = []
    for combination in combinations:
        # adding nan to each group of the combination
        nan_combination = []
        for n in range(len(combination)):
            # copying input combination
            new_combination = combination[:]
            # adding nan to the nth group
            new_combination[n] = new_combination[n] + [str_nan]
            # storing updated combination with attributed group to nan
            nan_combination += [new_combination]

        # if max_n_mod is not reached adding a combination with nans alone
        if len(combination) < max_n_mod:
            # copying input combination
            new_combination = combination[:]
            # adding a group for nans only
            nan_combination += [new_combination + [[str_nan]]]

        nan_combinations += nan_combination

    return nan_combinations


def order_apply_combination(order: GroupedList, combination: List[List[Any]]) -> GroupedList:
    """Converts a list of combination to a GroupedList"""

    order_copy = GroupedList(order)
    for combi in combination:
        order_copy.group_list(combi, combi[0])

    return order_copy


def xtab_apply_order(xtab: DataFrame, order: GroupedList) -> DataFrame:
    """Applies an order (combination) to a crosstab

    Parameters
    ----------
    xtab : DataFrame
        Crosstab
    order : GroupedList
        Combination of index to apply to the crosstab

    Returns
    -------
    Dict[str, Any]
        _description_
    """
    # checking for input values
    combi_xtab = None
    if xtab is not None:
        # grouping modalities in the crosstab
        groups = list(map(order.get_group, xtab.index))
        combi_xtab = xtab.groupby(groups, dropna=False, sort=False).sum()

    return combi_xtab


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
