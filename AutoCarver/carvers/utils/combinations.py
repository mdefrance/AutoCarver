"""Tool to build optimized buckets out of Quantitative and Qualitative features
for any task.
"""

from typing import Any
from pandas import DataFrame
from ...features import GroupedList


def combinations_at_index(
    start_idx: int, order: list[Any], nb_remaining_groups: int, min_group_size: int = 1
) -> tuple[list[Any], int, int]:
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
    raw_order: list[Any],
    max_group_size: int,
    min_group_size: int = 1,
    nb_remaining_group: int = None,
    current_combination: list[Any] = None,
    next_index: int = None,
    all_combinations: list[list[Any]] = None,
) -> list[list[Any]]:
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


def nan_combinations(
    raw_order: GroupedList,
    str_nan: str,
    max_n_mod: int,
) -> list[list[str]]:
    """All consecutive combinatios of non-nans with added nan to each possible group and a last
    group only with nan if the max_n_mod is not reached by the combination
    """
    # all possible consecutive combinations
    combinations = consecutive_combinations(raw_order, max_n_mod, min_group_size=1)
    # iterating over each combination
    nan_combis = []
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

        nan_combis += nan_combination

    return nan_combis


def order_apply_combination(order: GroupedList, combination: list[list[Any]]) -> GroupedList:
    """Converts a list of combination to a GroupedList

    Parameters
    ----------
    order : GroupedList
        _description_
    combination : list[list[Any]]
        _description_

    Returns
    -------
    GroupedList
        _description_
    """
    order_copy = GroupedList(order)
    for combi in combination:
        order_copy.group_list(combi, combi[0])

    return order_copy


def xagg_apply_combination(xagg: DataFrame, order: GroupedList) -> DataFrame:
    """Applies an order (combination) to a crosstab

    Parameters
    ----------
    xagg : DataFrame
        Crosstab
    order : GroupedList
        Combination of index to apply to the crosstab

    Returns
    -------
    dict[str, Any]
        Orderd crosstab.
    """
    # checking for input values
    combi_xagg = None
    if xagg is not None:
        # grouping modalities in the crosstab
        groups = list(map(order.get_group, xagg.index))
        combi_xagg = xagg.groupby(groups, dropna=False, sort=False).sum()

    return combi_xagg
