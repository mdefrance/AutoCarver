"""Tool to build optimized buckets out of Quantitative and Qualitative features
for any task.
"""

from typing import Any, Generator

from pandas import DataFrame

from ...features import BaseFeature, GroupedList


def combinations_at_index(
    start_index: int, elements: list[Any], remaining_groups: int
) -> Generator[list[Any], int, int]:
    """Gets all possible combinations of sizes up to the last element of a list"""

    # iterating over each possible length of groups
    for size in range(1, len(elements) + 1):
        next_index = start_index + size  # index from which to start the next group

        # checking that next index is not off the elements list
        if next_index < len(elements) + 1:
            # checking that there are remaining groups or that it is the last group
            if (remaining_groups > 1) | (next_index == len(elements)):
                combination = list(elements[start_index:next_index])
                yield (combination, next_index, remaining_groups - 1)


def consecutive_combinations(
    raw_order: list[Any],
    max_group_size: int,
    remaining_groups: int = None,
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
    if remaining_groups is None:
        remaining_groups = max_group_size
    if all_combinations is None:
        all_combinations = []

    # getting combinations for next index
    next_combinations = list(combinations_at_index(next_index, raw_order, remaining_groups))

    # stop case: no next_combinations possible -> adding to all_combinations
    if len(next_combinations) == 0 and 1 < len(current_combination) <= max_group_size:
        # saving up combination
        all_combinations += [current_combination]

        # resetting remaining number of groups
        remaining_groups = max_group_size

    # otherwise: adding all next_combinations to the current_combination
    for combination, new_next_index, new_remaining_groups in next_combinations:
        # going a rank further in the raw_xtab
        consecutive_combinations(
            raw_order,
            max_group_size,
            remaining_groups=new_remaining_groups,
            current_combination=current_combination + [combination],
            next_index=new_next_index,
            all_combinations=all_combinations,
        )

    return all_combinations


def nan_combinations(
    feature: BaseFeature,
    max_n_mod: int,
) -> list[list[str]]:
    """All consecutive combinations of non-nans with added nan to each possible group and a last
    group only with nan if the max_n_mod is not reached by the combination

    - feature must have has_nan = True
    - feature must have dropna = True
    - len(feature.labels) <= max_n_mod
    """
    # raw ordering without nans
    raw_labels = GroupedList(feature.labels[:])
    raw_labels.remove(feature.nan)  # nans are added within nan_combinations

    # all possible consecutive combinations
    combinations = consecutive_combinations(raw_labels, max_n_mod)

    # iterating over each combination
    nan_combis = []
    for combination in combinations:
        # adding nan to each group of the combination
        nan_combination = []
        for n in range(len(combination)):
            # copying input combination
            new_combination = combination[:]
            # adding nan to the nth group
            new_combination[n] = new_combination[n] + [feature.nan]
            # storing updated combination with attributed group to nan
            nan_combination += [new_combination]

        # if max_n_mod is not reached adding a combination with nans alone
        if len(combination) < max_n_mod:
            # copying input combination
            new_combination = combination[:]
            # adding a group for nans only
            nan_combination += [new_combination + [[feature.nan]]]

        nan_combis += nan_combination

    # adding a combination for all modalities vs nans
    nan_combis += [[list(raw_labels), [feature.nan]]]

    return nan_combis


def order_apply_combination(order: GroupedList, combination: list[list[Any]]) -> GroupedList:
    """Converts a list of combination to a GroupedList"""
    order_copy = GroupedList(order)
    for combi in combination:
        order_copy.group(combi, combi[0])

    return order_copy


def xagg_apply_combination(xagg: DataFrame, feature: BaseFeature) -> DataFrame:
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
        Ordered crosstab.
    """
    # checking for input values
    combi_xagg = None
    if xagg is not None:
        # grouping modalities in the crosstab
        groups = list(map(feature.label_per_value.get, xagg.index))
        combi_xagg = xagg.groupby(groups, dropna=False, sort=False).sum()

        # reindexing to ensure the right labels
        revised_index = feature.labels
        if feature.has_nan and feature.nan not in revised_index and not feature.dropna:
            revised_index += [feature.nan]

        combi_xagg.index = revised_index

    return combi_xagg


def combination_formatter(combination: list[list[str]]) -> dict[str, str]:
    """Attributes the first element of a group to all elements of a group"""
    return {modal: group[0] for group in combination for modal in group}


def format_combinations(combinations: list[list[str]]) -> list[dict[str, str]]:
    """Formats a list of combinations"""
    return [combination_formatter(combination) for combination in combinations]
