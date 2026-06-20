"""Tool to build optimized buckets out of Quantitative and Qualitative features
for any task.
"""

from collections.abc import Generator
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from AutoCarver.features import BaseFeature, GroupedList

if TYPE_CHECKING:
    # typing-only: avoids a runtime import cycle (combination_evaluator imports this module)
    from AutoCarver.combinations.utils.combination_evaluator import AggregatedSample


def combinations_at_index(
    start_index: int, elements: list[Any] | GroupedList, remaining_groups: int
) -> Generator[tuple[list[Any], int, int], None, None]:
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
    raw_order: "list[Any] | GroupedList",
    max_group_size: int,
) -> Generator[list[list[Any]], None, None]:
    """Yields all combinations of consecutive values of size up to ``max_group_size``.

    Yields one combination at a time (a list of groups, each group a list of
    contiguous values from ``raw_order``) so callers can stream them through
    grouping + scoring without ever materialising the full enumeration. The
    enumeration order is identical to the previous list-returning version.
    """
    yield from _consecutive_combinations_iter(raw_order, max_group_size, [], 0, max_group_size)


def _consecutive_combinations_iter(
    raw_order: "list[Any] | GroupedList",
    max_group_size: int,
    current_combination: list[list[Any]],
    next_index: int,
    remaining_groups: int,
) -> Generator[list[list[Any]], None, None]:
    """Recursive generator backing :func:`consecutive_combinations`.

    A combination is emitted (a) once we've consumed the whole ``raw_order``
    AND it contains ≥ 2 groups, OR (b) we cannot extend further. Note this
    mirrors the previous "no next_combinations + 1 < len(current) ≤
    max_group_size" stop condition exactly so the set of produced
    combinations is unchanged.
    """
    produced_child = False
    for combination, new_next_index, new_remaining_groups in combinations_at_index(
        next_index, raw_order, remaining_groups
    ):
        produced_child = True
        yield from _consecutive_combinations_iter(
            raw_order,
            max_group_size,
            current_combination + [combination],
            new_next_index,
            new_remaining_groups,
        )

    if not produced_child and 1 < len(current_combination) <= max_group_size:
        yield current_combination


def nan_combinations(
    feature: BaseFeature,
    max_n_mod: int,
) -> Generator[list[list[str]], None, None]:
    """Streams consecutive combinations augmented with nan placements.

    For each consecutive combination of the non-nan labels, yields:

    * one variant per group with the nan label folded into that group, and
    * one variant where the nan label forms its own additional group, when
      ``len(combination) < max_n_mod``.

    Finally yields the all-vs-nan partition. Order is preserved exactly from
    the previous list-returning version (so the carver picks the same first
    viable combination it would have before).

    Requires ``feature.has_nan == True`` and ``feature.dropna == True``.
    """
    # raw ordering without nans
    if feature.labels is None:
        raise RuntimeError(f"[nan_combinations] feature {feature} has no labels populated")
    raw_labels = GroupedList(feature.labels[:])
    raw_labels.remove(feature.nan)  # nans are added within nan_combinations

    for combination in consecutive_combinations(raw_labels, max_n_mod):
        # adding nan to each group of the combination
        for n in range(len(combination)):
            new_combination = combination[:]
            new_combination[n] = new_combination[n] + [feature.nan]
            yield new_combination

        # if max_n_mod is not reached adding a combination with nans alone
        if len(combination) < max_n_mod:
            yield combination[:] + [[feature.nan]]

    # adding a combination for all modalities vs nans
    yield [list(raw_labels), [feature.nan]]


def order_apply_combination(order: "GroupedList | list[Any] | None", combination: list[list[Any]]) -> GroupedList:
    """Converts a list of combination to a GroupedList"""
    if order is None:
        raise ValueError("[order_apply_combination] order must not be None")
    order_copy = GroupedList(order)
    for combi in combination:
        order_copy.group(combi, combi[0])

    return order_copy


def xagg_apply_combination(
    xagg: pd.Series | pd.DataFrame | None, feature: BaseFeature
) -> pd.Series | pd.DataFrame | None:
    """Applies an order (combination) to a crosstab

    Parameters
    ----------
    xagg : pd.DataFrame
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

        # reindexing to ensure the right labels — NOTE: this intentionally mutates
        # feature.labels (in-place +=) when the nan needs to be appended; downstream
        # consumers rely on that mutation.
        if feature.labels is None:
            raise RuntimeError(f"[xagg_apply_combination] feature {feature} has no labels populated")
        revised_index = feature.labels
        if feature.has_nan and feature.nan not in revised_index and not feature.dropna:
            revised_index += [feature.nan]

        combi_xagg.index = revised_index

    return combi_xagg


def group_crosstab(xagg: "AggregatedSample | pd.DataFrame", groupby: dict) -> "pd.DataFrame":
    """Sums a crosstab's rows into the groups defined by ``groupby``.

    ``groupby`` maps each raw modality to its group leader. Group leaders are
    ordered by first appearance (ordinal order), keeping the result independent
    of the cosmetic label strings — the same contract as
    :meth:`BinaryCombinationEvaluator._grouper`, shared here so crosstab-based
    evaluators (binary, ordinal) don't each reimplement it.

    Accepts anything exposing ``.index`` and ``.groupby`` (a ``pd.DataFrame`` or
    an :class:`AggregatedSample`); always returns a ``pd.DataFrame``.
    """
    leaders = [groupby.get(index_value, index_value) for index_value in xagg.index]
    # pass the key as an object ndarray (not a list): a list whose values happen
    # to match column labels would be read by pandas as "group by these columns"
    # rather than as a per-row grouping key.
    return xagg.groupby(np.asarray(leaders, dtype=object), sort=False).sum()


def combination_formatter(combination: list[list[str]]) -> dict[str, str]:
    """Attributes the first element of a group to all elements of a group"""
    return {modal: group[0] for group in combination for modal in group}


def format_combinations(combinations: list[list[list[str]]]) -> list[dict[str, str]]:
    """Formats a list of combinations (each combination is a list of groups, each group a list of modalities)."""
    return [combination_formatter(combination) for combination in combinations]
