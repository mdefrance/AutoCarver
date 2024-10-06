""" set of tests for associations within carvers"""

from pandas import DataFrame, Series

from AutoCarver.combinations.utils.combination_evaluator import CombinationEvaluator, filter_nan

# removing abstract parts of CombinationEvaluator
CombinationEvaluator.__abstractmethods__ = set()


def test_filter_nan_with_dataframe():
    xagg = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=["a", "NaN", "c"])
    str_nan = "NaN"
    result = filter_nan(xagg, str_nan)
    expected = DataFrame({"A": [1, 3], "B": [4, 6]}, index=["a", "c"])
    assert result.equals(expected)


def test_filter_nan_with_series():
    xagg = Series([1, 2, 3], index=["a", "NaN", "c"])
    str_nan = "NaN"
    result = filter_nan(xagg, str_nan)
    expected = Series([1, 3], index=["a", "c"])
    assert result.equals(expected)


def test_filter_nan_no_nan_in_index():
    xagg = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}, index=["a", "b", "c"])
    str_nan = "NaN"
    result = filter_nan(xagg, str_nan)
    expected = xagg.copy()
    assert result.equals(expected)


def test_filter_nan_empty_dataframe():
    xagg = DataFrame()
    str_nan = "NaN"
    result = filter_nan(xagg, str_nan)
    expected = DataFrame()
    assert result.equals(expected)


def test_filter_nan_none_input():
    xagg = None
    str_nan = "NaN"
    result = filter_nan(xagg, str_nan)
    expected = None
    assert result == expected


# def test_combination_evaluator_initialization():
#     feature = BaseFeature()  # Assuming BaseFeature is defined elsewhere
#     xagg = DataFrame({
#         'A': [1, 2, 3],
#         'B': [4, 5, 6]
#     })
#     evaluator = CombinationEvaluator(
#         feature=feature,
#         xagg=xagg,
#         sort_by='A',
#         max_n_mod=3,
#         min_freq=0.1
#     )
#     assert evaluator.feature == feature
#     assert evaluator.xagg.equals(xagg)
#     assert evaluator.sort_by == 'A'
#     assert evaluator.max_n_mod == 3
#     assert evaluator.min_freq == 0.1

# def test_combination_evaluator_historize_raw_combination():
#     feature = BaseFeature()  # Assuming BaseFeature is defined elsewhere
#     xagg = DataFrame({
#         'A': [1, 2, 3],
#         'B': [4, 5, 6]
#     })
#     evaluator = CombinationEvaluator(
#         feature=feature,
#         xagg=xagg,
#         sort_by='A',
#         max_n_mod=3,
#         min_freq=0.1
#     )
#     evaluator._historize_raw_combination()
#     # Add assertions based on expected behavior of _historize_raw_combination

# def test_combination_evaluator_group_xagg_by_combinations():
#     feature = BaseFeature()  # Assuming BaseFeature is defined elsewhere
#     xagg = DataFrame({
#         'A': [1, 2, 3],
#         'B': [4, 5, 6]
#     })
#     evaluator = CombinationEvaluator(
#         feature=feature,
#         xagg=xagg,
#         sort_by='A',
#         max_n_mod=3,
#         min_freq=0.1
#     )
#     combinations = [['A'], ['B']]
#     grouped_xaggs = evaluator._group_xagg_by_combinations(combinations)
#     assert len(grouped_xaggs) == 2
#     # Add more assertions based on expected behavior of _group_xagg_by_combinations

# def test_combination_evaluator_compute_associations():
#     feature = BaseFeature()  # Assuming BaseFeature is defined elsewhere
#     xagg = DataFrame({
#         'A': [1, 2, 3],
#         'B': [4, 5, 6]
#     })
#     evaluator = CombinationEvaluator(
#         feature=feature,
#         xagg=xagg,
#         sort_by='A',
#         max_n_mod=3,
#         min_freq=0.1
#     )
#     combinations = [['A'], ['B']]
#     grouped_xaggs = evaluator._group_xagg_by_combinations(combinations)
#     associations = evaluator._compute_associations(grouped_xaggs)
#     assert isinstance(associations, list)
#     assert len(associations) == 2
#     # Add more assertions based on expected behavior of _compute_associations

# def test_combination_evaluator_test_viability_train():
#     feature = BaseFeature()  # Assuming BaseFeature is defined elsewhere
#     xagg = DataFrame({
#         'A': [1, 2, 3],
#         'B': [4, 5, 6]
#     })
#     evaluator = CombinationEvaluator(
#         feature=feature,
#         xagg=xagg,
#         sort_by='A',
#         max_n_mod=3,
#         min_freq=0.1
#     )
#     combination = {'xagg': xagg}
#     viability = evaluator._test_viability_train(combination)
#     assert 'train' in viability
#     # Add more assertions based on expected behavior of _test_viability_train

# def test_combination_evaluator_test_viability_dev():
#     feature = BaseFeature()  # Assuming BaseFeature is defined elsewhere
#     xagg = DataFrame({
#         'A': [1, 2, 3],
#         'B': [4, 5, 6]
#     })
#     xagg_dev = DataFrame({
#         'A': [1, 2, 3],
#         'B': [4, 5, 6]
#     })
#     evaluator = CombinationEvaluator(
#         feature=feature,
#         xagg=xagg,
#         sort_by='A',
#         max_n_mod=3,
#         min_freq=0.1,
#         xagg_dev=xagg_dev
#     )
#     combination = {'xagg': xagg}
#     test_results = evaluator._test_viability_train(combination)
#     viability = evaluator._test_viability_dev(test_results, combination)
#     assert 'dev' in viability
#     # Add more assertions based on expected behavior of _test_viability_dev

# def test_combination_evaluator_get_viable_combination():
#     feature = BaseFeature()  # Assuming BaseFeature is defined elsewhere
#     xagg = DataFrame({
#         'A': [1, 2, 3],
#         'B': [4, 5, 6]
#     })
#     evaluator = CombinationEvaluator(
#         feature=feature,
#         xagg=xagg,
#         sort_by='A',
#         max_n_mod=3,
#         min_freq=0.1
#     )
#     combinations = [['A'], ['B']]
#     grouped_xaggs = evaluator._group_xagg_by_combinations(combinations)
#     associations = evaluator._compute_associations(grouped_xaggs)
#     viable_combination = evaluator._get_viable_combination(associations)
#     # Add assertions based on expected behavior of _get_viable_combination

# def test_combination_evaluator_get_best_association():
#     feature = BaseFeature()  # Assuming BaseFeature is defined elsewhere
#     xagg = DataFrame({
#         'A': [1, 2, 3],
#         'B': [4, 5, 6]
#     })
#     evaluator = CombinationEvaluator(
#         feature=feature,
#         xagg=xagg,
#         sort_by='A',
#         max_n_mod=3,
#         min_freq=0.1
#     )
#     combinations = [['A'], ['B']]
#     best_association = evaluator._get_best_association(combinations)
#     # Add assertions based on expected behavior of _get_best_association

# def test_combination_evaluator_apply_best_combination():
#     feature = BaseFeature()  # Assuming BaseFeature is defined elsewhere
#     xagg = DataFrame({
#         'A': [1, 2, 3],
#         'B': [4, 5, 6]
#     })
#     evaluator = CombinationEvaluator(
#         feature=feature,
#         xagg=xagg,
#         sort_by='A',
#         max_n_mod=3,
#         min_freq=0.1
#     )
#     best_association = {'combination': ['A']}
#     evaluator._apply_best_combination(best_association)
#     # Add assertions based on expected behavior of _apply_best_combination

# def test_combination_evaluator_get_best_combination_non_nan():
#     feature = BaseFeature()  # Assuming BaseFeature is defined elsewhere
#     xagg = DataFrame({
#         'A': [1, 2, 3],
#         'B': [4, 5, 6]
#     })
#     evaluator = CombinationEvaluator(
#         feature=feature,
#         xagg=xagg,
#         sort_by='A',
#         max_n_mod=3,
#         min_freq=0.1
#     )
#     best_combination = evaluator._get_best_combination_non_nan()
#     # Add assertions based on expected behavior of _get_best_combination_non_nan

# def test_combination_evaluator_get_best_combination_with_nan():
#     feature = BaseFeature()  # Assuming BaseFeature is defined elsewhere
#     xagg = DataFrame({
#         'A': [1, 2, 3],
#         'B': [4, 5, 6]
#     })
#     evaluator = CombinationEvaluator(
#         feature=feature,
#         xagg=xagg,
#         sort_by='A',
#         max_n_mod=3,
#         min_freq=0.1
#     )
#     best_combination = {'combination': ['A']}
#     best_combination_with_nan = evaluator._get_best_combination_with_nan(best_combination)
#     # Add assertions based on expected behavior of _get_best_combination_with_nan

# def test_combination_evaluator_get_best_combination():
#     feature = BaseFeature()  # Assuming BaseFeature is defined elsewhere
#     xagg = DataFrame({
#         'A': [1, 2, 3],
#         'B': [4, 5, 6]
#     })
#     evaluator = CombinationEvaluator(
#         feature=feature,
#         xagg=xagg,
#         sort_by='A',
#         max_n_mod=3,
#         min_freq=0.1
#     )
#     best_combination = evaluator.get_best_combination()
#     # Add assertions based on expected behavior of get_best_combination
