"""Property-based tests for the selectors' fit/transform contract.

Source: ``AutoCarver.selectors.utils.base_selector`` exercised through the
concrete :class:`ClassificationSelector` / :class:`RegressionSelector`. Checks
the per-type budget, subset/no-duplicate selection, row/column transform
invariants, the unfitted guards and determinism.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from strategies import clone_features, dataframe_and_features

from AutoCarver.selectors import ClassificationSelector, RegressionSelector


def _split_counts(selected):
    """(#quantitative, #qualitative) among selected features."""
    quanti = sum(1 for f in selected if f.is_quantitative)
    quali = sum(1 for f in selected if f.is_qualitative)
    return quanti, quali


@st.composite
def classification_problem(draw):
    X, features, y = draw(dataframe_and_features("binary"))
    n_best = draw(st.integers(min_value=1, max_value=len(features)))
    return X, features, y, n_best


@st.composite
def regression_problem(draw):
    X, features, y = draw(dataframe_and_features("continuous"))
    n_best = draw(st.integers(min_value=1, max_value=len(features)))
    return X, features, y, n_best


# --------------------------------------------------------------------------
# __init__ guards
# --------------------------------------------------------------------------
@given(dataframe_and_features("binary"))
@settings(max_examples=20)
def test_init_rejects_out_of_range_n_best(problem):
    """n_best_per_type must satisfy 0 < n_best_per_type <= len(features)."""
    _, features, _ = problem
    with pytest.raises(ValueError):
        ClassificationSelector(features, 0)
    with pytest.raises(ValueError):
        ClassificationSelector(features, len(features) + 1)


# --------------------------------------------------------------------------
# fit contract
# --------------------------------------------------------------------------
@given(classification_problem())
@settings(max_examples=20, deadline=None)
def test_fit_selects_subset_within_budget(problem):
    """After fit: fitted flag set; selection is a duplicate-free subset of the
    input features and respects the per-type budget."""
    X, features, y, n_best = problem
    input_versions = features.versions

    selector = ClassificationSelector(features, n_best)
    selector.fit(X, y)

    assert selector.is_fitted is True

    selected = list(selector.selected_features)
    selected_versions = [f.version for f in selected]

    # subset of inputs, no duplicates
    assert set(selected_versions).issubset(set(input_versions))
    assert len(selected_versions) == len(set(selected_versions))

    # per-type budget respected
    quanti, quali = _split_counts(selected)
    assert quanti <= n_best
    assert quali <= n_best


@given(classification_problem())
@settings(max_examples=20, deadline=None)
def test_transform_preserves_rows_and_restricts_columns(problem):
    """transform keeps every row and outputs exactly the selected versions
    (``selected_features`` reorders by type, so compare as a column set)."""
    X, features, y, n_best = problem
    selector = ClassificationSelector(features, n_best)
    selector.fit(X, y)

    transformed = selector.transform(X)
    selected_versions = {f.version for f in selector.selected_features}

    assert len(transformed) == len(X)
    assert list(transformed.index) == list(X.index)
    assert set(transformed.columns) == selected_versions
    assert set(transformed.columns).issubset(set(X.columns))


# --------------------------------------------------------------------------
# unfitted guards
# --------------------------------------------------------------------------
@given(classification_problem())
@settings(max_examples=20)
def test_unfitted_guards(problem):
    """transform / selected_features before fit raise RuntimeError."""
    X, features, _, n_best = problem
    selector = ClassificationSelector(features, n_best)
    with pytest.raises(RuntimeError):
        selector.transform(X)
    with pytest.raises(RuntimeError):
        _ = selector.selected_features


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------
@given(classification_problem())
@settings(max_examples=15, deadline=None)
def test_classification_selection_deterministic(problem):
    """Two selectors with identical inputs select identical features."""
    X, features, y, n_best = problem
    features2 = clone_features(features)

    first = ClassificationSelector(features, n_best).fit(X, y)
    second = ClassificationSelector(features2, n_best).fit(X, y)

    v1 = [f.version for f in first.selected_features]
    v2 = [f.version for f in second.selected_features]
    assert v1 == v2


@given(regression_problem())
@settings(max_examples=15, deadline=None)
def test_regression_selection_deterministic_and_subset(problem):
    """RegressionSelector also yields a deterministic, budget-respecting subset."""
    X, features, y, n_best = problem
    features2 = clone_features(features)

    first = RegressionSelector(features, n_best).fit(X, y)
    second = RegressionSelector(features2, n_best).fit(X, y)

    selected = list(first.selected_features)
    quanti, quali = _split_counts(selected)
    assert quanti <= n_best and quali <= n_best

    assert [f.version for f in first.selected_features] == [f.version for f in second.selected_features]
