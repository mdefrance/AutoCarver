"""End-to-end metamorphic properties for the :class:`Discretizer` pipeline.

Source: ``discretizers/discretizer.py`` (+ ``base_discretizer.py``), driven by
the ``dataframe_and_features`` strategy. Checks row/index/column preservation,
label membership of outputs, ordinal-encoding codes, copy non-mutation,
``fit_transform == fit().transform()``, determinism, the fit/transform guards
and NaN handling under both dropna modes.

Two carver-independent realities shape these tests:

* The discretizer legitimately *rejects* degenerate features (a single dominant
  modality, or none frequent enough) with a ``ValueError``. That is correct
  behaviour, not a metamorphic violation, so those inputs are filtered out via
  :func:`hypothesis.reject` (see :func:`_reject_degenerate`).
* The plain ``Discretizer`` does not propagate ``config.dropna`` onto its
  features (the carvers own that), so the dropna properties set
  ``features.dropna`` explicitly after fit — mirroring the existing tests.
"""

import pandas as pd
import pytest
from hypothesis import HealthCheck, given, reject, settings
from hypothesis import strategies as st
from pandas.api.types import is_numeric_dtype
from sklearn.exceptions import NotFittedError
from strategies import binary_target, categorical_column, clone_features, dataframe_and_features

from AutoCarver.discretizers.discretizer import Discretizer
from AutoCarver.discretizers.utils.base_discretizer import ProcessingConfig
from AutoCarver.features import Features

problem = dataframe_and_features("binary", with_nan=True)
SETTINGS = settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)


def _reject_degenerate(error: ValueError) -> None:
    """Filters out inputs the discretizer legitimately rejects (re-raises otherwise)."""
    if "frequent" in str(error):  # "too frequent modality or no frequent enough modalities"
        reject()
    raise error


def _fit(discretizer, X, y):
    """fit, filtering out degenerate-feature inputs."""
    try:
        return discretizer.fit(X, y)
    except ValueError as error:
        _reject_degenerate(error)


def _fit_transform(discretizer, X, y):
    """fit_transform, filtering out degenerate-feature inputs."""
    try:
        return discretizer.fit_transform(X, y)
    except ValueError as error:
        _reject_degenerate(error)


@st.composite
def qualitative_problem(draw):
    """A qualitative-only ``(X, Features, y)`` (no quantitative columns).

    Used for the ordinal-encoding property: the quantitative ordinal-encoding
    path has a known crash captured by
    :func:`test_ordinal_encoding_quantitative_indexerror`.
    """
    n = draw(st.integers(min_value=30, max_value=120))
    n_cat = draw(st.integers(min_value=0, max_value=2))
    n_ord = draw(st.integers(min_value=0, max_value=2))
    if n_cat + n_ord == 0:
        n_cat = 1
    columns, categoricals, ordinals = {}, [], {}
    for i in range(n_cat):
        categoricals.append(f"cat{i}")
        columns[f"cat{i}"] = draw(categorical_column(n, cardinality=draw(st.integers(2, 4)), nan_rate=0.2))
    for i in range(n_ord):
        order = [f"o{j}" for j in range(draw(st.integers(2, 4)))]
        ordinals[f"ord{i}"] = order
        columns[f"ord{i}"] = pd.Series(
            draw(st.lists(st.sampled_from([*order, None]), min_size=n, max_size=n)), dtype=object
        )
    X = pd.DataFrame(columns)
    features = Features(categoricals=categoricals or None, ordinals=ordinals or None)
    return X, features, draw(binary_target(n))


# --------------------------------------------------------------------------
# structure: rows / index / columns
# --------------------------------------------------------------------------
@given(problem)
@SETTINGS
def test_rows_index_columns_preserved(prob):
    """transform preserves row count, index and the column set."""
    X, features, y = prob
    out = _fit_transform(Discretizer(features, 0.2), X, y)

    assert len(out) == len(X)
    assert list(out.index) == list(X.index)
    assert set(out.columns) == set(X.columns)


# --------------------------------------------------------------------------
# label membership / ordinal encoding
# --------------------------------------------------------------------------
@given(problem)
@SETTINGS
def test_output_values_within_labels(prob):
    """With dropna on, every output value belongs to its feature's labels."""
    X, features, y = prob
    discretizer = Discretizer(features, 0.2)
    _fit(discretizer, X, y)
    features.dropna = True  # map NaNs to the nan label so no raw NaN remains
    out = discretizer.transform(X)

    for feature in features:
        labels = set(feature.labels)
        observed = set(out[feature.version].dropna().unique())
        assert observed.issubset(labels), feature.version


@given(qualitative_problem())
@SETTINGS
def test_ordinal_encoding_contiguous_codes(prob):
    """ordinal_encoding=True yields numeric columns whose labels are 0..k-1
    (qualitative features; the quantitative path is covered as an xfail below)."""
    X, features, y = prob
    discretizer = Discretizer(features, 0.2, config=ProcessingConfig(ordinal_encoding=True))
    _fit(discretizer, X, y)
    features.dropna = True
    out = discretizer.transform(X)

    for feature in features:
        labels = feature.labels
        # labels are a contiguous 0..k-1 integer range
        assert labels == list(range(len(labels)))
        assert is_numeric_dtype(out[feature.version])
        observed = set(out[feature.version].dropna().unique())
        assert observed.issubset(set(range(len(labels)))), feature.version


def test_ordinal_encoding_quantitative_indexerror():
    """A quantitative feature with rare modalities ordinal-encodes without crashing.

    Regression guard: QuantitativeDiscretizer runs an OrdinalDiscretizer merge after the
    feature is already ordinal-encoded; QuantitativeFeature._specific_update used to index
    self.values positionally with a shrinking GroupedList -> IndexError. The reverse map is
    now snapshotted once before the grouping loop.
    """
    col = [4.0] * 12 + [1.0] * 11 + [3.0] * 10 + [0.0] * 9 + [2.0] * 4 + [float("nan")] * 19
    X = pd.DataFrame({"num0": col})
    y = pd.Series([0, 1] * 32 + [0])
    Discretizer(Features(numericals=["num0"]), 0.2, config=ProcessingConfig(ordinal_encoding=True)).fit(X, y)


# --------------------------------------------------------------------------
# copy non-mutation
# --------------------------------------------------------------------------
@given(problem)
@SETTINGS
def test_copy_does_not_mutate_input(prob):
    """copy=True (default) leaves the input DataFrame untouched."""
    X, features, y = prob
    before = X.copy(deep=True)
    _fit_transform(Discretizer(features, 0.2), X, y)
    assert X.equals(before)


# --------------------------------------------------------------------------
# determinism / fit_transform equivalence
# --------------------------------------------------------------------------
@given(problem)
@SETTINGS
def test_fit_transform_equals_fit_then_transform(prob):
    """fit_transform(X, y) equals fit(X, y).transform(X) and is deterministic
    across two independent instances."""
    X, features, y = prob
    features2 = clone_features(features)

    out_ft = _fit_transform(Discretizer(features, 0.2), X, y)
    out_split = _fit(Discretizer(features2, 0.2), X, y).transform(X)

    assert out_ft.equals(out_split)


# --------------------------------------------------------------------------
# fit / transform guards
# --------------------------------------------------------------------------
@given(problem)
@SETTINGS
def test_transform_before_fit_raises(prob):
    """transform before fit raises NotFittedError."""
    X, features, _ = prob
    with pytest.raises(NotFittedError):
        Discretizer(features, 0.2).transform(X)


@given(problem)
@SETTINGS
def test_double_fit_raises(prob):
    """Re-fitting an already-fitted discretizer raises.

    The fitted-guard is a ``RuntimeError`` raised by ``BaseDiscretizer.fit``, but
    ``Discretizer.fit`` re-runs sub-discretization before reaching it, so on some
    inputs a different exception surfaces first — the invariant is that a second
    fit never silently succeeds.
    """
    X, features, y = prob
    discretizer = Discretizer(features, 0.2)
    _fit(discretizer, X, y)
    with pytest.raises((RuntimeError, ValueError)):
        discretizer.fit(X, y)


# --------------------------------------------------------------------------
# NaN handling
# --------------------------------------------------------------------------
@given(problem)
@SETTINGS
def test_nans_survive_when_dropna_false(prob):
    """With dropna left False, input NaNs survive as NaN per feature."""
    X, features, y = prob
    out = _fit_transform(Discretizer(features, 0.2), X, y)

    for feature in features:
        assert out[feature.version].isna().sum() == X[feature.name].isna().sum(), feature.version


@given(problem)
@SETTINGS
def test_nans_mapped_when_dropna_true(prob):
    """With dropna True, no raw NaN remains and the nan label is used."""
    X, features, y = prob
    discretizer = Discretizer(features, 0.2)
    _fit(discretizer, X, y)
    features.dropna = True
    out = discretizer.transform(X)

    for feature in features:
        assert out[feature.version].isna().sum() == 0, feature.version
        if X[feature.name].isna().any():
            nan_label = feature.label_per_value.get(feature.nan)
            assert (out[feature.version] == nan_label).any(), feature.version
