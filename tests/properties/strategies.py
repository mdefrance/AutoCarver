"""Reusable hypothesis strategies for AutoCarver property-based tests.

These build the kinds of inputs the pure functions under test expect:
finite floats, lists of distinct hashable values, valid ``GroupedList`` content
dicts (disjoint + self-membership), and contiguous partitions of an ordered list
(the "combination" shape used throughout ``combinations.utils``).

Phase 2 adds *estimator-input* strategies (:func:`categorical_column`,
:func:`numerical_column`, the target strategies and :func:`dataframe_and_features`)
that build small, valid ``(X, Features, y)`` tuples for the carvers, discretizers
and selectors.
"""

import numpy as np
import pandas as pd
from hypothesis import strategies as st

from AutoCarver.features import Features

# Finite floats with bounded magnitude — safe for quantile/Wilson math
# (no NaN, no inf, nothing that overflows scientific formatting).
safe_floats = st.floats(
    min_value=-1e6,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
    width=64,
)

# Short, hashable string labels used as feature modalities.
labels = st.text(alphabet="abcdefghijABCDEFGHIJ0123456789", min_size=1, max_size=4)

# Well-conditioned continuous floats: bounded magnitude, no subnormals, rounded to
# 6 decimals. Avoids the catastrophic-cancellation regime (e.g. ~1e-298 values that
# make a column numerically near-constant) where scipy and pandas correlation kernels
# legitimately disagree — so vectorized<->scalar parity stays meaningful.
continuous_floats = st.floats(
    min_value=-1e3,
    max_value=1e3,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
    width=64,
).map(lambda v: round(v, 6))


@st.composite
def distinct_labels(draw, min_size: int = 1, max_size: int = 8) -> list[str]:
    """A list of unique string labels (order preserved)."""
    return draw(st.lists(labels, min_size=min_size, max_size=max_size, unique=True))


@st.composite
def ordered_values(draw, min_size: int = 1, max_size: int = 8) -> list[int]:
    """A list of unique ints in a stable order (proxy for an ordinal order)."""
    return draw(st.lists(st.integers(min_value=0, max_value=50), min_size=min_size, max_size=max_size, unique=True))


@st.composite
def contiguous_partition(draw, min_groups: int = 1) -> tuple[list, list[list]]:
    """Returns ``(order, combination)`` where combination is a partition of
    ``order`` into contiguous, non-empty groups (the shape produced by
    ``consecutive_combinations`` / consumed by ``order_apply_combination``)."""
    order = draw(distinct_labels(min_size=max(min_groups, 1), max_size=8))
    n = len(order)
    # choose cut points to split [0, n) into contiguous chunks
    max_cuts = n - 1
    n_cuts = draw(st.integers(min_value=max(min_groups - 1, 0), max_value=max_cuts)) if max_cuts >= 0 else 0
    cuts = (
        sorted(draw(st.sets(st.integers(min_value=1, max_value=n - 1), min_size=n_cuts, max_size=n_cuts)))
        if n > 1
        else []
    )
    bounds = [0, *cuts, n]
    combination = [order[bounds[i] : bounds[i + 1]] for i in range(len(bounds) - 1)]
    return order, combination


@st.composite
def grouped_list_dict(draw, min_size: int = 1, max_size: int = 6) -> dict[str, list[str]]:
    """A ``{leader: [members...]}`` dict satisfying GroupedList invariants:
    disjoint values across groups and each leader present in its own values."""
    _, combination = draw(contiguous_partition(min_groups=min_size))
    combination = combination[:max_size]
    # leader is the first member; ensure self-membership (it already is).
    return {group[0]: list(group) for group in combination}


# ---------------------------------------------------------------------------
# Estimator-input strategies (carvers / discretizers / selectors)
# ---------------------------------------------------------------------------


@st.composite
def categorical_column(draw, nrows: int, *, cardinality: int | None = None, nan_rate: float = 0.0) -> pd.Series:
    """An object ``Series`` of ``nrows`` rows with a controllable modality count.

    ``cardinality`` (drawn 1..5 when not given) sets the number of distinct
    modalities; when ``nan_rate > 0`` some cells may be ``NaN`` (presence, not an
    exact rate). The column may degenerate to a single repeated modality.
    """
    if cardinality is None:
        cardinality = draw(st.integers(min_value=1, max_value=5))
    pool = [chr(ord("a") + i) for i in range(cardinality)]
    choices = [*pool, None] if nan_rate > 0 else list(pool)
    data = draw(st.lists(st.sampled_from(choices), min_size=nrows, max_size=nrows))
    return pd.Series(data, dtype=object)


@st.composite
def numerical_column(
    draw, nrows: int, *, ties: bool = True, nan_rate: float = 0.0, constant: bool = False
) -> pd.Series:
    """A float ``Series`` of ``nrows`` rows.

    ``constant`` returns an all-equal column (degenerate); ``ties`` draws from a
    small integer-valued pool (heavy ties), otherwise from a continuous range;
    ``nan_rate > 0`` may introduce ``NaN`` cells.
    """
    if constant:
        value = draw(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False, width=64))
        return pd.Series([value] * nrows, dtype=float)

    if ties:
        pool = [float(v) for v in range(draw(st.integers(min_value=2, max_value=8)))]
        choices = [*pool, np.nan] if nan_rate > 0 else pool
        data = draw(st.lists(st.sampled_from(choices), min_size=nrows, max_size=nrows))
    else:
        element = continuous_floats
        if nan_rate > 0:
            element = st.one_of(element, st.just(np.nan))
        data = draw(st.lists(element, min_size=nrows, max_size=nrows))
    return pd.Series(data, dtype=float)


@st.composite
def binary_target(draw, nrows: int) -> pd.Series:
    """An int ``Series`` of 0/1 guaranteed to contain both classes (``nrows >= 2``)."""
    data = draw(st.lists(st.integers(min_value=0, max_value=1), min_size=nrows, max_size=nrows))
    data[0], data[1] = 0, 1
    return pd.Series(data)


@st.composite
def continuous_target(draw, nrows: int) -> pd.Series:
    """A float ``Series`` guaranteed to have more than two distinct values (``nrows >= 3``)."""
    element = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False, width=64)
    data = draw(st.lists(element, min_size=nrows, max_size=nrows))
    data[0], data[1], data[2] = 0.0, 1.0, 2.0
    return pd.Series(data)


@st.composite
def multiclass_target(draw, nrows: int, *, n_classes: int | None = None) -> pd.Series:
    """An int ``Series`` guaranteed to contain at least ``n_classes`` (>=3) classes."""
    if n_classes is None:
        n_classes = draw(st.integers(min_value=3, max_value=4))
    data = draw(st.lists(st.integers(min_value=0, max_value=n_classes - 1), min_size=nrows, max_size=nrows))
    for cls in range(n_classes):
        data[cls] = cls
    return pd.Series(data)


@st.composite
def ordinal_target(draw, nrows: int, *, n_levels: int | None = None) -> pd.Series:
    """An int ``Series`` of integer-encoded ordered levels ``1..K`` (``K >= 3``).

    Every level is guaranteed present (``nrows >= n_levels``), so the carver's
    ``> 2`` ordered-level guard is satisfied.
    """
    if n_levels is None:
        n_levels = draw(st.integers(min_value=3, max_value=5))
    data = draw(st.lists(st.integers(min_value=1, max_value=n_levels), min_size=nrows, max_size=nrows))
    for level in range(n_levels):
        data[level] = level + 1
    return pd.Series(data)


def _target(draw, target_kind: str, nrows: int) -> pd.Series:
    """Draws a target ``Series`` matching ``target_kind``."""
    if target_kind == "binary":
        return draw(binary_target(nrows))
    if target_kind == "continuous":
        return draw(continuous_target(nrows))
    if target_kind == "multiclass":
        return draw(multiclass_target(nrows))
    if target_kind == "ordinal":
        return draw(ordinal_target(nrows))
    raise ValueError(f"unknown target_kind {target_kind!r}")


@st.composite
def dataframe_and_features(
    draw,
    target_kind: str = "binary",
    *,
    nrows: tuple[int, int] = (30, 120),
    with_nan: bool = False,
) -> tuple[pd.DataFrame, Features, pd.Series]:
    """Builds a small ``(X, Features, y)`` tuple.

    ``X`` mixes 1-3 categorical / ordinal / numerical columns; ``features`` is a
    matching :class:`Features`; ``y`` matches ``target_kind`` ("binary",
    "continuous", "multiclass" or "ordinal"). With ``with_nan`` the
    qualitative/categorical columns may carry ``NaN``.
    """
    n = draw(st.integers(min_value=nrows[0], max_value=nrows[1]))
    nan_rate = 0.2 if with_nan else 0.0

    n_cat = draw(st.integers(min_value=0, max_value=2))
    n_ord = draw(st.integers(min_value=0, max_value=2))
    n_num = draw(st.integers(min_value=0, max_value=2))
    if n_cat + n_ord + n_num == 0:
        n_num = 1

    columns: dict[str, pd.Series] = {}
    categoricals: list[str] = []
    ordinals: dict[str, list[str]] = {}
    numericals: list[str] = []

    for i in range(n_cat):
        name = f"cat{i}"
        categoricals.append(name)
        columns[name] = draw(categorical_column(n, cardinality=draw(st.integers(2, 4)), nan_rate=nan_rate))

    for i in range(n_ord):
        name = f"ord{i}"
        order = [f"o{j}" for j in range(draw(st.integers(min_value=2, max_value=4)))]
        ordinals[name] = order
        ord_choices = [*order, None] if with_nan else order
        data = draw(st.lists(st.sampled_from(ord_choices), min_size=n, max_size=n))
        columns[name] = pd.Series(data, dtype=object)

    for i in range(n_num):
        name = f"num{i}"
        numericals.append(name)
        columns[name] = draw(numerical_column(n, ties=draw(st.booleans()), nan_rate=nan_rate))

    X = pd.DataFrame(columns)
    features = Features(
        categoricals=categoricals or None,
        ordinals=ordinals or None,
        numericals=numericals or None,
    )
    y = _target(draw, target_kind, n)
    return X, features, y


def clone_features(features: Features) -> Features:
    """Deep-copies an (unfitted) :class:`Features` via its JSON round-trip.

    Used by determinism properties that need two equivalent, independent feature
    sets (estimators mutate their features in place during fit).
    """
    return Features.load(features.to_json())
