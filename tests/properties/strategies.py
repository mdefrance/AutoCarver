"""Reusable hypothesis strategies for AutoCarver property-based tests.

These build the kinds of inputs the pure functions under test expect:
finite floats, lists of distinct hashable values, valid ``GroupedList`` content
dicts (disjoint + self-membership), and contiguous partitions of an ordered list
(the "combination" shape used throughout ``combinations.utils``).
"""

from hypothesis import strategies as st

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
