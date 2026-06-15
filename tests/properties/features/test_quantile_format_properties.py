"""Property-based tests for quantile label formatting.

The core contract: ``format_quantiles`` must produce one more label than it has
thresholds, each a valid interval, and crucially all labels must be **unique**
(downstream ``GroupedList(format_quantiles(...))`` dedupes silently, which
previously surfaced as ``KeyError: inf``).
"""

import re

from hypothesis import given
from hypothesis import strategies as st
from strategies import safe_floats

from AutoCarver.features.quantitatives.quantitative_feature import format_quantiles, min_decimals_to_differentiate

# sorted lists of distinct floats (quantile thresholds are sorted & unique)
sorted_distinct_floats = st.lists(safe_floats, min_size=0, max_size=10, unique=True).map(sorted)

INTERVAL_RE = re.compile(r"^\((-inf|[^,]+), ([^,]+|inf)[\]\)]$")


@given(sorted_distinct_floats, st.integers(min_value=0, max_value=5))
def test_min_decimals_respects_floor(numbers, min_decimals):
    """Result is never below the requested minimum."""
    assert min_decimals_to_differentiate(numbers, min_decimals) >= min_decimals


@given(sorted_distinct_floats)
def test_min_decimals_makes_strings_distinct(numbers):
    """At the returned precision, distinct numbers format to distinct strings.

    This is the whole point of the function — it guards label uniqueness.
    """
    decimals = min_decimals_to_differentiate(numbers, min_decimals=1)
    formatted = {f"{n:.{decimals}e}" for n in numbers}
    assert len(formatted) == len(numbers)


@given(sorted_distinct_floats)
def test_format_quantiles_count(numbers):
    """format_quantiles returns len(thresholds) + 1 labels."""
    labels = format_quantiles(numbers)
    assert len(labels) == len(numbers) + 1


@given(sorted_distinct_floats)
def test_format_quantiles_labels_unique(numbers):
    """All produced labels are unique (the KeyError:inf regression guard)."""
    labels = format_quantiles(numbers)
    assert len(set(labels)) == len(labels)


@given(sorted_distinct_floats)
def test_format_quantiles_bounds_and_shape(numbers):
    """First label is left-open at -inf, last is right-open at inf; each is a
    valid interval string."""
    labels = format_quantiles(numbers)
    assert labels[0].startswith("(-inf,")
    assert labels[-1].endswith("inf)")
    for label in labels:
        assert INTERVAL_RE.match(label), label


def test_format_quantiles_empty():
    """No thresholds → a single all-spanning interval."""
    assert format_quantiles([]) == ["(-inf, inf)"]
