"""Property-based test for the OrdinalCarver's ``target_scale="ridit"`` default.

Source: ``carvers/ordinal_carver.py`` + ``combinations/ordinal/ordinal_target_rates.py``.
The ridit scale promises that only the *order* of the target levels matters: the
whole carving pipeline (modality pre-sort, DP search, viability vetoes) must be
invariant under any strictly increasing re-encoding of the integer levels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from AutoCarver import OrdinalCarver
from AutoCarver.features import Features

SETTINGS = settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@st.composite
def counts_and_reencoding(draw) -> tuple[np.ndarray, np.ndarray]:
    """A (modalities × levels) count table (every cell ≥ 1, so levels/modalities are
    all observed) plus a random strictly increasing integer re-encoding of the levels."""
    n_mod = draw(st.integers(min_value=3, max_value=5))
    n_lev = draw(st.integers(min_value=3, max_value=4))
    cells = draw(st.lists(st.integers(min_value=1, max_value=6), min_size=n_mod * n_lev, max_size=n_mod * n_lev))
    increments = draw(st.lists(st.integers(min_value=1, max_value=9), min_size=n_lev, max_size=n_lev))
    return np.array(cells).reshape(n_mod, n_lev), np.cumsum(increments)


def _expand(table: np.ndarray, levels: list[int]) -> tuple[pd.DataFrame, pd.Series]:
    """Expands a count table to long-form ``(X, y)`` observations."""
    modalities, y = [], []
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            modalities += [f"m{i}"] * int(table[i, j])
            y += [levels[j]] * int(table[i, j])
    return pd.DataFrame({"cat": modalities}), pd.Series(y, name="target")


def _fit_content(X: pd.DataFrame, y: pd.Series) -> dict | None:
    """Fitted grouping of the single feature (``None`` when viability dropped it)."""
    carver = OrdinalCarver(min_freq=0.05, max_n_mod=3, features=Features(categoricals=["cat"]))
    carver.fit(X, y)
    if "cat" in carver.features:
        return {str(key): list(values) for key, values in carver.features("cat").content.items()}
    return None


@given(counts_and_reencoding())
@SETTINGS
def test_ridit_carving_invariant_under_monotone_reencoding(params):
    """The default carving is identical whether y is encoded 1..K or through any
    random strictly increasing integer map of those levels (including the
    feature being dropped on both or neither)."""
    table, new_levels = params
    n_lev = table.shape[1]

    X1, y1 = _expand(table, levels=list(range(1, n_lev + 1)))
    X2, y2 = _expand(table, levels=[int(level) for level in new_levels])

    assert _fit_content(X1, y1) == _fit_content(X2, y2)
