"""Set of tests for the correspondence_analysis module."""

import numpy as np
import pandas as pd
import pytest

from AutoCarver.stats.correspondence_analysis import CAAxis, ca_row_scores, fit_ca_axis


def _monotone_xtab() -> pd.DataFrame:
    """4 modalities with a clear monotone-ish signal against 3 classes."""
    return pd.DataFrame(
        {1: [80, 40, 10, 5], 2: [10, 40, 40, 10], 3: [5, 10, 40, 80]},
        index=["A", "B", "C", "D"],
    )


def test_fit_ca_axis_not_degenerate_on_structured_table():
    axis = fit_ca_axis(_monotone_xtab())
    assert isinstance(axis, CAAxis)
    assert not axis.degenerate


def test_ca_row_scores_orders_modalities_by_signal():
    xtab = _monotone_xtab()
    axis = fit_ca_axis(xtab)
    scores = ca_row_scores(xtab, axis).sort_values()
    assert list(scores.index) in (["D", "C", "B", "A"], ["A", "B", "C", "D"])


def test_determinism_repeated_fit():
    """Fitting twice on the same table gives identical scores."""
    xtab = _monotone_xtab()
    axis1 = fit_ca_axis(xtab)
    axis2 = fit_ca_axis(xtab)
    scores1 = ca_row_scores(xtab, axis1)
    scores2 = ca_row_scores(xtab, axis2)
    pd.testing.assert_series_equal(scores1, scores2)


def test_row_permutation_invariance():
    """Permuting the input rows must not change the score attributed to each label."""
    xtab = _monotone_xtab()
    permuted = xtab.loc[["D", "C", "B", "A"]]

    axis = fit_ca_axis(xtab)
    axis_permuted = fit_ca_axis(permuted)

    scores = ca_row_scores(xtab, axis)
    scores_permuted = ca_row_scores(permuted, axis_permuted)

    pd.testing.assert_series_equal(scores.sort_index(), scores_permuted.sort_index())


def test_label_independence():
    """Renaming the modality labels must not change the relative ordering."""
    xtab = _monotone_xtab()
    renamed = xtab.copy()
    renamed.index = ["w", "x", "y", "z"]

    axis = fit_ca_axis(xtab)
    axis_renamed = fit_ca_axis(renamed)

    order = list(ca_row_scores(xtab, axis).sort_values().index)
    order_renamed = list(ca_row_scores(renamed, axis_renamed).sort_values().index)

    rename_map = dict(zip(["A", "B", "C", "D"], ["w", "x", "y", "z"]))
    assert order_renamed == [rename_map[label] for label in order]


@pytest.mark.parametrize(
    "xtab",
    [
        pd.DataFrame({1: [10, 10, 10], 2: [10, 10, 10], 3: [10, 10, 10]}, index=["A", "B", "C"]),
        pd.DataFrame({1: [10, 1], 2: [1, 10], 3: [5, 5]}, index=["A", "B"]),
        pd.DataFrame({1: [10], 2: [1], 3: [5]}, index=["A"]),
    ],
)
def test_degenerate_cases_fall_back_deterministically(xtab: pd.DataFrame):
    """Uniform-profile tables and <=2-row tables are flagged degenerate and
    fall back to a deterministic (frequency-descending) order."""
    axis = fit_ca_axis(xtab)
    assert axis.degenerate
    scores = ca_row_scores(xtab, axis).sort_values()
    row_totals = xtab.sum(axis=1)
    # ascending score order == descending frequency order
    assert list(scores.index) == list(row_totals.sort_values(ascending=False).index)


def test_ca_row_scores_projects_a_different_table_onto_the_fixed_axis():
    """A "dev" table (different counts, same columns/labels) is projected using
    the train-fit axis — the scores need not match train's, but must be
    well-defined and deterministic."""
    train = _monotone_xtab()
    axis = fit_ca_axis(train)

    dev = pd.DataFrame(
        {1: [70, 35, 12, 8], 2: [15, 45, 38, 12], 3: [8, 12, 35, 70]},
        index=["A", "B", "C", "D"],
    )
    dev_scores = ca_row_scores(dev, axis)
    assert not dev_scores.isna().any()
    # same monotone signal on dev -> same relative order as train
    assert list(dev_scores.sort_values().index) in (["D", "C", "B", "A"], ["A", "B", "C", "D"])


def test_ca_row_scores_handles_zero_total_row():
    """A row with zero total across all columns must not raise / NaN-propagate."""
    xtab = _monotone_xtab()
    axis = fit_ca_axis(xtab)
    dev = pd.DataFrame({1: [0, 5], 2: [0, 5], 3: [0, 5]}, index=["empty", "other"])
    scores = ca_row_scores(dev, axis)
    assert np.isfinite(scores["empty"])
