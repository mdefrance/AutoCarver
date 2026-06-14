"""Set of tests for qualitative_discretizers module."""

import numpy as np
import pandas as pd
from pytest import raises

from AutoCarver.discretizers.qualitatives.ordinal_discretizer import (
    OrdinalDiscretizer,
    compute_stats,
    find_closest_modality,
    find_common_modalities,
    is_next_modality_closer,
    is_next_modality_closer_by_target_rate,
    update_stats,
)
from AutoCarver.discretizers.utils.base_discretizer import ProcessingConfig
from AutoCarver.features import Features, GroupedList, OrdinalFeature


def test_find_closest_modality_single_element():
    """Test find_closest_modality when there is only one element"""
    idx = 0
    frequencies = np.array([0.1])
    target_rates = np.array([0.1])
    min_freq = 0.15
    result = find_closest_modality(idx, frequencies, target_rates, min_freq)
    expected = 0
    assert result == expected


def test_find_closest_modality_lowest_ranked():
    """Test find_closest_modality when idx is the lowest ranked modality"""
    idx = 0
    frequencies = np.array([0.1, 0.2, 0.3, 0.4])
    target_rates = np.array([0.1, 0.2, 0.3, 0.4])
    min_freq = 0.15
    result = find_closest_modality(idx, frequencies, target_rates, min_freq)
    expected = 1
    assert result == expected


def test_find_closest_modality_highest_ranked():
    """Test find_closest_modality when idx is the highest ranked modality"""
    idx = 3
    frequencies = np.array([0.1, 0.2, 0.3, 0.4])
    target_rates = np.array([0.1, 0.2, 0.3, 0.4])
    min_freq = 0.15
    result = find_closest_modality(idx, frequencies, target_rates, min_freq)
    expected = 2
    assert result == expected


def test_find_closest_modality_middle_closer_previous():
    """Test find_closest_modality when idx is in the middle and next modality is closer"""
    idx = 1
    frequencies = np.array([0.1, 0.2, 0.1])
    target_rates = np.array([0.1, 0.2, 0.4])
    min_freq = 0.15
    result = find_closest_modality(idx, frequencies, target_rates, min_freq)
    expected = 0
    assert result == expected


def test_find_closest_modality_middle_closer_next():
    """Test find_closest_modality when idx is in the middle and previous modality is closer"""
    idx = 1
    frequencies = np.array([0.1, 0.2, 0.1])
    target_rates = np.array([0.4, 0.2, 0.1])
    min_freq = 0.15
    result = find_closest_modality(idx, frequencies, target_rates, min_freq)
    expected = 2
    assert result == expected


def test_is_next_modality_closer_by_target_rate_closer():
    """Test when the next modality is closer by target rate"""
    idx = 1
    target_rates = np.array([0.1, 0.3, 0.4])
    result = is_next_modality_closer_by_target_rate(idx, target_rates)
    assert result is True


def test_is_next_modality_closer_by_target_rate_not_closer():
    """Test when the next modality is not closer by target rate"""
    idx = 1
    target_rates = np.array([0.4, 0.3, 0.1])
    result = is_next_modality_closer_by_target_rate(idx, target_rates)
    assert result is False


def test_is_next_modality_closer_by_target_rate_equal_distance():
    """Test when the next and previous modalities are equally distant by target rate"""
    idx = 1
    target_rates = np.array([0.2, 0.3, 0.4])
    result = is_next_modality_closer_by_target_rate(idx, target_rates)
    assert result is False


def test_is_next_modality_closer_by_target_rate_negative_values():
    """Test when the target rates include negative values"""
    idx = 1
    target_rates = np.array([-0.1, -0.3, -0.4])
    result = is_next_modality_closer_by_target_rate(idx, target_rates)
    assert result is True


def test_is_next_modality_closer_by_target_rate_zero_values():
    """Test when the target rates include zero values"""
    idx = 1
    target_rates = np.array([0.1, 0.0, -0.1])
    result = is_next_modality_closer_by_target_rate(idx, target_rates)
    assert result is False


def test_is_next_modality_closer_no_frequency():
    """Test when the current modality has no frequency"""

    # bigger frequency for the previous modality
    idx = 1
    frequencies = np.array([0.2, 0.0, 0.3])
    target_rates = np.array([0.1, 0.2, 0.3])
    min_freq = 0.15
    result = is_next_modality_closer(idx, frequencies, target_rates, min_freq)
    assert not result

    # bigger frequency for the next modality
    frequencies = np.array([0.3, 0.0, 0.2])
    result = is_next_modality_closer(idx, frequencies, target_rates, min_freq)
    assert result


def test_is_next_modality_closer_underrepresented():
    """Test when the next modality is underrepresented and the previous is not"""

    # next modality is underrepresented
    idx = 1
    frequencies = np.array([0.3, 0.2, 0.1])
    target_rates = np.array([0.1, 0.2, 0.3])
    min_freq = 0.15
    result = is_next_modality_closer(idx, frequencies, target_rates, min_freq)
    assert result is True

    # previous modality is underrepresented
    frequencies = np.array([0.1, 0.2, 0.3])
    result = is_next_modality_closer(idx, frequencies, target_rates, min_freq)
    assert result is False


def test_is_next_modality_closer_both_underrepresented():
    """Test when both the next and previous modalities are underrepresented"""

    # next modality is closer by target rate
    idx = 1
    frequencies = np.array([0.1, 0.2, 0.1])
    target_rates = np.array([0.05, 0.2, 0.3])
    min_freq = 0.15
    result = is_next_modality_closer(idx, frequencies, target_rates, min_freq)
    assert result is True

    # previous modality is closer by target rate
    target_rates = np.array([0.1, 0.2, 0.35])
    result = is_next_modality_closer(idx, frequencies, target_rates, min_freq)
    assert result is False


def test_is_next_modality_closer_both_overrepresented():
    """Test when both the next and previous modalities are overrepresented"""

    # next modality is closer by target rate
    idx = 1
    frequencies = np.array([0.4, 0.2, 0.5])
    target_rates = np.array([0.05, 0.2, 0.3])
    min_freq = 0.15
    result = is_next_modality_closer(idx, frequencies, target_rates, min_freq)
    assert result is True

    # previous modality is closer by target rate
    target_rates = np.array([0.1, 0.2, 0.35])
    result = is_next_modality_closer(idx, frequencies, target_rates, min_freq)
    assert result is False


def test_compute_stats_basic():
    """Test compute_stats with basic input"""
    df_feature = pd.Series(["A", "B", "A", "C", "B", "A"])
    y = pd.Series([1, 0, 1, 0, 1, 0])
    labels = GroupedList(["A", "B", "C"])
    stats, len_df = compute_stats(df_feature, y, labels)
    expected_stats = np.vstack((np.array([3, 2, 1]), np.array([2, 1, 0])))  # frequencies  # target rates
    expected_len_df = 6
    assert (stats == expected_stats).all()
    assert len_df == expected_len_df


def test_compute_stats_with_nans():
    """Test compute_stats with np.nan values in df_feature"""
    df_feature = pd.Series(["A", "B", np.nan, "C", "B", "A"])
    y = pd.Series([1, 0, 1, 0, 1, 0])
    labels = GroupedList(["A", "B", "C"])
    stats, len_df = compute_stats(df_feature, y, labels)
    expected_stats = np.vstack((np.array([2, 2, 1]), np.array([1, 1, 0])))  # frequencies  # target rates
    expected_len_df = 6  # len_df includes np.nan values
    assert (stats == expected_stats).all()
    assert len_df == expected_len_df


def test_compute_stats_with_missing_value():
    """Test compute_stats with missing value in df_feature"""
    df_feature = pd.Series(["A", "B", "A", np.nan, "B", "A"])
    y = pd.Series([1, 0, 1, 0, 1, 0])
    labels = GroupedList(["A", "B", "C"])
    stats, len_df = compute_stats(df_feature, y, labels)
    expected_stats = np.vstack((np.array([3, 2, 0]), np.array([2, 1, np.nan])))  # frequencies  # target rates
    expected_len_df = 6  # len_df includes np.nan values

    # Check frequencies
    assert (stats[0] == expected_stats[0]).all()
    # Check target rates with np.isnan
    assert (np.isnan(stats[1]) == np.isnan(expected_stats[1])).all()
    assert (stats[1][~np.isnan(stats[1])] == expected_stats[1][~np.isnan(expected_stats[1])]).all()
    assert len_df == expected_len_df


def test_compute_stats_all_nans():
    """Test compute_stats with all np.nan values in df_feature"""
    df_feature = pd.Series([None, None, None])
    y = pd.Series([1, 0, 1])
    labels = GroupedList(["A", "B", "C"])
    stats, len_df = compute_stats(df_feature, y, labels)
    expected_stats = np.vstack((np.array([0, 0, 0]), np.array([np.nan, np.nan, np.nan])))  # frequencies  # target rates
    expected_len_df = 3  # len_df includes np.nan values

    # Check frequencies
    assert (stats[0] == expected_stats[0]).all()
    # Check target rates with np.isnan
    assert np.isnan(stats[1]).all() and np.isnan(expected_stats[1]).all()
    assert len_df == expected_len_df


def test_compute_stats_empty_input():
    """Test compute_stats with empty input"""
    df_feature = pd.Series([])
    y = pd.Series([])
    labels = GroupedList([])
    stats, len_df = compute_stats(df_feature, y, labels)
    expected_stats = np.vstack((np.array([]), np.array([])))  # frequencies  # target rates
    expected_len_df = 0  # empty input
    assert (stats == expected_stats).all()
    assert len_df == expected_len_df


def test_compute_stats_single_modality():
    """Test compute_stats with a single modality"""
    df_feature = pd.Series(["A", "A", "A"])
    y = pd.Series([1, 0, 1])
    labels = GroupedList(["A"])
    stats, len_df = compute_stats(df_feature, y, labels)
    expected_stats = np.vstack((np.array([3]), np.array([2])))  # frequencies  # target rates
    expected_len_df = 3
    assert (stats == expected_stats).all()
    assert len_df == expected_len_df


def test_update_stats_basic():
    """Test update_stats with basic input"""
    df_feature = pd.Series(["A", "B", "A", "C", "B", "A"])
    y = pd.Series([1, 0, 1, 0, 1, 0])
    labels = GroupedList(["A", "B", "C"])
    stats, _ = compute_stats(df_feature, y, labels)
    updated_stats = update_stats(stats, discarded_idx=2, kept_idx=0)
    expected_stats = np.vstack(
        (np.array([4, 2]), np.array([2, 1]))  # updated frequencies  # updated target rates
    )
    assert (updated_stats == expected_stats).all()


def test_update_stats_with_nans():
    """Test update_stats with np.nan values in df_feature"""
    df_feature = pd.Series(["A", "B", None, "C", "B", "A"])
    y = pd.Series([1, 0, 1, 0, 1, 0])
    labels = GroupedList(["A", "B", "C"])
    stats, _ = compute_stats(df_feature, y, labels)
    updated_stats = update_stats(stats, discarded_idx=2, kept_idx=0)
    expected_stats = np.vstack(
        (np.array([3, 2]), np.array([1, 1]))  # updated frequencies  # updated target rates
    )
    assert (updated_stats == expected_stats).all()


def test_update_stats_with_missing_value():
    """Test update_stats with missing value in df_feature"""
    df_feature = pd.Series(["A", "B", "A", np.nan, "B", "A"])
    y = pd.Series([1, 0, 1, 0, 1, 0])
    labels = GroupedList(["A", "B", "C"])
    stats, _ = compute_stats(df_feature, y, labels)
    updated_stats = update_stats(stats, discarded_idx=2, kept_idx=0)
    expected_stats = np.vstack(
        (np.array([3, 2]), np.array([2, 1]))  # updated frequencies  # updated target rates
    )
    assert (updated_stats == expected_stats).all()


def test_update_stats_all_nans():
    """Test update_stats with all np.nan values in df_feature"""
    df_feature = pd.Series([None, None, None])
    y = pd.Series([1, 0, 1])
    labels = GroupedList(["A", "B", "C"])
    stats, _ = compute_stats(df_feature, y, labels)
    updated_stats = update_stats(stats, discarded_idx=2, kept_idx=0)
    expected_stats = np.vstack(
        (np.array([0, 0]), np.array([np.nan, np.nan]))  # updated frequencies  # updated target rates
    )
    assert (updated_stats[0] == expected_stats[0]).all()
    assert (np.isnan(updated_stats[1]) == np.isnan(expected_stats[1])).all()


def test_update_stats_empty_input():
    """Test update_stats with empty input"""
    df_feature = pd.Series([])
    y = pd.Series([])
    labels = GroupedList([])
    stats, _ = compute_stats(df_feature, y, labels)
    with raises(IndexError):
        update_stats(stats, discarded_idx=0, kept_idx=0)


def test_update_stats_single_modality():
    """Test update_stats with a single modality"""
    df_feature = pd.Series(["A", "A", "A"])
    y = pd.Series([1, 0, 1])
    labels = GroupedList(["A"])
    stats, _ = compute_stats(df_feature, y, labels)
    with raises(IndexError):
        update_stats(stats, discarded_idx=0, kept_idx=1)


def test_find_common_modalities_all_overrepresented():
    """All modalities have observed proportion >> min_freq → nothing merged."""
    df_feature = pd.Series(["A", "B", "A", "C", "B", "A"])
    y = pd.Series([1, 0, 1, 0, 1, 0])
    min_freq = 0 / 6
    labels = ["A", "B", "C"]
    result = find_common_modalities(df_feature, y, min_freq, labels, alpha=0.05)
    expected = GroupedList(labels)
    assert result == expected
    assert result.content == expected.content


def test_find_common_modalities_all_significantly_below():
    """All modalities significantly below min_freq (Wilson upper < min_freq) → all merged."""
    # n=400, three modalities of count ~10 each → Wilson upper(10, 400, 0.05) ≈ 0.046 < 0.5.
    df_feature = pd.Series(["A"] * 10 + ["B"] * 10 + ["C"] * 10 + ["D"] * 370)
    y = pd.Series([0] * 400)
    labels = ["A", "B", "C", "D"]
    result = find_common_modalities(df_feature, y, min_freq=0.5, labels=labels, alpha=0.05)
    # A/B/C all get merged into D (the only one above the CI floor); merge order is
    # driven by argmin walk, so the listed order under D mirrors the discard sequence.
    assert set(result.content["D"]) == {"A", "B", "C", "D"}
    assert list(result.content.keys()) == ["D"]


def test_find_common_modalities_borderline_modality_survives_on_small_n():
    """On small samples, the Wilson CI is wide — a modality with proportion just below
    min_freq isn't significantly below it, so the OrdinalDiscretizer doesn't merge."""

    # n=6, C count=1 (~17%) vs min_freq=3/6=50%; Wilson upper(1,6,0.05) ≈ 0.56 > 0.5 → keep.
    df_feature = pd.Series(["A", "B", np.nan, "C", "B", "A"])
    y = pd.Series([1, 0, 1, 0, 1, 0])
    labels = ["A", "B", "C"]
    result = find_common_modalities(df_feature, y, min_freq=3 / 6, labels=labels, alpha=0.05)
    assert result == GroupedList(labels)


def test_find_common_modalities_with_equal_min_freq():
    """A modality at exactly min_freq passes (Wilson upper ≥ min_freq)."""
    df_feature = pd.Series(["A", "B", "A", "C", "B", "A"])
    y = pd.Series([1, 0, 1, 0, 1, 0])
    min_freq = 1 / 6
    labels = ["A", "B", "C"]
    result = find_common_modalities(df_feature, y, min_freq, labels, alpha=0.05)
    expected = GroupedList(labels)
    assert result == expected
    assert result.content == expected.content


def test_find_common_modalities_increased_freq_large_n():
    """At large n the CI is tight: a 17%-modality IS significantly below 50% and gets merged."""

    # n=600, C count=100 (~17%); Wilson upper(100, 600, 0.05) ≈ 0.198 < 0.5 → merge.
    df_feature = pd.Series(["A"] * 300 + ["B"] * 200 + ["C"] * 100)
    y = pd.Series([0] * 600)
    labels = ["A", "B", "C"]
    result = find_common_modalities(df_feature, y, min_freq=3 / 6, labels=labels, alpha=0.05)
    expected = GroupedList(labels)
    expected.group("C", "B")
    assert result == expected


def test_find_common_modalities_with_missing_value_small_n():
    """Missing label (count=0) on small n: Wilson upper(0, 6) ≈ 0.39 > min_freq=2/6 → not merged.

    The CI captures the uncertainty around 0 counts on tiny samples.
    """
    df_feature = pd.Series(["A", "B", "A", np.nan, "B", "A"])
    y = pd.Series([1, 0, 1, 0, 1, 0])
    labels = ["A", "B", "C"]
    result = find_common_modalities(df_feature, y, min_freq=2 / 6, labels=labels, alpha=0.05)
    assert result == GroupedList(labels)


def test_find_common_modalities_empty_input():
    """Test find_common_modalities with empty input"""
    df_feature = pd.Series([])
    y = pd.Series([])
    min_freq = 0.2
    labels = []
    result = find_common_modalities(df_feature, y, min_freq, labels, alpha=0.05)
    expected = GroupedList([])
    assert result == expected
    assert result.content == expected.content


def test_find_common_modalities_single_modality():
    """Test find_common_modalities with a single modality"""
    df_feature = pd.Series(["A", "A", "A"])
    y = pd.Series([1, 0, 1])
    min_freq = 0.2
    labels = ["A"]
    result = find_common_modalities(df_feature, y, min_freq, labels, alpha=0.05)
    expected = GroupedList(labels)
    assert result == expected
    assert result.content == expected.content


def test_find_common_modalities_unexpected_value():
    """Test find_common_modalities with an unexpected value in df_feature"""
    df_feature = pd.Series(["A", "B", "A", "C", "B", "A"])
    y = pd.Series([1, 0, 1, 0, 1, 0])
    min_freq = 2 / 6
    labels = ["A", "B"]
    result = find_common_modalities(df_feature, y, min_freq, labels, alpha=0.05)
    expected = GroupedList(labels)
    assert result == expected


def test_find_common_modalities_small_n_does_not_overmerge():
    """A modality 1 row short of min_freq on a small sample is NOT significantly below it —
    the Wilson CI is wide enough that the proportion is statistically consistent with
    min_freq. Merging on this evidence alone is too aggressive."""

    # n=8000, min_freq=0.025 → target count 200. count=198 (2 short): Wilson upper ≈ 0.0284 > 0.025 → keep.
    n = 8000
    df_feature = pd.Series(["A"] * (n - 198) + ["B"] * 198)
    y = pd.Series([0] * n)
    result = find_common_modalities(df_feature, y, min_freq=0.025, labels=["A", "B"], alpha=0.05)
    assert result == GroupedList(["A", "B"]), "modality within CI of min_freq must survive"


def test_find_common_modalities_significantly_below_at_large_n():
    """At large n the CI is tight — a modality clearly below min_freq is merged."""

    # n=8000, min_freq=0.025; count=100 (~0.0125): Wilson upper ≈ 0.015 < 0.025 → merge.
    n = 8000
    df_feature = pd.Series(["A"] * (n - 100) + ["B"] * 100)
    y = pd.Series([0] * n)
    result = find_common_modalities(df_feature, y, min_freq=0.025, labels=["A", "B"], alpha=0.05)
    expected = GroupedList(["A", "B"])
    expected.group("B", "A")
    assert result == expected


def test_ordinal_discretizer_no_merge_on_small_n():
    """On n=6 with min_freq=0.5, the Wilson CI covers each modality's proportion so
    none is *significantly* below the floor — nothing gets merged."""

    df = pd.DataFrame({"feature1": ["A", "B", "A", "C", "B", "A"], "feature2": ["X", "Y", "X", "Z", "Y", "X"]})
    y = pd.Series([1, 0, 1, 0, 1, 0])
    feature1 = OrdinalFeature("feature1", ["A", "B", "C"])
    feature2 = OrdinalFeature("feature2", ["X", "Y", "Z"])
    discretizer = OrdinalDiscretizer([feature1, feature2], min_freq=3 / 6)
    discretizer.fit(df, y)

    assert feature1.values == GroupedList(["A", "B", "C"])
    assert feature2.values == GroupedList(["X", "Y", "Z"])

    transformed = discretizer.transform(df)
    pd.testing.assert_frame_equal(transformed, df)


def test_ordinal_discretizer_merges_at_large_n():
    """At large n the CI is tight and the discretizer merges the genuinely-rare modality."""

    # n=600, C / Z modality count = 50 (~8%) vs min_freq=50% → Wilson upper ≈ 0.107 < 0.5 → merge.
    df = pd.DataFrame(
        {
            "feature1": ["A"] * 300 + ["B"] * 250 + ["C"] * 50,
            "feature2": ["X"] * 300 + ["Y"] * 250 + ["Z"] * 50,
        }
    )
    y = pd.Series([0] * 600)
    feature1 = OrdinalFeature("feature1", ["A", "B", "C"])
    feature2 = OrdinalFeature("feature2", ["X", "Y", "Z"])
    discretizer = OrdinalDiscretizer([feature1, feature2], min_freq=3 / 6)
    discretizer.fit(df, y)

    expected_feature1 = GroupedList(["A", "B", "C"])
    expected_feature1.group("C", "B")
    assert feature1.values == expected_feature1

    expected_feature2 = GroupedList(["X", "Y", "Z"])
    expected_feature2.group("Z", "Y")
    assert feature2.values == expected_feature2


def test_ordinal_discretizer_with_nans():
    """Test OrdinalDiscretizer with basic input"""
    df = pd.DataFrame({"feature1": ["A", "B", np.nan, "C", "B", "A"], "feature2": ["X", "Y", np.nan, "Z", "Y", "X"]})
    y = pd.Series([1, 0, 1, 0, 1, 0])
    feature1 = OrdinalFeature("feature1", ["A", "B", "C"])
    feature2 = OrdinalFeature("feature2", ["X", "Y", "Z"])
    ordinals = [feature1, feature2]
    discretizer = OrdinalDiscretizer(ordinals, min_freq=1 / 6)
    discretizer.fit(df, y)

    # Check feature1
    expected_feature1 = GroupedList(["A", "B", "C"])
    assert feature1.values == expected_feature1
    assert feature1.content == expected_feature1.content

    # Check feature2
    expected_feature2 = GroupedList(["X", "Y", "Z"])
    assert feature2.values == expected_feature2
    assert feature2.content == expected_feature2.content

    # Check transformed data
    transformed = discretizer.transform(df)
    df_expected = pd.DataFrame(
        {
            "feature1": ["A", "B", np.nan, "C", "B", "A"],
            "feature2": ["X", "Y", np.nan, "Z", "Y", "X"],
        }
    )
    print(transformed)
    assert transformed.equals(df_expected), "Transformed data does not match expected data"


def test_ordinal_discretizer_missing_value_on_small_n_keeps_label():
    """A label that is absent (count=0) on n=6 is still within Wilson CI of min_freq=2/6,
    so the OrdinalDiscretizer does not merge it."""

    df = pd.DataFrame({"feature1": ["A", "B", "A", np.nan, "B", "A"], "feature2": ["X", "Y", "X", np.nan, "Y", "X"]})
    y = pd.Series([1, 0, 1, 0, 1, 0])
    feature1 = OrdinalFeature("feature1", ["A", "B", "C"])
    feature2 = OrdinalFeature("feature2", ["X", "Y", "Z"])
    discretizer = OrdinalDiscretizer([feature1, feature2], min_freq=2 / 6)
    discretizer.fit(df, y)

    assert feature1.values == GroupedList(["A", "B", "C"])
    assert feature2.values == GroupedList(["X", "Y", "Z"])


def test_ordinal_discretizer_with_all_underrepresented():
    """min_freq=5/6 is high enough that all modalities of an n=6 sample are
    significantly below it under the Wilson CI → everything collapses."""

    df = pd.DataFrame({"feature1": ["A", "B", "A", "C", "B", "A"], "feature2": ["X", "Y", "X", "Z", "Y", "X"]})
    y = pd.Series([1, 0, 1, 0, 1, 0])
    feature1 = OrdinalFeature("feature1", ["A", "B", "C"])
    feature2 = OrdinalFeature("feature2", ["X", "Y", "Z"])
    discretizer = OrdinalDiscretizer([feature1, feature2], min_freq=5 / 6)
    discretizer.fit(df, y)

    expected_feature1 = GroupedList(["A", "B", "C"])
    expected_feature1.group("A", "B")
    expected_feature1.group("C", "B")
    assert feature1.values == expected_feature1

    expected_feature2 = GroupedList(["X", "Y", "Z"])
    expected_feature2.group("X", "Y")
    expected_feature2.group("Z", "Y")
    assert feature2.values == expected_feature2

    transformed = discretizer.transform(df)
    df_expected = pd.DataFrame(
        {
            "feature1": ["A to C"] * 6,
            "feature2": ["X to Z"] * 6,
        }
    )
    assert transformed.equals(df_expected)


def test_ordinal_discretizer(x_train: pd.DataFrame, target: str) -> None:
    """Tests OrdinalDiscretizer

    Parameters
    ----------
    x_train : pd.DataFrame
        Simulated Train DataFrame
    target: str
        Target feature
    """
    # defining values_orders
    order = ["Low-", "Low", "Low+", "Medium-", "Medium", "Medium+", "High-", "High", "High+"]
    # ordering for base qualitative ordinal feature
    groupedlist = GroupedList(order)

    # ordering for qualitative ordinal feature that contains NaNs
    groupedlist_lownan = GroupedList(order)

    # storing per feature orders
    ordinal_values = {
        "Qualitative_Ordinal": groupedlist,
        "Qualitative_Ordinal_lownan": groupedlist_lownan,
    }
    features = Features(ordinals=ordinal_values)

    # minimum frequency per modality + apply(find_common_modalities) outputs a Series
    min_freq = 0.01

    # discretizing features
    discretizer = OrdinalDiscretizer(ordinals=features, min_freq=min_freq, config=ProcessingConfig(copy=True))
    x_disc = discretizer.fit_transform(x_train, x_train[target])

    feature = "Qualitative_Ordinal"
    expected_ordinal_01 = {
        "Low-": ["Low", "Low-"],
        "Low+": ["Low+"],
        "Medium-": ["Medium-"],
        "Medium": ["Medium"],
        "Medium+": ["High-", "Medium+"],
        "High": ["High"],
        "High+": ["High+"],
    }
    print(
        discretizer.features(feature).content,
        features(feature).content,
        x_disc[feature].value_counts(dropna=False, normalize=True).round(2),
    )
    assert features(feature).content == expected_ordinal_01, "Missing value in order not correctly grouped"

    expected_ordinal_lownan_01 = {
        "Low+": ["Low", "Low-", "Low+"],
        "Medium-": ["Medium-"],
        "Medium": ["Medium"],
        "Medium+": ["High-", "Medium+"],
        "High": ["High"],
        "High+": ["High+"],
    }
    assert features("Qualitative_Ordinal_lownan").content == expected_ordinal_lownan_01, (
        "Missing value in order not correctly grouped or introduced nans."
    )

    # minimum frequency per modality + apply(find_common_modalities) outputs a DataFrame
    min_freq = 0.08

    # discretizing features
    features = Features(ordinals=ordinal_values)
    discretizer = OrdinalDiscretizer(ordinals=features, min_freq=min_freq, config=ProcessingConfig(copy=True))
    discretizer.fit_transform(x_train, x_train[target])

    expected_ordinal_08 = {
        "Low+": ["Low-", "Low", "Low+"],
        "Medium-": ["Medium-"],
        "Medium": ["Medium"],
        "High": ["Medium+", "High-", "High"],
        "High+": ["High+"],
    }
    assert features("Qualitative_Ordinal").content == expected_ordinal_08, "Values not correctly grouped"

    expected_ordinal_lownan_08 = {
        "Low+": ["Low", "Low-", "Low+"],
        "Medium-": ["Medium-"],
        "Medium": ["Medium"],
        "High": ["Medium+", "High-", "High"],
        "High+": ["High+"],
    }
    assert features("Qualitative_Ordinal_lownan").content == expected_ordinal_lownan_08, (
        "NaNs should stay by themselves."
    )
