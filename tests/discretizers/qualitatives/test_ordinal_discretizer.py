"""Set of tests for qualitative_discretizers module."""

from numpy import array, isnan, nan, vstack
from pandas import DataFrame, Series
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
from AutoCarver.features import Features, GroupedList, OrdinalFeature


def test_find_closest_modality_single_element():
    """Test find_closest_modality when there is only one element"""
    idx = 0
    frequencies = array([0.1])
    target_rates = array([0.1])
    min_freq = 0.15
    result = find_closest_modality(idx, frequencies, target_rates, min_freq)
    expected = 0
    assert result == expected


def test_find_closest_modality_lowest_ranked():
    """Test find_closest_modality when idx is the lowest ranked modality"""
    idx = 0
    frequencies = array([0.1, 0.2, 0.3, 0.4])
    target_rates = array([0.1, 0.2, 0.3, 0.4])
    min_freq = 0.15
    result = find_closest_modality(idx, frequencies, target_rates, min_freq)
    expected = 1
    assert result == expected


def test_find_closest_modality_highest_ranked():
    """Test find_closest_modality when idx is the highest ranked modality"""
    idx = 3
    frequencies = array([0.1, 0.2, 0.3, 0.4])
    target_rates = array([0.1, 0.2, 0.3, 0.4])
    min_freq = 0.15
    result = find_closest_modality(idx, frequencies, target_rates, min_freq)
    expected = 2
    assert result == expected


def test_find_closest_modality_middle_closer_previous():
    """Test find_closest_modality when idx is in the middle and next modality is closer"""
    idx = 1
    frequencies = array([0.1, 0.2, 0.1])
    target_rates = array([0.1, 0.2, 0.4])
    min_freq = 0.15
    result = find_closest_modality(idx, frequencies, target_rates, min_freq)
    expected = 0
    assert result == expected


def test_find_closest_modality_middle_closer_next():
    """Test find_closest_modality when idx is in the middle and previous modality is closer"""
    idx = 1
    frequencies = array([0.1, 0.2, 0.1])
    target_rates = array([0.4, 0.2, 0.1])
    min_freq = 0.15
    result = find_closest_modality(idx, frequencies, target_rates, min_freq)
    expected = 2
    assert result == expected


def test_is_next_modality_closer_by_target_rate_closer():
    """Test when the next modality is closer by target rate"""
    idx = 1
    target_rates = array([0.1, 0.3, 0.4])
    result = is_next_modality_closer_by_target_rate(idx, target_rates)
    assert result is True


def test_is_next_modality_closer_by_target_rate_not_closer():
    """Test when the next modality is not closer by target rate"""
    idx = 1
    target_rates = array([0.4, 0.3, 0.1])
    result = is_next_modality_closer_by_target_rate(idx, target_rates)
    assert result is False


def test_is_next_modality_closer_by_target_rate_equal_distance():
    """Test when the next and previous modalities are equally distant by target rate"""
    idx = 1
    target_rates = array([0.2, 0.3, 0.4])
    result = is_next_modality_closer_by_target_rate(idx, target_rates)
    assert result is False


def test_is_next_modality_closer_by_target_rate_negative_values():
    """Test when the target rates include negative values"""
    idx = 1
    target_rates = array([-0.1, -0.3, -0.4])
    result = is_next_modality_closer_by_target_rate(idx, target_rates)
    assert result is True


def test_is_next_modality_closer_by_target_rate_zero_values():
    """Test when the target rates include zero values"""
    idx = 1
    target_rates = array([0.1, 0.0, -0.1])
    result = is_next_modality_closer_by_target_rate(idx, target_rates)
    assert result is False


def test_is_next_modality_closer_no_frequency():
    """Test when the current modality has no frequency"""

    # bigger frequency for the previous modality
    idx = 1
    frequencies = array([0.2, 0.0, 0.3])
    target_rates = array([0.1, 0.2, 0.3])
    min_freq = 0.15
    result = is_next_modality_closer(idx, frequencies, target_rates, min_freq)
    assert not result

    # bigger frequency for the next modality
    frequencies = array([0.3, 0.0, 0.2])
    result = is_next_modality_closer(idx, frequencies, target_rates, min_freq)
    assert result


def test_is_next_modality_closer_underrepresented():
    """Test when the next modality is underrepresented and the previous is not"""

    # next modality is underrepresented
    idx = 1
    frequencies = array([0.3, 0.2, 0.1])
    target_rates = array([0.1, 0.2, 0.3])
    min_freq = 0.15
    result = is_next_modality_closer(idx, frequencies, target_rates, min_freq)
    assert result is True

    # previous modality is underrepresented
    frequencies = array([0.1, 0.2, 0.3])
    result = is_next_modality_closer(idx, frequencies, target_rates, min_freq)
    assert result is False


def test_is_next_modality_closer_both_underrepresented():
    """Test when both the next and previous modalities are underrepresented"""

    # next modality is closer by target rate
    idx = 1
    frequencies = array([0.1, 0.2, 0.1])
    target_rates = array([0.05, 0.2, 0.3])
    min_freq = 0.15
    result = is_next_modality_closer(idx, frequencies, target_rates, min_freq)
    assert result is True

    # previous modality is closer by target rate
    target_rates = array([0.1, 0.2, 0.35])
    result = is_next_modality_closer(idx, frequencies, target_rates, min_freq)
    assert result is False


def test_is_next_modality_closer_both_overrepresented():
    """Test when both the next and previous modalities are overrepresented"""

    # next modality is closer by target rate
    idx = 1
    frequencies = array([0.4, 0.2, 0.5])
    target_rates = array([0.05, 0.2, 0.3])
    min_freq = 0.15
    result = is_next_modality_closer(idx, frequencies, target_rates, min_freq)
    assert result is True

    # previous modality is closer by target rate
    target_rates = array([0.1, 0.2, 0.35])
    result = is_next_modality_closer(idx, frequencies, target_rates, min_freq)
    assert result is False


def test_compute_stats_basic():
    """Test compute_stats with basic input"""
    df_feature = Series(["A", "B", "A", "C", "B", "A"])
    y = Series([1, 0, 1, 0, 1, 0])
    labels = GroupedList(["A", "B", "C"])
    stats, len_df = compute_stats(df_feature, y, labels)
    expected_stats = vstack((array([3, 2, 1]), array([2, 1, 0])))  # frequencies  # target rates
    expected_len_df = 6
    assert (stats == expected_stats).all()
    assert len_df == expected_len_df


def test_compute_stats_with_nans():
    """Test compute_stats with NaN values in df_feature"""
    df_feature = Series(["A", "B", nan, "C", "B", "A"])
    y = Series([1, 0, 1, 0, 1, 0])
    labels = GroupedList(["A", "B", "C"])
    stats, len_df = compute_stats(df_feature, y, labels)
    expected_stats = vstack((array([2, 2, 1]), array([1, 1, 0])))  # frequencies  # target rates
    expected_len_df = 6  # len_df includes NaN values
    assert (stats == expected_stats).all()
    assert len_df == expected_len_df


def test_compute_stats_with_missing_value():
    """Test compute_stats with missing value in df_feature"""
    df_feature = Series(["A", "B", "A", nan, "B", "A"])
    y = Series([1, 0, 1, 0, 1, 0])
    labels = GroupedList(["A", "B", "C"])
    stats, len_df = compute_stats(df_feature, y, labels)
    expected_stats = vstack((array([3, 2, 0]), array([2, 1, nan])))  # frequencies  # target rates
    expected_len_df = 6  # len_df includes NaN values

    # Check frequencies
    assert (stats[0] == expected_stats[0]).all()
    # Check target rates with isnan
    assert (isnan(stats[1]) == isnan(expected_stats[1])).all()
    assert (stats[1][~isnan(stats[1])] == expected_stats[1][~isnan(expected_stats[1])]).all()
    assert len_df == expected_len_df


def test_compute_stats_all_nans():
    """Test compute_stats with all NaN values in df_feature"""
    df_feature = Series([None, None, None])
    y = Series([1, 0, 1])
    labels = GroupedList(["A", "B", "C"])
    stats, len_df = compute_stats(df_feature, y, labels)
    expected_stats = vstack((array([0, 0, 0]), array([nan, nan, nan])))  # frequencies  # target rates
    expected_len_df = 3  # len_df includes NaN values

    # Check frequencies
    assert (stats[0] == expected_stats[0]).all()
    # Check target rates with isnan
    assert isnan(stats[1]).all() and isnan(expected_stats[1]).all()
    assert len_df == expected_len_df


def test_compute_stats_empty_input():
    """Test compute_stats with empty input"""
    df_feature = Series([])
    y = Series([])
    labels = GroupedList([])
    stats, len_df = compute_stats(df_feature, y, labels)
    expected_stats = vstack((array([]), array([])))  # frequencies  # target rates
    expected_len_df = 0  # empty input
    assert (stats == expected_stats).all()
    assert len_df == expected_len_df


def test_compute_stats_single_modality():
    """Test compute_stats with a single modality"""
    df_feature = Series(["A", "A", "A"])
    y = Series([1, 0, 1])
    labels = GroupedList(["A"])
    stats, len_df = compute_stats(df_feature, y, labels)
    expected_stats = vstack((array([3]), array([2])))  # frequencies  # target rates
    expected_len_df = 3
    assert (stats == expected_stats).all()
    assert len_df == expected_len_df


def test_update_stats_basic():
    """Test update_stats with basic input"""
    df_feature = Series(["A", "B", "A", "C", "B", "A"])
    y = Series([1, 0, 1, 0, 1, 0])
    labels = GroupedList(["A", "B", "C"])
    stats, _ = compute_stats(df_feature, y, labels)
    updated_stats = update_stats(stats, discarded_idx=2, kept_idx=0)
    expected_stats = vstack(
        (array([4, 2]), array([2, 1]))  # updated frequencies  # updated target rates
    )
    assert (updated_stats == expected_stats).all()


def test_update_stats_with_nans():
    """Test update_stats with NaN values in df_feature"""
    df_feature = Series(["A", "B", None, "C", "B", "A"])
    y = Series([1, 0, 1, 0, 1, 0])
    labels = GroupedList(["A", "B", "C"])
    stats, _ = compute_stats(df_feature, y, labels)
    updated_stats = update_stats(stats, discarded_idx=2, kept_idx=0)
    expected_stats = vstack(
        (array([3, 2]), array([1, 1]))  # updated frequencies  # updated target rates
    )
    assert (updated_stats == expected_stats).all()


def test_update_stats_with_missing_value():
    """Test update_stats with missing value in df_feature"""
    df_feature = Series(["A", "B", "A", nan, "B", "A"])
    y = Series([1, 0, 1, 0, 1, 0])
    labels = GroupedList(["A", "B", "C"])
    stats, _ = compute_stats(df_feature, y, labels)
    updated_stats = update_stats(stats, discarded_idx=2, kept_idx=0)
    expected_stats = vstack(
        (array([3, 2]), array([2, 1]))  # updated frequencies  # updated target rates
    )
    assert (updated_stats == expected_stats).all()


def test_update_stats_all_nans():
    """Test update_stats with all NaN values in df_feature"""
    df_feature = Series([None, None, None])
    y = Series([1, 0, 1])
    labels = GroupedList(["A", "B", "C"])
    stats, _ = compute_stats(df_feature, y, labels)
    updated_stats = update_stats(stats, discarded_idx=2, kept_idx=0)
    expected_stats = vstack(
        (array([0, 0]), array([nan, nan]))  # updated frequencies  # updated target rates
    )
    assert (updated_stats[0] == expected_stats[0]).all()
    assert (isnan(updated_stats[1]) == isnan(expected_stats[1])).all()


def test_update_stats_empty_input():
    """Test update_stats with empty input"""
    df_feature = Series([])
    y = Series([])
    labels = GroupedList([])
    stats, _ = compute_stats(df_feature, y, labels)
    with raises(IndexError):
        update_stats(stats, discarded_idx=0, kept_idx=0)


def test_update_stats_single_modality():
    """Test update_stats with a single modality"""
    df_feature = Series(["A", "A", "A"])
    y = Series([1, 0, 1])
    labels = GroupedList(["A"])
    stats, _ = compute_stats(df_feature, y, labels)
    with raises(IndexError):
        update_stats(stats, discarded_idx=0, kept_idx=1)


def test_find_common_modalities_all_overrepresented():
    """Test find_common_modalities with all values overrepresented"""
    df_feature = Series(["A", "B", "A", "C", "B", "A"])
    y = Series([1, 0, 1, 0, 1, 0])
    min_freq = 0 / 6
    labels = ["A", "B", "C"]
    result = find_common_modalities(df_feature, y, min_freq, labels)
    expected = GroupedList(labels)
    assert result == expected
    assert result.content == expected.content


def test_find_common_modalities_all_underrepresented():
    """Test find_common_modalities with all values underrepresented"""
    df_feature = Series(["A", "B", "A", "C", "B", "A"])
    y = Series([1, 0, 1, 0, 1, 0])
    min_freq = 4 / 6
    labels = ["A", "B", "C"]
    result = find_common_modalities(df_feature, y, min_freq, labels)
    expected = GroupedList(labels)
    expected.group("C", "B")
    expected.group("A", "B")
    assert result == expected
    assert result.content == expected.content


def test_find_common_modalities_with_nans():
    """Test find_common_modalities with NaN values in df_feature"""
    df_feature = Series(["A", "B", nan, "C", "B", "A"])
    y = Series([1, 0, 1, 0, 1, 0])
    min_freq = 2 / 6
    labels = ["A", "B", "C"]
    result = find_common_modalities(df_feature, y, min_freq, labels)
    expected = GroupedList(labels)
    expected.group("C", "B")
    assert result == expected
    assert result.content == expected.content


def test_find_common_modalities_with_equal_min_freq():
    """Test find_common_modalities with equal min_freq"""
    df_feature = Series(["A", "B", "A", "C", "B", "A"])
    y = Series([1, 0, 1, 0, 1, 0])
    min_freq = 1 / 6
    labels = ["A", "B", "C"]
    result = find_common_modalities(df_feature, y, min_freq, labels)
    expected = GroupedList(labels)
    assert result == expected
    assert result.content == expected.content


def test_find_common_modalities_with_increased_freq():
    """Test find_common_modalities with increased min_freq"""
    df_feature = Series(["A", "B", "A", "C", "B", "A"])
    y = Series([1, 0, 1, 0, 1, 0])
    min_freq = 2 / 6
    labels = ["A", "B", "C"]
    result = find_common_modalities(df_feature, y, min_freq, labels)
    expected = GroupedList(labels)
    expected.group("C", "B")
    assert result == expected
    assert result.content == expected.content


def test_find_common_modalities_with_missing_value():
    """Test find_common_modalities with missing value in df_feature"""
    df_feature = Series(["A", "B", "A", nan, "B", "A"])
    y = Series([1, 0, 1, 0, 1, 0])
    min_freq = 2 / 6
    labels = ["A", "B", "C"]
    result = find_common_modalities(df_feature, y, min_freq, labels)
    expected = GroupedList(labels)
    expected.group("C", "B")
    assert result == expected
    assert result.content == expected.content


def test_find_common_modalities_empty_input():
    """Test find_common_modalities with empty input"""
    df_feature = Series([])
    y = Series([])
    min_freq = 0.2
    labels = []
    result = find_common_modalities(df_feature, y, min_freq, labels)
    expected = GroupedList([])
    assert result == expected
    assert result.content == expected.content


def test_find_common_modalities_single_modality():
    """Test find_common_modalities with a single modality"""
    df_feature = Series(["A", "A", "A"])
    y = Series([1, 0, 1])
    min_freq = 0.2
    labels = ["A"]
    result = find_common_modalities(df_feature, y, min_freq, labels)
    expected = GroupedList(labels)
    assert result == expected
    assert result.content == expected.content


def test_find_common_modalities_unexpected_value():
    """Test find_common_modalities with an unexpected value in df_feature"""
    df_feature = Series(["A", "B", "A", "C", "B", "A"])
    y = Series([1, 0, 1, 0, 1, 0])
    min_freq = 2 / 6
    labels = ["A", "B"]
    result = find_common_modalities(df_feature, y, min_freq, labels)
    expected = GroupedList(labels)
    assert result == expected


def test_ordinal_discretizer_with_increasing_freq():
    """Test OrdinalDiscretizer with basic input"""
    df = DataFrame({"feature1": ["A", "B", "A", "C", "B", "A"], "feature2": ["X", "Y", "X", "Z", "Y", "X"]})
    y = Series([1, 0, 1, 0, 1, 0])
    feature1 = OrdinalFeature("feature1", ["A", "B", "C"])
    feature2 = OrdinalFeature("feature2", ["X", "Y", "Z"])
    ordinals = [feature1, feature2]
    discretizer = OrdinalDiscretizer(ordinals, min_freq=2 / 6)
    discretizer.fit(df, y)

    # Check feature1
    expected_feature1 = GroupedList(["A", "B", "C"])
    expected_feature1.group("C", "B")
    assert feature1.values == expected_feature1
    assert feature1.content == expected_feature1.content

    # Check feature2
    expected_feature2 = GroupedList(["X", "Y", "Z"])
    expected_feature2.group("Z", "Y")
    assert feature2.values == expected_feature2
    assert feature2.content == expected_feature2.content

    # Check transformed data
    transformed = discretizer.transform(df)
    assert not transformed.isnull().values.any(), "Transformed data should not contain NaNs"
    df_expected = DataFrame(
        {
            "feature1": ["A", "B to C", "A", "B to C", "B to C", "A"],
            "feature2": ["X", "Y to Z", "X", "Y to Z", "Y to Z", "X"],
        }
    )
    print(transformed)
    assert transformed.equals(df_expected), "Transformed data does not match expected data"


def test_ordinal_discretizer_with_nans():
    """Test OrdinalDiscretizer with basic input"""
    df = DataFrame({"feature1": ["A", "B", nan, "C", "B", "A"], "feature2": ["X", "Y", nan, "Z", "Y", "X"]})
    y = Series([1, 0, 1, 0, 1, 0])
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
    df_expected = DataFrame(
        {
            "feature1": ["A", "B", nan, "C", "B", "A"],
            "feature2": ["X", "Y", nan, "Z", "Y", "X"],
        }
    )
    print(transformed)
    assert transformed.equals(df_expected), "Transformed data does not match expected data"


def test_ordinal_discretizer_with_missing_value():
    """Test OrdinalDiscretizer with basic input"""
    df = DataFrame({"feature1": ["A", "B", "A", nan, "B", "A"], "feature2": ["X", "Y", "X", nan, "Y", "X"]})
    y = Series([1, 0, 1, 0, 1, 0])
    feature1 = OrdinalFeature("feature1", ["A", "B", "C"])
    feature2 = OrdinalFeature("feature2", ["X", "Y", "Z"])
    ordinals = [feature1, feature2]
    discretizer = OrdinalDiscretizer(ordinals, min_freq=1 / 6)
    discretizer.fit(df, y)

    # Check feature1
    expected_feature1 = GroupedList(["A", "B", "C"])
    expected_feature1.group("C", "B")
    assert feature1.values == expected_feature1
    assert feature1.content == expected_feature1.content

    # Check feature2
    expected_feature2 = GroupedList(["X", "Y", "Z"])
    expected_feature2.group("Z", "Y")
    assert feature2.values == expected_feature2
    assert feature2.content == expected_feature2.content

    # Check transformed data
    transformed = discretizer.transform(df)
    df_expected = DataFrame(
        {
            "feature1": ["A", "B to C", "A", nan, "B to C", "A"],
            "feature2": ["X", "Y to Z", "X", nan, "Y to Z", "X"],
        }
    )
    assert transformed.equals(df_expected), "Transformed data does not match expected data"


def test_ordinal_discretizer_with_all_underrepresented():
    """Test OrdinalDiscretizer with basic input"""
    df = DataFrame({"feature1": ["A", "B", "A", "C", "B", "A"], "feature2": ["X", "Y", "X", "Z", "Y", "X"]})
    y = Series([1, 0, 1, 0, 1, 0])
    feature1 = OrdinalFeature("feature1", ["A", "B", "C"])
    feature2 = OrdinalFeature("feature2", ["X", "Y", "Z"])
    ordinals = [feature1, feature2]
    discretizer = OrdinalDiscretizer(ordinals, min_freq=5 / 6)
    discretizer.fit(df, y)

    # Check feature1
    expected_feature1 = GroupedList(["A", "B", "C"])
    expected_feature1.group("A", "B")
    expected_feature1.group("C", "B")
    assert feature1.values == expected_feature1
    assert feature1.content == expected_feature1.content

    # Check feature2
    expected_feature2 = GroupedList(["X", "Y", "Z"])
    expected_feature2.group("X", "Y")
    expected_feature2.group("Z", "Y")
    assert feature2.values == expected_feature2
    assert feature2.content == expected_feature2.content

    # Check transformed data
    transformed = discretizer.transform(df)
    df_expected = DataFrame(
        {
            "feature1": ["A to C", "A to C", "A to C", "A to C", "A to C", "A to C"],
            "feature2": ["X to Z", "X to Z", "X to Z", "X to Z", "X to Z", "X to Z"],
        }
    )
    assert transformed.equals(df_expected), "Transformed data does not match expected data"


def test_ordinal_discretizer(x_train: DataFrame, target: str) -> None:
    """Tests OrdinalDiscretizer

    Parameters
    ----------
    x_train : DataFrame
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
    discretizer = OrdinalDiscretizer(ordinals=features, min_freq=min_freq, copy=True)
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
    discretizer = OrdinalDiscretizer(ordinals=features, min_freq=min_freq, copy=True)
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
