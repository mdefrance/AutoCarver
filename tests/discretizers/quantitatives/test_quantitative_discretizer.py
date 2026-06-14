"""Set of tests for discretizers module."""

import numpy as np
import pandas as pd
from pytest import raises

from AutoCarver.discretizers.quantitatives.quantitative_discretizer import (
    QuantitativeDiscretizer,
    check_frequencies,
    check_quantitative_dtypes,
    min_value_counts,
)
from AutoCarver.discretizers.utils.base_discretizer import Sample
from AutoCarver.features import Features, GroupedList, QuantitativeFeature


def test_check_quantitative_dtypes_all_numeric():
    """Test check_quantitative_dtypes with all numeric columns"""
    df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4.0, 5.0, 6.0]})
    feature_versions = ["feature1", "feature2"]
    check_quantitative_dtypes(df, feature_versions, "test")


def test_check_quantitative_dtypes_non_numeric():
    """Test check_quantitative_dtypes with non-numeric columns"""
    df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": ["a", "b", "c"]})
    feature_versions = ["feature1", "feature2"]
    with raises(ValueError):
        check_quantitative_dtypes(df, feature_versions, "test")


def test_check_quantitative_dtypes_mixed_types():
    """Test check_quantitative_dtypes with mixed types"""
    df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4.0, 5.0, 6.0], "feature3": [1.0, "b", "c"]})
    feature_versions = ["feature1", "feature2", "feature3"]
    with raises(ValueError):
        check_quantitative_dtypes(df, feature_versions, "test")
    feature_versions = ["feature1", "feature2"]
    check_quantitative_dtypes(df, feature_versions, "test")


def test_min_value_counts_basic():
    """Test min_value_counts with basic input"""
    x = pd.Series([1, 2, 2, 3, 3, 3], name="feature")
    features = Features.from_list([QuantitativeFeature(name="feature")])
    result = min_value_counts(x, features)
    assert result == 1  # Minimum count is 1


def test_min_value_counts_with_nans():
    """Test min_value_counts with np.nan values"""
    # without dropna
    x = pd.Series([1, 2, 2, 3, 3, 3, np.nan, np.nan], name="feature")
    features = Features.from_list([QuantitativeFeature(name="feature")])
    result = min_value_counts(x, features, dropna=False)
    assert result == 1

    # with dropna
    x = pd.Series([1, 2, 2, 3, 3, 3, np.nan, np.nan], name="feature")
    features = Features.from_list([QuantitativeFeature(name="feature")])
    result = min_value_counts(x, features, dropna=True)
    assert result == 1


def test_min_value_counts_with_labels():
    """Test min_value_counts with predefined labels"""
    # with missing label
    x = pd.Series(
        [
            "(-inf, 1.00e+00]",
            "(1.00e+00, 2.00e+00]",
            "(1.00e+00, 2.00e+00]",
            "(2.00e+00, 3.00e+00]",
            "(2.00e+00, 3.00e+00]",
            "(2.00e+00, 3.00e+00]",
        ],
        name="feature",
    )
    feature = QuantitativeFeature(name="feature")
    feature.update(GroupedList([1, 2, 3, 4, np.inf]))
    features = Features.from_list([feature])
    result = min_value_counts(x, features)
    assert result == 0  # Label 4 has a count of 0

    # without missing label
    x = pd.Series(
        [
            "(-inf, 1.00e+00]",
            "(1.00e+00, 2.00e+00]",
            "(1.00e+00, 2.00e+00]",
            "(2.00e+00, 3.00e+00]",
            "(2.00e+00, 3.00e+00]",
            "(3.00e+00, inf)",
        ],
        name="feature",
    )
    feature = QuantitativeFeature(name="feature")
    feature.update(GroupedList([1, 2, 3, np.inf]))
    features = Features.from_list([feature])
    result = min_value_counts(x, features)
    assert result == 1


def test_check_frequencies_no_rare_modalities():
    """Test check_frequencies with no rare modalities"""
    df = pd.DataFrame({"feature1": [1, 2, 2, 3, 3, 3], "feature2": [4, 4, 4, 5, 5, 5]})
    features = Features.from_list(
        [
            QuantitativeFeature(name="feature1"),
            QuantitativeFeature(name="feature2"),
        ]
    )
    half_min_freq = 0.1
    result = check_frequencies(df, features, half_min_freq, alpha=0.05)
    assert result == []  # No rare modalities


def test_check_frequencies_with_rare_modalities():
    """Test check_frequencies flags features whose smallest modality is significantly
    below ``min_freq`` (Wilson upper bound < min_freq)."""

    # n=8000 modality count 50 → Wilson upper ≈ 0.0083 ≪ 0.025 → flagged
    # n=8000 modality count 400 → Wilson upper ≈ 0.055 ≫ 0.025 → not flagged
    n = 8000
    df = pd.DataFrame(
        {
            "feature1": [0.0] * 50 + [1.0] * (n - 50),
            "feature2": [0.0] * 400 + [1.0] * (n - 400),
        }
    )
    features = Features.from_list([QuantitativeFeature(name="feature1"), QuantitativeFeature(name="feature2")])

    result = check_frequencies(df, features, min_freq=0.025, alpha=0.05)
    versions = [f.version for f in result]
    assert "feature1" in versions
    assert "feature2" not in versions


def test_check_frequencies_with_nans():
    """Test check_frequencies with np.nan values"""
    df = pd.DataFrame({"feature1": [1, 2, 2, 3, 3, np.nan], "feature2": [4, 4, 4, 5, 5, 5]})
    features = Features.from_list(
        [
            QuantitativeFeature(name="feature1"),
            QuantitativeFeature(name="feature2"),
        ]
    )
    half_min_freq = 0.1
    result = check_frequencies(df, features, half_min_freq, alpha=0.05)
    assert result == []  # No rare modalities even with NaNs


def test_check_frequencies_all_rare_modalities():
    """All modalities significantly below min_freq → all features flagged."""
    n = 8000
    # both features have a singleton modality (count=1) — clearly below 0.025 under CI
    df = pd.DataFrame(
        {
            "feature1": [0.0] * 1 + [1.0] * (n - 1),
            "feature2": [0.0] * 2 + [1.0] * (n - 2),
        }
    )
    features = Features.from_list(
        [
            QuantitativeFeature(name="feature1"),
            QuantitativeFeature(name="feature2"),
        ]
    )
    result = check_frequencies(df, features, min_freq=0.025, alpha=0.05)
    versions = {f.version for f in result}
    assert versions == {"feature1", "feature2"}


def test_check_frequencies_borderline_modality_not_flagged_on_small_n():
    """A small sample cannot distinguish a freq just below min_freq from min_freq itself —
    the Wilson CI captures min_freq, so the modality survives."""

    # n=100, freq=0.08 vs min_freq=0.10 — Wilson upper(8, 100, 0.05) ≈ 0.152 > 0.10 → not flagged
    df = pd.DataFrame({"feature1": [0.0] * 8 + [1.0] * 92})
    features = Features.from_list([QuantitativeFeature(name="feature1")])
    result = check_frequencies(df, features, min_freq=0.10, alpha=0.05)
    assert result == []


def test_check_frequencies_borderline_modality_flagged_on_large_n():
    """The same proportion at large n IS significantly below min_freq under the CI."""

    # n=10000, freq=0.08 vs min_freq=0.10 — Wilson upper(800, 10000, 0.05) ≈ 0.0856 < 0.10 → flagged
    n = 10000
    df = pd.DataFrame({"feature1": [0.0] * 800 + [1.0] * (n - 800)})
    features = Features.from_list([QuantitativeFeature(name="feature1")])
    result = check_frequencies(df, features, min_freq=0.10, alpha=0.05)
    assert [f.version for f in result] == ["feature1"]


def test_quantitative_discretizer_initialization():
    """Tests QuantitativeDiscretizer initialization"""
    feature1 = QuantitativeFeature(name="feature1")
    feature2 = QuantitativeFeature(name="feature2")
    quantitatives = [feature1, feature2]
    min_freq = 0.05
    discretizer = QuantitativeDiscretizer(quantitatives=quantitatives, min_freq=min_freq)
    assert discretizer.min_freq == min_freq
    assert feature1 in discretizer.features
    assert feature2 in discretizer.features


def test_quantitative_discretizer_prepare_sample():
    """Tests _prepare_sample method of QuantitativeDiscretizer"""
    quantitatives = [QuantitativeFeature(name="feature1"), QuantitativeFeature(name="feature2")]
    min_freq = 0.05
    discretizer = QuantitativeDiscretizer(quantitatives=quantitatives, min_freq=min_freq)

    data = {"feature1": [1, 2, 3, 4, 5], "feature2": [5.0, 4.0, 3.0, 2.0, 1.0]}
    df = pd.DataFrame(data)
    y = pd.Series([0, 1, 0, 1, 0])

    sample = discretizer._prepare_sample(Sample(df, y))
    prepared_df = sample.X
    assert prepared_df["feature1"].dtype != object
    assert prepared_df["feature1"].tolist() == [1, 2, 3, 4, 5]
    assert prepared_df["feature2"].dtype != object
    assert prepared_df["feature2"].tolist() == [5.0, 4.0, 3.0, 2.0, 1.0]


def test_continuous_discretizer_fit():
    """Test fitting the QuantitativeDiscretizer"""
    feature1 = QuantitativeFeature(name="feature1")
    feature2 = QuantitativeFeature(name="feature2")
    quantitatives = [feature1, feature2]
    min_freq = 0.2
    discretizer = QuantitativeDiscretizer(quantitatives=quantitatives, min_freq=min_freq)
    y = pd.Series([0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1])

    # Create a sample DataFrame
    data = {
        "feature1": [
            2,
            3,
            4,
            5,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            3,
            4,
            5,
            2,
            3,
            4,
            5,
        ],
        "feature2": [
            np.nan,
            3,
            4,
            np.nan,
            1,
            1,
            1,
            1,
            1,
            np.nan,
            np.nan,
            2,
            2,
            2,
            2,
            2,
            np.nan,
            np.nan,
            5,
            np.nan,
            3,
            4,
            5,
        ],
    }

    df = pd.DataFrame(data)

    # Fit the discretizer
    discretizer.fit(df, y)

    # Check if the features have been fitted
    assert feature1.has_nan is False
    assert feature2.has_nan is True
    print(feature1.content)
    print(feature2.content)
    # under the Wilson-CI gating, the borderline ``4`` and ``5`` bins survive
    # on this 23-row sample (CI overlaps min_freq=0.2).
    assert feature1.content == {1: [1], 2: [2], 4: [4], np.inf: [np.inf]}
    assert feature2.content == {1.0: [1.0], 2.0: [2.0], np.inf: [5.0, np.inf]}

    # Check if the discretizer has been fitted
    transformed_df = discretizer.transform(df)
    print(transformed_df)
    expected = pd.DataFrame(
        {
            "feature1": [
                "(1.00e+00, 2.00e+00]",
                "(2.00e+00, 4.00e+00]",
                "(2.00e+00, 4.00e+00]",
                "(4.00e+00, inf)",
                "(-inf, 1.00e+00]",
                "(-inf, 1.00e+00]",
                "(-inf, 1.00e+00]",
                "(-inf, 1.00e+00]",
                "(-inf, 1.00e+00]",
                "(-inf, 1.00e+00]",
                "(1.00e+00, 2.00e+00]",
                "(1.00e+00, 2.00e+00]",
                "(1.00e+00, 2.00e+00]",
                "(1.00e+00, 2.00e+00]",
                "(1.00e+00, 2.00e+00]",
                "(1.00e+00, 2.00e+00]",
                "(2.00e+00, 4.00e+00]",
                "(2.00e+00, 4.00e+00]",
                "(4.00e+00, inf)",
                "(1.00e+00, 2.00e+00]",
                "(2.00e+00, 4.00e+00]",
                "(2.00e+00, 4.00e+00]",
                "(4.00e+00, inf)",
            ],
            "feature2": [
                np.nan,
                "(2.00e+00, inf)",
                "(2.00e+00, inf)",
                np.nan,
                "(-inf, 1.00e+00]",
                "(-inf, 1.00e+00]",
                "(-inf, 1.00e+00]",
                "(-inf, 1.00e+00]",
                "(-inf, 1.00e+00]",
                np.nan,
                np.nan,
                "(1.00e+00, 2.00e+00]",
                "(1.00e+00, 2.00e+00]",
                "(1.00e+00, 2.00e+00]",
                "(1.00e+00, 2.00e+00]",
                "(1.00e+00, 2.00e+00]",
                np.nan,
                np.nan,
                "(2.00e+00, inf)",
                np.nan,
                "(2.00e+00, inf)",
                "(2.00e+00, inf)",
                "(2.00e+00, inf)",
            ],
        }
    )
    assert transformed_df.equals(expected)


def test_quantitative_discretizer(x_train: pd.DataFrame, target: str):
    """Tests QuantitativeDiscretizer

    Parameters
    ----------
    x_train : pd.DataFrame
        Simulated Train DataFrame
    target: str
        Target feature
    """

    quantitatives = [
        "Quantitative",
        "Discrete_Quantitative",
        "Discrete_Quantitative_highnan",
        "Discrete_Quantitative_lownan",
        "Discrete_Quantitative_rarevalue",
    ]
    min_freq = 0.1
    features = Features(numericals=quantitatives)

    discretizer = QuantitativeDiscretizer(quantitatives=features, min_freq=min_freq)
    x_discretized = discretizer.fit_transform(x_train, x_train[target])

    assert not features("Discrete_Quantitative_lownan").values.contains(features("Discrete_Quantitative_lownan").nan), (
        "Missing order should not be grouped with ordinal_discretizer"
    )

    assert all(x_discretized["Quantitative"].value_counts(normalize=True) >= min_freq), (
        "Non-np.nan value was not grouped"
    )

    print(x_train.Discrete_Quantitative_rarevalue.value_counts(dropna=False, normalize=True))

    print(features("Discrete_Quantitative_rarevalue").content)
    # post-pass now uses min_freq directly: ``4.0`` bin (~15%) survives but the inf bucket
    # (5.0 + 6.0 ≈ 8%) is below the floor and gets merged into the ``4.0`` bin (kept as inf).
    assert features("Discrete_Quantitative_rarevalue").values == [
        1.0,
        2.0,
        3.0,
        np.inf,
    ], "Rare values should be grouped to the closest one and np.inf should be kept whatsoever (OrdinalDiscretizer)"
