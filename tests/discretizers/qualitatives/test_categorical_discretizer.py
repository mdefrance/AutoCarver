"""Set of tests for qualitative_discretizers module."""

import numpy as np
import pandas as pd
from pytest import raises

from AutoCarver.discretizers.qualitatives.categorical_discretizer import (
    CategoricalDiscretizer,
    series_target_rate,
    series_value_counts,
)
from AutoCarver.discretizers.utils.base_discretizer import ProcessingConfig, Sample
from AutoCarver.features import CategoricalFeature, Features, FeaturesConfig, GroupedList


def test_series_value_counts():
    """Tests series_value_counts function — returns integer counts (not proportions)."""
    x = pd.Series(["a", "b", "a", "b", "c", "c", "c"])
    result = series_value_counts(x, dropna=False)
    assert result == {"c": 3, "a": 2, "b": 2}

    # with NaN, dropna=True
    x = pd.Series(["a", "b", "a", "b", "c", "c", "c", None])
    result = series_value_counts(x, dropna=True)
    assert result == {"c": 3, "a": 2, "b": 2}

    # with NaN, dropna=False
    x = pd.Series(["a", "b", "a", "b", "c", "c", "c", None])
    result = series_value_counts(x, dropna=False)
    assert result == {"c": 3, "a": 2, "b": 2, None: 1}


def test_series_target_rate():
    """Tests series_target_rate function"""
    # Test series_target_rate with basic input
    x = pd.Series(["a", "b", "a", "b", "c", "c", "c"])
    y = pd.Series([1, 0, 1, 0, 1, 0, 1])
    result = series_target_rate(x, y)
    expected = {"a": 1.0, "b": 0.0, "c": 2 / 3}
    assert result == expected

    # Test series_target_rate with np.nan values in x
    x = pd.Series(["a", "b", "a", "b", "c", "c", "c", None])
    y = pd.Series([1, 0, 1, 0, 1, 0, 1, 1])
    result = series_target_rate(x, y)
    expected = {"a": 1.0, "b": 0.0, "c": 2 / 3}
    assert result == expected

    # Test series_target_rate when all targets are the same
    x = pd.Series(["a", "b", "a", "b", "c", "c", "c"])
    y = pd.Series([1, 1, 1, 1, 1, 1, 1])
    result = series_target_rate(x, y)
    expected = {"a": 1.0, "b": 1.0, "c": 1.0}
    assert result == expected

    # Test series_target_rate with empty series
    x = pd.Series([])
    y = pd.Series([])
    result = series_target_rate(x, y)
    expected = {}
    assert result == expected


def test_categoricaldiscretizer_initialization():
    """Tests CategoricalDiscretizer initialization"""
    features = [CategoricalFeature("feature1"), CategoricalFeature("feature2")]
    categorical_discretizer = CategoricalDiscretizer(features, min_freq=0.02)
    assert isinstance(categorical_discretizer.features, Features)
    assert categorical_discretizer.min_freq == 0.02


def test_prepare_sample():
    """Tests CategoricalDiscretizer _prepare_sample method"""

    feature1 = CategoricalFeature("feature1")
    feature2 = CategoricalFeature("feature2")
    features = [feature1, feature2]
    categorical_discretizer = CategoricalDiscretizer(features, min_freq=0.02)

    X = pd.DataFrame({"feature1": ["a", "b", "a", "b", np.nan], "feature2": ["x", "y", "x", "y", np.nan]})

    sample = categorical_discretizer._prepare_sample(Sample(X, None))
    X_prepared = sample.X

    assert X_prepared["feature1"].tolist() == ["a", "b", "a", "b", np.nan]
    assert X_prepared["feature2"].tolist() == ["x", "y", "x", "y", np.nan]


def test_group_feature_rare_modalities():
    """Tests CategoricalDiscretizer._group_feature_rare_modalities with the Wilson CI.

    At n=1000 and min_freq=0.10, modalities with count ≈ 50 (~5%, Wilson upper ≈ 0.066)
    are significantly below 10% and get grouped, while modalities with count ≈ 300
    (~30%, Wilson upper > 0.10) survive.
    """

    feature1 = CategoricalFeature("feature1")
    feature2 = CategoricalFeature("feature2")
    categorical_discretizer = CategoricalDiscretizer([feature1, feature2], min_freq=0.10)

    X = pd.DataFrame(
        {
            "feature1": ["x"] * 50 + ["a"] * 50 + ["b"] * 300 + ["c"] * 600,
            "feature2": ["a"] * 50 + ["x"] * 50 + ["y"] * 300 + ["z"] * 600,
        }
    )
    categorical_discretizer.features.fit(X)
    frequencies = X[categorical_discretizer.features.versions].apply(series_value_counts, axis=0)
    grouped_x = categorical_discretizer._group_feature_rare_modalities(feature1, X, frequencies)

    assert feature1.has_default
    # "x" and "a" merge into default; "b" and "c" survive
    assert sorted(set(grouped_x["feature1"].values)) == sorted([feature1.default, "b", "c"])
    assert set(feature1.content[feature1.default]) == {"a", "x", feature1.default}

    # test with very low min_freq — nothing should be grouped
    feature1 = CategoricalFeature("feature1")
    feature2 = CategoricalFeature("feature2")
    categorical_discretizer = CategoricalDiscretizer([feature1, feature2], min_freq=0.001)

    X = pd.DataFrame(
        {
            "feature1": ["x", "b", "a", "b", "c", "c", "c"],
            "feature2": ["a", "y", "x", "y", "z", "z", "z"],
        }
    )
    categorical_discretizer.features.fit(X)
    frequencies = X[categorical_discretizer.features.versions].apply(series_value_counts, axis=0)
    grouped_x = categorical_discretizer._group_feature_rare_modalities(feature1, X, frequencies)

    assert not feature1.has_default
    assert grouped_x["feature1"].tolist() == ["x", "b", "a", "b", "c", "c", "c"]
    assert feature1.values == ["x", "b", "a", "c"]

    # min_freq = 2.0 forces every modality significantly below the floor → everything grouped.
    feature1 = CategoricalFeature("feature1")
    feature2 = CategoricalFeature("feature2")
    categorical_discretizer = CategoricalDiscretizer([feature1, feature2], min_freq=2.0)

    X = pd.DataFrame(
        {
            "feature1": ["x", "b", "a", "b", "c", "c", "c"],
            "feature2": ["a", "y", "x", "y", "z", "z", "z"],
        }
    )
    categorical_discretizer.features.fit(X)
    frequencies = X[categorical_discretizer.features.versions].apply(series_value_counts, axis=0)
    grouped_x = categorical_discretizer._group_feature_rare_modalities(feature1, X, frequencies)

    assert feature1.has_default
    assert grouped_x["feature1"].tolist() == [feature1.default] * 7
    assert set(feature1.content[feature1.default]) == {"a", "x", "b", "c", feature1.default}


def test_categorical_ci_gating_keeps_near_threshold_groups_clearly_rare():
    """At n=1000 and min_freq=0.05, a modality at count=49 is within the Wilson CI of
    min_freq (upper ≈ 0.065 > 0.05) → survives. A modality at count=20 (upper ≈ 0.031)
    is significantly below 5% and gets grouped into the default.
    """

    n = 1000
    feature1 = CategoricalFeature("feature1")
    categorical_discretizer = CategoricalDiscretizer([feature1], min_freq=0.05)
    # "edge" at 49 rows — Wilson upper ≈ 0.065 → not significantly below 5% → keep
    # "rare" at 20 rows — Wilson upper ≈ 0.031 → significantly below 5% → group
    X = pd.DataFrame({"feature1": ["edge"] * 49 + ["rare"] * 20 + ["fat"] * (n - 69)})
    categorical_discretizer.features.fit(X)
    frequencies = X[categorical_discretizer.features.versions].apply(series_value_counts, axis=0)
    grouped_x = categorical_discretizer._group_feature_rare_modalities(feature1, X, frequencies)

    assert "edge" in grouped_x["feature1"].values, "near-threshold modality must survive under CI"
    assert "rare" not in grouped_x["feature1"].values, "clearly-rare modality must be grouped"


def test_carver_halves_min_freq_so_categorical_keeps_mid_freq_modalities():
    """End-to-end regression for the ``other_parties`` example: when the user runs a
    Carver with ``min_freq=0.05``, the Carver passes ``min_freq/2 = 0.025`` to
    CategoricalDiscretizer. A 4.67%-frequent modality (well above 2.5%) survives,
    whereas the old strict ``< min_freq`` comparison inside CategoricalDiscretizer
    would have grouped it into ``__OTHER__``.
    """

    n = 1000
    # "mid" at 4.7% — well above 2.5% but below 5%
    feature1 = CategoricalFeature("feature1")
    categorical_discretizer = CategoricalDiscretizer([feature1], min_freq=0.025)  # carver-halved
    X = pd.DataFrame({"feature1": ["mid"] * 47 + ["fat"] * (n - 47)})
    categorical_discretizer.features.fit(X)
    frequencies = X[categorical_discretizer.features.versions].apply(series_value_counts, axis=0)
    grouped_x = categorical_discretizer._group_feature_rare_modalities(feature1, X, frequencies)

    assert "mid" in grouped_x["feature1"].values, "4.7% modality must survive under carver-halved min_freq"
    assert not feature1.has_default


def test_group_rare_modalities():
    """Tests CategoricalDiscretizer._group_rare_modalities — at n=1000 with min_freq=0.10
    the singleton-count modalities are clearly significantly below the floor.
    """
    feature1 = CategoricalFeature("feature1")
    feature2 = CategoricalFeature("feature2")
    categorical_discretizer = CategoricalDiscretizer([feature1, feature2], min_freq=0.10)

    n = 1000
    X = pd.DataFrame(
        {
            "feature1": ["x"] * 50 + ["a"] * 50 + ["b"] * 300 + ["c"] * (n - 400),
            "feature2": ["a"] * 50 + ["x"] * 50 + ["y"] * 300 + ["z"] * (n - 400),
        }
    )
    categorical_discretizer.features.fit(X)
    grouped_x = categorical_discretizer._group_rare_modalities(X)
    assert isinstance(grouped_x, pd.DataFrame)
    assert feature1.has_default
    assert feature2.has_default


def test_target_sort():
    """Tests CategoricalDiscretizer _target_sort method"""
    feature1 = CategoricalFeature("feature1")
    feature2 = CategoricalFeature("feature2")
    categorical_discretizer = CategoricalDiscretizer([feature1, feature2], min_freq=0.2)

    X = pd.DataFrame({"feature1": ["a", "b", "a", "b"], "feature2": ["x", "y", "x", "y"]})
    y = pd.Series([1, 0, 1, 0])

    categorical_discretizer.features.fit(X)
    categorical_discretizer._target_sort(X, y)

    assert feature1.values == ["b", "a"]
    assert feature2.values == ["y", "x"]


def test_categoricaldiscretizer_fit():
    """Tests CategoricalDiscretizer fit method.

    Scaled to n≈1000 so the Wilson CI is tight enough to flag the 5%-frequent
    "x"/"a" modalities as significantly below min_freq=0.10.
    """

    n = 999
    base_x = ["x"] * 50 + ["a"] * 50 + ["b"] * 300 + ["c"] * (n - 400) + [np.nan]
    base_y_binary = [1] * 50 + [1] * 50 + [0] * 300 + [1] * (n - 400) + [1]

    # test with binary target
    feature1 = CategoricalFeature("feature1")
    feature2 = CategoricalFeature("feature2")
    categorical_discretizer = CategoricalDiscretizer([feature1, feature2], min_freq=0.10)

    X = pd.DataFrame({"feature1": base_x, "feature2": base_x})
    y = pd.Series(base_y_binary)

    categorical_discretizer.fit(X, y)

    assert feature1.has_default
    assert feature2.has_default
    assert set(feature1.values) == {"c", "b", feature1.default}
    assert set(feature2.values) == {"c", "b", feature2.default}

    transformed_x = categorical_discretizer.transform(X)
    # the two near-threshold modalities x/a both land in default; b/c keep their labels.
    f1_default_label = feature1.label_per_value[feature1.default]
    f2_default_label = feature2.label_per_value[feature2.default]
    assert (transformed_x["feature1"].iloc[:100] == f1_default_label).all()
    assert (transformed_x["feature1"].iloc[100:400] == "b").all()
    assert (transformed_x["feature1"].iloc[400:-1] == "c").all()
    assert pd.isna(transformed_x["feature1"].iloc[-1])
    assert (transformed_x["feature2"].iloc[:100] == f2_default_label).all()

    # test with continuous target
    feature1 = CategoricalFeature("feature1")
    feature2 = CategoricalFeature("feature2")
    categorical_discretizer = CategoricalDiscretizer([feature1, feature2], min_freq=0.10)

    X = pd.DataFrame({"feature1": base_x, "feature2": base_x})
    rng = np.random.default_rng(0)
    y_cont = pd.Series(rng.normal(loc=base_y_binary, scale=0.1))

    categorical_discretizer.fit(X, y_cont)

    assert feature1.has_default
    assert feature2.has_default


def test_categorical_discretizer(x_train: pd.DataFrame, target: str) -> None:
    """Tests CategoricalDiscretizer

    Parameters
    ----------
    x_train : pd.DataFrame
        Simulated Train DataFrame
    target: str
        Target feature
    """

    # setting new str_default
    str_default = "_DEFAULT_"

    # defining values_orders
    order = ["Category A", "Category B", "Category C", "Category D", "Category E", "Category F"]
    # ordering for base qualitative ordinal feature
    groupedlist = GroupedList(order)

    # ordering for qualitative ordinal feature that contains NaNs
    groupedlist_grouped = GroupedList(order)
    groupedlist_grouped.group("Category A", "Category D")

    # ordering for base qualitative ordinal feature
    order = ["Low-", "Low", "Low+", "Medium-", "Medium", "Medium+", "High-", "High", "High+"]
    groupedlist_ordinal = GroupedList(order)
    groupedlist_ordinal.group(["Low-", "Low"], "Low+")
    groupedlist_ordinal.group(["Medium+", "High-"], "High")

    # storing per feature orders
    values_orders = {
        "Qualitative_grouped": GroupedList(groupedlist_grouped),
        "Qualitative_highnan": GroupedList(groupedlist),
        "Qualitative_lownan": GroupedList(groupedlist),
        "Qualitative_Ordinal": GroupedList(groupedlist_ordinal),
    }

    categoricals = [
        "Qualitative_grouped",
        "Qualitative_lownan",
        "Qualitative_highnan",
        "Qualitative_Ordinal",
    ]
    features = Features(categoricals=categoricals, config=FeaturesConfig(default=str_default))
    features.update(values_orders, replace=True)

    min_freq = 0.02
    # unwanted value in values_orders
    with raises(ValueError):
        discretizer = CategoricalDiscretizer(
            categoricals=features, min_freq=min_freq, config=ProcessingConfig(copy=True)
        )
        _ = discretizer.fit_transform(x_train, x_train[target])

    # correct feature ordering
    features = Features(categoricals=categoricals + ["Qualitative"], config=FeaturesConfig(default=str_default))

    discretizer = CategoricalDiscretizer(categoricals=features, min_freq=min_freq, config=ProcessingConfig(copy=True))
    _ = discretizer.fit_transform(x_train, x_train[target])

    assert features("Qualitative_Ordinal").values.get(str_default) == [
        "Low-",
        str_default,
    ], "Non frequent modalities should be grouped with default value."

    quali_expected_order = {
        "binary_target": ["Category D", str_default, "Category F", "Category C", "Category E"],
        "continuous_target": [str_default, "Category C", "Category E", "Category F", "Category D"],
    }
    assert features("Qualitative").values == quali_expected_order[target], "Incorrect ordering by target rate"

    quali_expected = {
        str_default: ["Category A", str_default],
        "Category C": ["Category C"],
        "Category F": ["Category F"],
        "Category E": ["Category E"],
        "Category D": ["Category D"],
    }
    assert features("Qualitative").content == quali_expected, (
        "Values less frequent than min_freq should be grouped into default_value"
    )

    quali_lownan_expected_order = {
        "binary_target": [
            "Category D",
            "Category F",
            "Category C",
            "Category E",
        ],
        "continuous_target": [
            "Category C",
            "Category E",
            "Category F",
            "Category D",
        ],
    }
    assert features("Qualitative_lownan").values == quali_lownan_expected_order[target], (
        "Incorrect ordering by target rate"
    )

    quali_lownan_expected = {
        "Category C": ["Category C"],
        "Category F": ["Category F"],
        "Category E": ["Category E"],
        "Category D": ["Category D"],
    }
    assert features("Qualitative_lownan").content == quali_lownan_expected, (
        "If any, np.nan values should be put into str_nan and kept by themselves"
    )

    quali_highnan_expected_order = {
        "binary_target": [
            "Category D",
            str_default,
            "Category C",
            "Category E",
        ],
        "continuous_target": [
            str_default,
            "Category C",
            "Category E",
            "Category D",
        ],
    }
    assert features("Qualitative_highnan").values == quali_highnan_expected_order[target], (
        "Incorrect ordering by target rate"
    )

    quali_highnan_expected = {
        str_default: ["Category A", str_default],
        "Category C": ["Category C"],
        "Category E": ["Category E"],
        "Category D": ["Category D"],
    }
    assert features("Qualitative_highnan").content == quali_highnan_expected, (
        "If any, np.nan values should be put into str_nan and kept by themselves"
    )
