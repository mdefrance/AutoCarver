"""Set of tests for qualitative_discretizers module."""

from numpy import nan
from pandas import DataFrame, Series
from pytest import raises

from AutoCarver.discretizers.qualitatives.categorical_discretizer import (
    CategoricalDiscretizer,
    series_target_rate,
    series_value_counts,
)
from AutoCarver.discretizers.utils.base_discretizer import Sample
from AutoCarver.features import CategoricalFeature, Features, GroupedList


def test_series_value_counts():
    """Tests series_value_counts function"""
    # Test series_value_counts with normalize=True
    x = Series(["a", "b", "a", "b", "c", "c", "c"])
    result = series_value_counts(x, dropna=False, normalize=True)
    expected = {"c": 3 / 7, "a": 2 / 7, "b": 2 / 7}
    assert result == expected

    # Test series_value_counts with normalize=False
    x = Series(["a", "b", "a", "b", "c", "c", "c"])
    result = series_value_counts(x, dropna=False, normalize=False)
    expected = {"c": 3, "a": 2, "b": 2}
    assert result == expected

    # Test series_value_counts with dropna=True
    x = Series(["a", "b", "a", "b", "c", "c", "c", None])
    result = series_value_counts(x, dropna=True, normalize=True)
    expected = {"c": 3 / 7, "a": 2 / 7, "b": 2 / 7}
    assert result == expected

    # Test series_value_counts with dropna=False
    x = Series(["a", "b", "a", "b", "c", "c", "c", None])
    result = series_value_counts(x, dropna=False, normalize=True)
    expected = {"c": 3 / 8, "a": 2 / 8, "b": 2 / 8, None: 1 / 8}
    assert result == expected


def test_series_target_rate():
    """Tests series_target_rate function"""
    # Test series_target_rate with basic input
    x = Series(["a", "b", "a", "b", "c", "c", "c"])
    y = Series([1, 0, 1, 0, 1, 0, 1])
    result = series_target_rate(x, y)
    expected = {"a": 1.0, "b": 0.0, "c": 2 / 3}
    assert result == expected

    # Test series_target_rate with NaN values in x
    x = Series(["a", "b", "a", "b", "c", "c", "c", None])
    y = Series([1, 0, 1, 0, 1, 0, 1, 1])
    result = series_target_rate(x, y)
    expected = {"a": 1.0, "b": 0.0, "c": 2 / 3}
    assert result == expected

    # Test series_target_rate when all targets are the same
    x = Series(["a", "b", "a", "b", "c", "c", "c"])
    y = Series([1, 1, 1, 1, 1, 1, 1])
    result = series_target_rate(x, y)
    expected = {"a": 1.0, "b": 1.0, "c": 1.0}
    assert result == expected

    # Test series_target_rate with empty series
    x = Series([])
    y = Series([])
    result = series_target_rate(x, y)
    expected = {}
    assert result == expected


def test_categoricaldiscretizer_initialization():
    """Tests CategoricalDiscretizer initialization"""
    features = [CategoricalFeature("feature1"), CategoricalFeature("feature2")]
    categorical_discretizer = CategoricalDiscretizer(features, min_freq=0.02)
    assert isinstance(categorical_discretizer.features, Features)
    assert categorical_discretizer.min_freq == 0.02


def test_prepare_data():
    """Tests CategoricalDiscretizer _prepare_data method"""

    feature1 = CategoricalFeature("feature1")
    feature2 = CategoricalFeature("feature2")
    features = [feature1, feature2]
    categorical_discretizer = CategoricalDiscretizer(features, min_freq=0.02)

    X = DataFrame({"feature1": ["a", "b", "a", "b", nan], "feature2": ["x", "y", "x", "y", nan]})

    sample = categorical_discretizer._prepare_data(Sample(X, None))
    X_prepared = sample.X

    assert X_prepared["feature1"].tolist() == ["a", "b", "a", "b", nan]
    assert X_prepared["feature2"].tolist() == ["x", "y", "x", "y", nan]


def test_group_feature_rare_modalities():
    """Tests CategoricalDiscretizer _group_feature_rare_modalities method"""

    # test with min_freq = 0.2
    feature1 = CategoricalFeature("feature1")
    feature2 = CategoricalFeature("feature2")
    categorical_discretizer = CategoricalDiscretizer([feature1, feature2], min_freq=0.2)

    X = DataFrame(
        {
            "feature1": ["x", "b", "a", "b", "c", "c", "c"],
            "feature2": ["a", "y", "x", "y", "z", "z", "z"],
        }
    )
    categorical_discretizer.features.fit(X)
    frequencies = X[categorical_discretizer.features.versions].apply(series_value_counts, axis=0)
    grouped_x = categorical_discretizer._group_feature_rare_modalities(feature1, X, frequencies)

    assert feature1.default in grouped_x["feature1"].values
    assert feature1.has_default
    assert grouped_x["feature1"].tolist() == [
        feature1.default,
        "b",
        feature1.default,
        "b",
        "c",
        "c",
        "c",
    ]
    assert feature1.content[feature1.default] == ["a", "x", feature1.default]

    # test with min_freq = 0.001
    feature1 = CategoricalFeature("feature1")
    feature2 = CategoricalFeature("feature2")
    categorical_discretizer = CategoricalDiscretizer([feature1, feature2], min_freq=0.001)

    X = DataFrame(
        {
            "feature1": ["x", "b", "a", "b", "c", "c", "c"],
            "feature2": ["a", "y", "x", "y", "z", "z", "z"],
        }
    )
    categorical_discretizer.features.fit(X)
    frequencies = X[categorical_discretizer.features.versions].apply(series_value_counts, axis=0)
    grouped_x = categorical_discretizer._group_feature_rare_modalities(feature1, X, frequencies)

    assert feature1.default not in grouped_x["feature1"].values
    assert not feature1.has_default
    assert grouped_x["feature1"].tolist() == ["x", "b", "a", "b", "c", "c", "c"]
    assert feature1.values == ["x", "b", "a", "c"]

    # test with min_freq = 1.0
    feature1 = CategoricalFeature("feature1")
    feature2 = CategoricalFeature("feature2")
    categorical_discretizer = CategoricalDiscretizer([feature1, feature2], min_freq=1.0)

    X = DataFrame(
        {
            "feature1": ["x", "b", "a", "b", "c", "c", "c"],
            "feature2": ["a", "y", "x", "y", "z", "z", "z"],
        }
    )
    categorical_discretizer.features.fit(X)
    frequencies = X[categorical_discretizer.features.versions].apply(series_value_counts, axis=0)
    grouped_x = categorical_discretizer._group_feature_rare_modalities(feature1, X, frequencies)

    assert isinstance(grouped_x, DataFrame)
    assert feature1.default in grouped_x["feature1"].values
    assert feature1.has_default
    assert grouped_x["feature1"].tolist() == [
        feature1.default,
        feature1.default,
        feature1.default,
        feature1.default,
        feature1.default,
        feature1.default,
        feature1.default,
    ]
    assert feature1.content[feature1.default] == ["a", "x", "b", "c", feature1.default]


def test_group_rare_modalities():
    """Tests CategoricalDiscretizer _group_rare_modalities method"""
    feature1 = CategoricalFeature("feature1")
    feature2 = CategoricalFeature("feature2")
    categorical_discretizer = CategoricalDiscretizer([feature1, feature2], min_freq=0.2)

    X = DataFrame(
        {
            "feature1": ["x", "b", "a", "b", "c", "c", "c"],
            "feature2": ["a", "y", "x", "y", "z", "z", "z"],
        }
    )
    categorical_discretizer.features.fit(X)
    grouped_x = categorical_discretizer._group_rare_modalities(X)
    assert isinstance(grouped_x, DataFrame)
    assert feature1.has_default
    assert feature2.has_default


def test_target_sort():
    """Tests CategoricalDiscretizer _target_sort method"""
    feature1 = CategoricalFeature("feature1")
    feature2 = CategoricalFeature("feature2")
    categorical_discretizer = CategoricalDiscretizer([feature1, feature2], min_freq=0.2)

    X = DataFrame({"feature1": ["a", "b", "a", "b"], "feature2": ["x", "y", "x", "y"]})
    y = Series([1, 0, 1, 0])

    categorical_discretizer.features.fit(X)
    categorical_discretizer._target_sort(X, y)

    assert feature1.values == ["b", "a"]
    assert feature2.values == ["y", "x"]


def test_categoricaldiscretizer_fit():
    """Tests CategoricalDiscretizer fit method"""

    # test with binary target
    feature1 = CategoricalFeature("feature1")
    feature2 = CategoricalFeature("feature2")
    categorical_discretizer = CategoricalDiscretizer([feature1, feature2], min_freq=0.2)

    X = DataFrame(
        {
            "feature1": ["x", "b", "a", "b", "c", "c", "c", nan],
            "feature2": ["a", "y", "x", "y", "z", "z", "z", nan],
        }
    )
    y = Series([1, 0, 1, 0, 1, 1, 1, 1])

    categorical_discretizer.fit(X, y)

    assert feature1.has_default
    assert feature2.has_default
    assert feature1.values == ["b", feature1.default, "c"]
    assert feature2.values == ["y", feature2.default, "z"]

    transformed_x = categorical_discretizer.transform(X)

    assert transformed_x["feature1"].tolist() == [
        feature1.label_per_value[feature1.default],
        "b",
        feature1.label_per_value[feature1.default],
        "b",
        "c",
        "c",
        "c",
        nan,
    ]
    assert transformed_x["feature2"].tolist() == [
        feature2.label_per_value[feature2.default],
        "y",
        feature2.label_per_value[feature2.default],
        "y",
        "z",
        "z",
        "z",
        nan,
    ]

    # test with continuous target
    feature1 = CategoricalFeature("feature1")
    feature2 = CategoricalFeature("feature2")
    categorical_discretizer = CategoricalDiscretizer([feature1, feature2], min_freq=0.2)

    X = DataFrame(
        {
            "feature1": ["x", "b", "a", "b", "c", "c", "c", nan],
            "feature2": ["a", "y", "x", "y", "z", "z", "z", nan],
        }
    )
    y = Series([1.2, 0.1, 0.9, -0.2, 1, 1.5, 1.35, 1])

    categorical_discretizer.fit(X, y)

    assert feature1.has_default
    assert feature2.has_default
    assert feature1.values == ["b", feature1.default, "c"]
    assert feature2.values == ["y", feature2.default, "z"]

    transformed_x = categorical_discretizer.transform(X)

    assert transformed_x["feature1"].tolist() == [
        feature1.label_per_value[feature1.default],
        "b",
        feature1.label_per_value[feature1.default],
        "b",
        "c",
        "c",
        "c",
        nan,
    ]
    assert transformed_x["feature2"].tolist() == [
        feature2.label_per_value[feature2.default],
        "y",
        feature2.label_per_value[feature2.default],
        "y",
        "z",
        "z",
        "z",
        nan,
    ]


def test_categorical_discretizer(x_train: DataFrame, target: str) -> None:
    """Tests CategoricalDiscretizer

    Parameters
    ----------
    x_train : DataFrame
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
    features = Features(categoricals=categoricals, default=str_default)
    features.update(values_orders, replace=True)

    min_freq = 0.02
    # unwanted value in values_orders
    with raises(ValueError):
        discretizer = CategoricalDiscretizer(categoricals=features, min_freq=min_freq, copy=True)
        _ = discretizer.fit_transform(x_train, x_train[target])

    # correct feature ordering
    features = Features(categoricals=categoricals + ["Qualitative"], default=str_default)

    discretizer = CategoricalDiscretizer(categoricals=features, min_freq=min_freq, copy=True)
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
        "If any, NaN values should be put into str_nan and kept by themselves"
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
        "If any, NaN values should be put into str_nan and kept by themselves"
    )
