"""Set of tests for binary_carver module."""

from pathlib import Path

import pandas as pd
from pytest import FixtureRequest, fixture, raises

from AutoCarver import BinaryCarver
from AutoCarver.carvers.binary_carver import get_crosstab
from AutoCarver.carvers.utils.base_carver import Sample, Samples
from AutoCarver.combinations import (
    CombinationEvaluator,
    CramervCombinations,
    KruskalCombinations,
    TschuprowtCombinations,
)
from AutoCarver.config import Constants
from AutoCarver.discretizers import DiscretizerConfig
from AutoCarver.features import DatetimeFeature, Features, OrdinalFeature


@fixture(params=[CramervCombinations, TschuprowtCombinations])
def evaluator(request: FixtureRequest) -> CombinationEvaluator:
    """Evaluator instance fixture, passed as combination_evaluator= to the carver."""
    return request.param()


@fixture(scope="module", params=["tschuprowt", "cramerv"])
def sort_by(request) -> str:
    """sorting measure"""
    return request.param


def test_get_crosstab_basic():
    """Test get_crosstab with basic data."""
    X = pd.DataFrame({"feature": ["A", "B", "A", "B", "C"]})
    y = pd.Series([1, 0, 1, 0, 1])
    feature = OrdinalFeature("feature", ["A", "B", "C"])
    result = get_crosstab(X, y, feature)
    expected = pd.DataFrame({0: [0, 2, 0], 1: [2, 0, 1]}, index=["A", "B", "C"])
    assert result.equals(expected)


def test_get_crosstab_with_nan():
    """Test get_crosstab with missing values."""
    X = pd.DataFrame({"feature": ["A", "B", "A", "B", None]})
    y = pd.Series([1, 0, 1, 0, 1])
    feature = OrdinalFeature("feature", ["A", "B", "C"])
    result = get_crosstab(X, y, feature)
    expected = pd.DataFrame({0: [0, 2, 0], 1: [2, 0, 0]}, index=["A", "B", "C"])
    assert result.equals(expected)


def test_get_crosstab_unordered_labels():
    """Test get_crosstab with unordered labels."""
    X = pd.DataFrame({"feature": ["A", "B", "A", "B", "C"]})
    y = pd.Series([1, 0, 1, 0, 1])
    feature = OrdinalFeature("feature", ["C", "A", "B"])
    result = get_crosstab(X, y, feature)
    expected = pd.DataFrame({0: [0, 0, 2], 1: [1, 2, 0]}, index=["C", "A", "B"])
    assert result.equals(expected)


def test_get_crosstab_missing_labels():
    """Test get_crosstab with missing labels."""
    X = pd.DataFrame({"feature": ["A", "B", "A", "B", "C"]})
    y = pd.Series([1, 0, 1, 0, 1])
    feature = OrdinalFeature("feature", ["A", "B"])
    result = get_crosstab(X, y, feature)
    expected = pd.DataFrame({0: [0, 2], 1: [2, 0]}, index=["A", "B"])
    assert result.equals(expected)


def test_get_crosstab_extra_labels():
    """Test get_crosstab with extra labels."""
    X = pd.DataFrame({"feature": ["A", "B", "A", "B", "C"]})
    y = pd.Series([1, 0, 1, 0, 1])
    feature = OrdinalFeature("feature", ["A", "B", "C", "D"])
    result = get_crosstab(X, y, feature)
    expected = pd.DataFrame({0: [0, 2, 0, 0], 1: [2, 0, 1, 0]}, index=["A", "B", "C", "D"])
    assert result.equals(expected)


def test_binary_carver_initialization():
    """Test BinaryCarver initialization."""
    features = Features(
        categoricals=["feature1"],
        ordinals={"feature2": ["low", "medium", "high"]},
        quantitatives=["feature3"],
    )
    carver = BinaryCarver(min_freq=0.1, max_n_mod=5, features=features)
    assert carver.min_freq == 0.1
    # half_min_freq lives on the carver — it owns the halving that's applied to
    # min_freq before discretizers are invoked; discretizers themselves use min_freq
    # directly (with a 1-row tolerance).
    assert carver.half_min_freq == 0.05
    assert carver.features == features
    assert carver.config.dropna is True
    assert isinstance(carver.combination_evaluator, TschuprowtCombinations)
    assert carver.max_n_mod == 5

    max_n_mod = 8
    carver = BinaryCarver(
        min_freq=0.1,
        features=features,
        max_n_mod=max_n_mod,
        combination_evaluator=TschuprowtCombinations(),
    )
    assert isinstance(carver.combination_evaluator, TschuprowtCombinations)
    assert carver.max_n_mod == max_n_mod

    carver = BinaryCarver(
        min_freq=0.1,
        features=features,
        max_n_mod=max_n_mod,
        combination_evaluator=CramervCombinations(),
    )
    assert isinstance(carver.combination_evaluator, CramervCombinations)
    assert carver.max_n_mod == max_n_mod

    with raises(ValueError):
        BinaryCarver(
            min_freq=0.1,
            features=features,
            max_n_mod=max_n_mod,
            combination_evaluator=KruskalCombinations(),
        )


def test_binary_carver_prepare_samples(evaluator: CombinationEvaluator):
    """Test BinaryCarver _prepare_samples method."""
    features = Features(
        categoricals=["feature1"],
        ordinals={"feature2": ["low", "medium", "high"]},
        quantitatives=["feature3"],
    )
    carver = BinaryCarver(min_freq=0.1, max_n_mod=5, features=features, combination_evaluator=evaluator)
    X = pd.DataFrame({"feature1": ["A", "B", "A"], "feature2": ["low", "medium", "high"], "feature3": [1, 2, 3]})

    # with wrong target
    y = pd.Series([0, 1, 2])
    samples = Samples(train=Sample(X, y))

    with raises(ValueError):
        carver._prepare_samples(samples)

    # with right target
    y = pd.Series([0, 1, 0])
    samples = Samples(train=Sample(X, y))

    prepared_samples = carver._prepare_samples(samples)
    assert isinstance(prepared_samples, Samples)


def test_binary_carver_aggregator(evaluator: CombinationEvaluator):
    """Test BinaryCarver _aggregator method."""
    features = Features(
        categoricals=["feature1"],
        ordinals={"feature2": ["low", "medium", "high"]},
        quantitatives=["feature3"],
    )
    carver = BinaryCarver(min_freq=0.1, max_n_mod=5, features=features, combination_evaluator=evaluator)
    X = pd.DataFrame({"feature1": ["A", "B", "A"], "feature2": ["low", "medium", "high"], "feature3": [1, 2, 3]})
    y = pd.Series([0, 1, 0])
    xtabs = carver._aggregator(X, y)
    print(xtabs)

    expected = {
        "feature1": pd.DataFrame({0: [2, 0], 1: [0, 1]}, index=["A", "B"]),
        "feature2": pd.DataFrame({0: [1, 0, 1], 1: [0, 1, 0]}, index=["low", "medium", "high"]),
        "feature3": pd.DataFrame({0: [1, 0, 1], 1: [0, 1, 0]}, index=[1, 2, 3]),
    }
    assert isinstance(xtabs, dict)
    assert "feature1" in xtabs
    assert "feature2" in xtabs
    assert "feature3" in xtabs
    for feature in features.names:
        assert all(xtabs[feature].index == expected[feature].index)
        assert (xtabs[feature].values == expected[feature].values).all()


def test_carve_feature_with_best_combination(evaluator):
    """Test BinaryCarver _carve_feature method."""

    features = Features(
        categoricals=["feature1"],
        ordinals={"feature2": ["low", "medium", "high"]},
        quantitatives=["feature3"],
    )
    X = pd.DataFrame(
        {
            "feature1": ["A", "B", "A", "C"],
            "feature2": ["low", "medium", "high", "high"],
            "feature3": [1, 2, 3, float("nan")],
        }
    )
    y = pd.Series([0, 1, 0, 1])
    samples = Samples(train=Sample(X, y))

    # initializing carver
    carver = BinaryCarver(
        features=features,
        min_freq=0.1,
        max_n_mod=5,
        combination_evaluator=evaluator,
        config=DiscretizerConfig(dropna=True, verbose=False),
    )
    carver._prepare_samples(samples)

    # getting aggregated data
    xaggs = carver._aggregator(**samples.train)
    xaggs_dev = carver._aggregator(**samples.dev)

    # carving a feature
    feature = features[0]
    carver._carve_feature(feature, xaggs, xaggs_dev, "1/1")
    print(feature.content)
    assert feature in carver.features
    assert feature.content == {"A": ["A"], "B": ["C", "B"]}

    # carving a feature
    feature = features[1]
    carver._carve_feature(feature, xaggs, xaggs_dev, "1/1")
    print(feature.content)
    assert feature in carver.features
    assert feature.content == {"low": ["low"], "medium": ["medium"], "high": ["high"]}

    # carving a feature
    feature = features[2]
    carver._carve_feature(feature, xaggs, xaggs_dev, "1/1")
    print(feature.content)
    assert feature in carver.features
    assert feature.content == {
        1.0: [1.0],
        2.0: [2.0],
        float("inf"): [3.0, float("inf")],
        "__NAN__": ["__NAN__"],
    }


def test_carve_feature_without_best_combination(evaluator: CombinationEvaluator):
    """Test _carve_feature method without best combination."""

    features = Features(
        categoricals=["feature1"],
        ordinals={"feature2": ["low", "medium", "high"]},
        quantitatives=["feature3"],
    )
    # n=12 so each singleton (1/12) falls below the carver-halved rare floor (0.45 − 1/12).
    X = pd.DataFrame(
        {
            "feature1": ["A", "A", "A", "A", "A", "A", "B", "C", "D", "E", "F", "G"],
            "feature2": [
                "low",
                "low",
                "low",
                "low",
                "low",
                "low",
                "medium",
                "medium",
                "high",
                "high",
                "high",
                "high",
            ],
            "feature3": [1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, float("nan")],
        }
    )
    y = pd.Series([0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1])
    samples = Samples(train=Sample(X, y))

    # initializing carver
    carver = BinaryCarver(
        features=features,
        min_freq=0.9,
        max_n_mod=5,
        combination_evaluator=evaluator,
        config=DiscretizerConfig(dropna=True, verbose=False),
    )
    carver._prepare_samples(samples)

    # getting aggregated data
    xaggs = carver._aggregator(**samples.train)
    xaggs_dev = carver._aggregator(**samples.dev)

    # carving a feature
    feature = features[0]
    carver._carve_feature(feature, xaggs, xaggs_dev, "1/1")
    print(feature.content)
    assert feature not in carver.features
    # main intent: rare modalities collapsed into __OTHER__ and feature got dropped.
    assert feature.has_default
    assert feature.content["A"] == ["A"]


def test_fit_with_best_combination(evaluator):
    """Test BinaryCarver fit method with best combination."""

    features = Features(
        categoricals=["feature1"],
        ordinals={"feature2": ["low", "medium", "high"]},
        quantitatives=["feature3"],
    )
    X = pd.DataFrame(
        {
            "feature1": ["A", "B", "A", "C"],
            "feature2": ["low", "medium", "high", "high"],
            "feature3": [1, 2, 3, float("nan")],
        }
    )
    y = pd.Series([0, 1, 0, 1])

    # initializing carver
    carver = BinaryCarver(
        features=features,
        min_freq=0.1,
        max_n_mod=5,
        combination_evaluator=evaluator,
        config=DiscretizerConfig(dropna=True, verbose=False),
    )

    # fitting carver
    carver.fit(X, y)

    feature = features[0]
    print(feature.content)
    assert feature in carver.features
    assert feature.content == {"A": ["A"], "B": ["C", "B"]}

    # carving a feature
    feature = features[1]
    print(feature.content)
    assert feature in carver.features
    assert feature.content == {"low": ["low"], "medium": ["medium"], "high": ["high"]}

    # carving a feature
    feature = features[2]
    print(feature.content)
    assert feature in carver.features
    assert feature.content == {
        1.0: [1.0],
        2.0: [2.0],
        float("inf"): [3.0, float("inf")],
        "__NAN__": ["__NAN__"],
    }


def test_fit_without_best_combination(evaluator: CombinationEvaluator):
    """Test BinaryCarver fit method without best combination."""

    features = Features(
        categoricals=["feature1"],
        ordinals={"feature2": ["low", "medium", "high"]},
        quantitatives=["feature3"],
    )
    X = pd.DataFrame(
        {
            "feature1": ["A", "B", "A", "C"],
            "feature2": ["low", "medium", "high", "high"],
            "feature3": [1, 2, 3, float("nan")],
        }
    )
    y = pd.Series([0, 1, 0, 1])

    # initializing carver
    carver = BinaryCarver(
        features=features,
        min_freq=0.9,
        max_n_mod=5,
        combination_evaluator=evaluator,
        config=DiscretizerConfig(dropna=True, verbose=False),
    )

    # fitting carver
    carver.fit(X, y)

    # carving a feature
    assert len(features) == 0


def test_binary_carver_fit_transform_with_small_data_not_ordinal(evaluator: CombinationEvaluator):
    """Test BinaryCarver fit_transform method."""
    features = Features(
        categoricals=["feature1"],
        ordinals={"feature2": ["low", "medium", "high"]},
        quantitatives=["feature3"],
    )
    carver = BinaryCarver(
        min_freq=0.1,
        max_n_mod=5,
        features=features,
        combination_evaluator=evaluator,
        config=DiscretizerConfig(dropna=True, ordinal_encoding=False, copy=False),
    )
    idx = ["a", "b", "c", "d"]
    X = pd.DataFrame(
        {
            "feature1": ["A", "B", "A", "C"],
            "feature2": ["low", "medium", "high", "high"],
            "feature3": [1, 2, 3, float("nan")],
        },
        index=idx,
    )
    y = pd.Series([0, 1, 0, 1], index=idx)
    X_transformed = carver.fit_transform(X, y)

    print(carver.features("feature1").content)
    print(X_transformed)
    # Under Wilson-CI gating the borderline ``2.0`` bin survives on this n=4 sample,
    # so feature3 keeps three numeric buckets (1, 2, >2) plus NaN.
    expected = pd.DataFrame(
        {
            "feature1": ["A", "B, C", "A", "B, C"],
            "feature2": ["low", "medium", "high", "high"],
            "feature3": [
                "x <= 1.00e+00",
                "1.00e+00 < x <= 2.00e+00",
                "2.00e+00 < x",
                "__NAN__",
            ],
        },
        index=idx,
    )
    assert isinstance(X_transformed, pd.DataFrame)
    assert all(X_transformed.index == expected.index)
    assert all(X_transformed.index == X.index)
    assert all(X_transformed.columns == expected.columns)
    assert all(X.columns == expected.columns)
    print(
        "X values",
        X.values,
        "\n\nX transfor",
        X_transformed.values,
        "\n",
        (X.values == expected.values),
        "\n",
        (X_transformed.values == expected.values),
    )
    assert X.equals(expected)
    assert X_transformed.equals(expected)


def test_binary_carver_fit_transform_with_small_data_ordinal(evaluator: CombinationEvaluator):
    """Test BinaryCarver fit_transform method."""
    features = Features(
        categoricals=["feature1"],
        ordinals={"feature2": ["low", "medium", "high"]},
        quantitatives=["feature3"],
    )
    carver = BinaryCarver(
        min_freq=0.1,
        max_n_mod=5,
        features=features,
        combination_evaluator=evaluator,
        config=DiscretizerConfig(dropna=True, ordinal_encoding=True, copy=False),
    )
    idx = ["a", "b", "c", "d"]
    X = pd.DataFrame(
        {
            "feature1": ["A", "B", "A", "C"],
            "feature2": ["low", "medium", "high", "high"],
            "feature3": [1, 2, 3, float("nan")],
        },
        index=idx,
    )
    y = pd.Series([0, 1, 0, 1], index=idx)
    X_transformed = carver.fit_transform(X, y)

    print(carver.features("feature1").content)
    print(X_transformed)
    expected = pd.DataFrame(
        {
            "feature1": [0, 1, 0, 1],
            "feature2": [0, 1, 2, 2],
            # Under Wilson-CI gating the borderline ``2.0`` bin survives, so feature3
            # ends up with 4 buckets (1.0, 2.0, 3.0+, nan) instead of 3.
            "feature3": [0, 1, 2, 3],
        },
        index=idx,
    )
    assert isinstance(X_transformed, pd.DataFrame)
    assert all(X_transformed.index == expected.index)
    assert all(X_transformed.index == X.index)
    assert all(X_transformed.columns == expected.columns)
    assert all(X.columns == expected.columns)
    print(
        "X values",
        X.values,
        "\n\nX transfor",
        X_transformed.values,
        "\n",
        (X.values == expected.values),
        "\n",
        (X_transformed.values == expected.values),
    )
    pd.testing.assert_frame_equal(X, expected, check_dtype=False)
    pd.testing.assert_frame_equal(X_transformed, expected, check_dtype=False)


def test_binary_carver_fit_transform_with_large_data(evaluator: CombinationEvaluator):
    """Test BinaryCarver fit_transform method."""
    features = Features(
        categoricals=["feature1"],
        ordinals={"feature2": ["low", "medium", "high"]},
        quantitatives=["feature3"],
    )
    carver = BinaryCarver(
        min_freq=0.1,
        max_n_mod=5,
        features=features,
        combination_evaluator=evaluator,
        config=DiscretizerConfig(dropna=True, ordinal_encoding=False, copy=False),
    )
    idx = [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
    ]  # , "r", "s", "t", "u", "v"]
    X = pd.DataFrame(
        {
            "feature1": [
                "A",
                "B",
                "A",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
            ],
            "feature2": [
                "low",
                "medium",
                "high",
                "high",
                "low",
                "low",
                "low",
                "low",
                "low",
                "low",
                "high",
                "low",
                "low",
                "low",
                "low",
                "low",
                "low",
            ],
            "feature3": [1, 2, 3, float("nan"), 3, 1, 2, 3, 1, 2, float("nan"), 3, 1, 2, 3, 1, 2],
        },
        index=idx,
    )
    y = pd.Series([0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1], index=idx)
    X_transformed = carver.fit_transform(X, y)

    print(carver.features("feature1").content)
    print(X_transformed)
    # Under Wilson-CI gating the categorical "B" and ordinal "medium"/"high" bins
    # are no longer significantly under the min_freq floor at n=17, so the carver
    # leaves them as their own labels instead of merging into "B, C" / "medium to high".
    expected = pd.DataFrame(
        {
            "feature1": [
                "A",
                "B",
                "A",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
            ],
            "feature2": [
                "low",
                "medium",
                "high",
                "high",
                "low",
                "low",
                "low",
                "low",
                "low",
                "low",
                "high",
                "low",
                "low",
                "low",
                "low",
                "low",
                "low",
            ],
            "feature3": [
                "x <= 1.00e+00",
                "1.00e+00 < x <= 2.00e+00",
                "2.00e+00 < x",
                "1.00e+00 < x <= 2.00e+00",
                "2.00e+00 < x",
                "x <= 1.00e+00",
                "1.00e+00 < x <= 2.00e+00",
                "2.00e+00 < x",
                "x <= 1.00e+00",
                "1.00e+00 < x <= 2.00e+00",
                "1.00e+00 < x <= 2.00e+00",
                "2.00e+00 < x",
                "x <= 1.00e+00",
                "1.00e+00 < x <= 2.00e+00",
                "2.00e+00 < x",
                "x <= 1.00e+00",
                "1.00e+00 < x <= 2.00e+00",
            ],
        },
        index=idx,
    )
    assert isinstance(X_transformed, pd.DataFrame)
    assert all(X_transformed.index == expected.index)
    assert all(X_transformed.index == X.index)
    assert all(X_transformed.columns == expected.columns)
    assert all(X.columns == expected.columns)
    print(
        "X values",
        X.values,
        "\n\nX transfor",
        X_transformed.values,
        "\n",
        (X.values == expected.values),
    )
    assert X.equals(expected)
    assert X_transformed.equals(expected)


def test_binary_carver_fit_transform_with_target_only_nan(evaluator: CombinationEvaluator):
    """Test BinaryCarver fit_transform method."""
    features = Features(
        categoricals=["feature1"],
        ordinals={"feature2": ["low", "medium", "high"]},
        quantitatives=["feature3"],
    )
    carver = BinaryCarver(
        min_freq=0.1,
        max_n_mod=5,
        features=features,
        combination_evaluator=evaluator,
        config=DiscretizerConfig(dropna=True, ordinal_encoding=False, copy=False),
    )
    idx = ["a", "b", "c", "d"]
    X = pd.DataFrame(
        {
            "feature1": ["A", "B", "A", "C"],
            "feature2": ["low", "medium", "high", "high"],
            "feature3": [1, 2, 3, float("nan")],
        },
        index=idx,
    )
    y = pd.Series([0, 0, 0, 1], index=idx)
    X_transformed = carver.fit_transform(X, y)

    print(carver.features("feature1").content)
    print(X_transformed)
    # under Wilson-CI gating the borderline ``2.0`` numeric bin survives.
    expected = pd.DataFrame(
        {
            "feature1": ["A, B", "A, B", "A, B", "C"],
            "feature2": ["low", "medium to high", "medium to high", "medium to high"],
            "feature3": [
                "x <= 1.00e+00",
                "1.00e+00 < x <= 3.00e+00",
                "1.00e+00 < x <= 3.00e+00",
                "3.00e+00 < x",
            ],
        },
        index=idx,
    )
    assert isinstance(X_transformed, pd.DataFrame)
    assert all(X_transformed.index == expected.index)
    assert all(X_transformed.index == X.index)
    assert all(X_transformed.columns == expected.columns)
    assert all(X.columns == expected.columns)
    print(
        X.values,
        "\n",
        expected.values,
        "\n",
        X_transformed.values,
        "\n",
        (X.values == expected.values),
    )
    assert X.equals(expected)
    assert X_transformed.equals(expected)


def test_binary_carver_fit_transform_with_wrong_dev(evaluator: CombinationEvaluator):
    """Test BinaryCarver fit_transform method."""
    features = Features(
        categoricals=["feature1"],
        ordinals={"feature2": ["low", "medium", "high"]},
        quantitatives=["feature3"],
    )
    carver = BinaryCarver(
        min_freq=0.1,
        max_n_mod=5,
        features=features,
        combination_evaluator=evaluator,
        config=DiscretizerConfig(dropna=True, ordinal_encoding=True, copy=False),
    )
    idx = ["a", "b", "c", "d"]
    X = pd.DataFrame(
        {
            "feature1": ["A", "B", "A", "C"],
            "feature2": ["low", "medium", "high", "high"],
            "feature3": [1, 2, 3, float("nan")],
        },
        index=idx,
    )
    y = pd.Series([0, 1, 0, 1], index=idx)
    X_dev = pd.DataFrame(
        {
            "feature1": ["A", "B", "A", "C"],
            "feature2": ["low", "medium", "high", "high"],
            "feature3": [1, 2, 3, float("nan")],
        },
        index=idx,
    )
    y_dev = pd.Series([1, 0, 1, 0], index=idx)
    X_transformed = carver.fit_transform(X, y, X_dev=X_dev, y_dev=y_dev)

    # rank-inversion between train and dev → categorical/ordinal features get
    # dropped. The quantitative feature can still find a single-bucket viable
    # carving (trivially distinct rates pass when there is only one modality)
    # so its column survives in encoded form. The categorical/ordinal features
    # are dropped from the carver and from the transformed DataFrame's logic
    # (raw values left in place).
    assert isinstance(X_transformed, pd.DataFrame)
    assert all(X_transformed.index == X.index)
    assert "feature1" not in {f.name for f in carver.features}
    assert "feature2" not in {f.name for f in carver.features}


def test_binary_carver_save_load(tmp_path: Path, evaluator: CombinationEvaluator):
    """Test BinaryCarver save and load methods."""
    features = Features(
        categoricals=["feature1"],
        ordinals={"feature2": ["low", "medium", "high"]},
        quantitatives=["feature3"],
    )
    carver = BinaryCarver(min_freq=0.1, max_n_mod=5, features=features, combination_evaluator=evaluator)
    carver_file = tmp_path / "binary_carver.json"
    carver.save(str(carver_file))
    loaded_carver = BinaryCarver.load(str(carver_file))
    assert carver.min_freq == loaded_carver.min_freq
    for feature in carver.features:
        assert feature in loaded_carver.features
        assert feature.content == loaded_carver.features[feature.name].content
    assert carver.config.dropna == loaded_carver.config.dropna
    assert carver.config.verbose == loaded_carver.config.verbose
    assert carver.config.copy == loaded_carver.config.copy
    assert carver.combination_evaluator.__class__ == loaded_carver.combination_evaluator.__class__
    assert carver.max_n_mod == loaded_carver.max_n_mod
    assert carver.combination_evaluator.sort_by == loaded_carver.combination_evaluator.sort_by
    assert carver.combination_evaluator.verbose == loaded_carver.combination_evaluator.verbose


def test_binary_carver_end_to_end_with_datetime(tmp_path: Path, evaluator: CombinationEvaluator):
    """End-to-end BinaryCarver on a datetime feature: fit, transform, save/load, re-transform."""
    n = 60
    idx = list(range(n))
    dates = pd.date_range("2020-01-01", periods=n, freq="D").tolist()
    dates[7] = pd.NaT  # a missing datetime
    X = pd.DataFrame({"signup": dates, "cat": ["A", "B"] * (n // 2)}, index=idx)

    # target correlated with how late the signup date is (earlier -> 0, later -> 1)
    y = pd.Series([0 if i < n // 2 else 1 for i in range(n)], index=idx)

    features = Features(categoricals=["cat"], datetimes=[("signup", "2020-01-01")])
    carver = BinaryCarver(
        min_freq=0.2,
        max_n_mod=4,
        features=features,
        combination_evaluator=evaluator,
        config=DiscretizerConfig(dropna=True, copy=True),
    )
    X_transformed = carver.fit_transform(X, y)

    # the datetime feature is recognized as such, fitted, and carved into buckets
    signup = carver.features("signup")
    assert isinstance(signup, DatetimeFeature)
    assert signup.reference_date == "2020-01-01"
    assert signup.is_fitted and signup.has_nan
    assert len(X_transformed) == n

    # transformed datetime column only contains learned bucket labels
    assert set(X_transformed["signup"].dropna().unique()).issubset(set(signup.labels))

    # save / load preserves the DatetimeFeature (type + reference_date + carved buckets)
    carver_file = tmp_path / "binary_carver_datetime.json"
    carver.save(str(carver_file))
    loaded = BinaryCarver.load(str(carver_file))
    loaded_signup = loaded.features("signup")
    assert isinstance(loaded_signup, DatetimeFeature)
    assert loaded_signup.reference_date == "2020-01-01"
    assert loaded_signup.content == signup.content

    # the loaded carver transforms fresh raw datetimes identically
    assert loaded.transform(X).equals(carver.transform(X))


def _fit_binary_carver(
    x_train: pd.DataFrame,
    x_dev_1: pd.DataFrame,
    qualitative_features: list[str],
    quantitative_features: list[str],
    values_orders: dict[str, list[str]],
    chained_features: list[str],
    level0_to_level1: dict[str, list[str]],
    level1_to_level2: dict[str, list[str]],
    evaluator: CombinationEvaluator,
    *,
    ordinal_encoding: bool = False,
    dropna: bool = True,
    copy: bool = True,
    min_freq: float = 0.1,
) -> tuple[BinaryCarver, pd.DataFrame, pd.DataFrame, Features]:
    """Build features, fit BinaryCarver, transform train and dev.

    Common setup factored out of the original mega-test; each focused test below
    calls it with only the parameter combinations it actually exercises.
    """
    features = Features(
        categoricals=qualitative_features,
        ordinals=values_orders,
        quantitatives=quantitative_features,
    )
    for feature_name in ["nan", "ones", "ones_nan"]:
        features.remove(feature_name)

    # (chained_features / level0_to_level1 / level1_to_level2 are retained as parameters for
    # fixture compatibility; rare-modality rollup is now handled inside the carver pipeline.)
    auto_carver = BinaryCarver(
        min_freq=min_freq,
        max_n_mod=4,
        combination_evaluator=evaluator,
        features=features,
        config=DiscretizerConfig(dropna=dropna, ordinal_encoding=ordinal_encoding, copy=copy, verbose=False),
    )
    x_discretized = auto_carver.fit_transform(
        x_train,
        x_train["binary_target"],
        X_dev=x_dev_1,
        y_dev=x_dev_1["binary_target"],
    )
    x_dev_discretized = auto_carver.transform(x_dev_1)
    return auto_carver, x_discretized, x_dev_discretized, features


def test_binary_carver_uneligible_features_raises(
    x_train: pd.DataFrame,
    x_dev_1: pd.DataFrame,
    quantitative_features: list[str],
    qualitative_features: list[str],
    values_orders: dict[str, list[str]],
    evaluator: CombinationEvaluator,
) -> None:
    """ValueError when features contain uneligible columns ("nan"/"ones"/"ones_nan")."""
    features = Features(
        categoricals=qualitative_features,
        ordinals=values_orders,
        quantitatives=quantitative_features,
    )
    with raises(ValueError):
        auto_carver = BinaryCarver(
            min_freq=0.1,
            max_n_mod=4,
            combination_evaluator=evaluator,
            features=features,
            config=DiscretizerConfig(verbose=False),
        )
        auto_carver.fit_transform(
            x_train,
            x_train["binary_target"],
            X_dev=x_dev_1,
            y_dev=x_dev_1["binary_target"],
        )
        auto_carver.transform(x_dev_1)


def test_binary_carver_end_to_end_invariants(
    x_train: pd.DataFrame,
    x_dev_1: pd.DataFrame,
    quantitative_features: list[str],
    qualitative_features: list[str],
    values_orders: dict[str, list[str]],
    chained_features: list[str],
    level0_to_level1: dict[str, list[str]],
    level1_to_level2: dict[str, list[str]],
    evaluator: CombinationEvaluator,
    dropna: bool,
) -> None:
    """Modality counts, NaN handling, train/dev robustness, value preservation."""
    raw_x_train = x_train.copy()
    target = "binary_target"
    auto_carver, x_discretized, x_dev_discretized, features = _fit_binary_carver(
        x_train,
        x_dev_1,
        qualitative_features,
        quantitative_features,
        values_orders,
        chained_features,
        level0_to_level1,
        level1_to_level2,
        evaluator,
        dropna=dropna,
    )
    feature_names = features.names

    # max_n_mod respected on train + dev
    assert all(x_discretized[feature_names].nunique() <= auto_carver.max_n_mod), (
        "Too many buckets after carving of train sample"
    )
    assert all(x_dev_discretized[feature_names].nunique() <= auto_carver.max_n_mod), (
        "Too many buckets after carving of test sample"
    )

    # NaN handling matches dropna
    if not dropna:
        assert all(raw_x_train[feature_names].isna().mean() == x_discretized[feature_names].isna().mean()), (
            "Some Nans are being dropped (grouped) or more nans than expected"
        )
    else:
        assert all(x_discretized[feature_names].isna().mean() == 0), "Some Nans are not dropped (grouped)"

    # train/dev modality counts match
    assert all(x_discretized[feature_names].nunique() == x_dev_discretized[feature_names].nunique()), (
        "More buckets in train or test samples"
    )

    # robustness: same modalities show up in train and dev per feature
    for feature in feature_names:
        train_target_rate = x_discretized.groupby(feature)[target].mean().sort_values()
        dev_target_rate = x_dev_discretized.groupby(feature)[target].mean().sort_values()
        assert all(train_target_rate.index == dev_target_rate.index), (
            f"Not robust feature {feature} was not dropped, or robustness test not working"
        )

    # all initial qualitative values still present in carved values
    for feature in features.qualitatives:
        fitted_values = feature.values.values
        init_values = raw_x_train[feature.name].fillna(feature.nan).unique()
        if not dropna:
            init_values = [value for value in init_values if value != feature.nan]
        assert all(value in fitted_values for value in init_values), (
            f"Missing value in output! Some values have been dropped for qualitative feature: {feature.name}"
        )


def test_binary_carver_copy_semantics(
    x_train: pd.DataFrame,
    x_dev_1: pd.DataFrame,
    quantitative_features: list[str],
    qualitative_features: list[str],
    values_orders: dict[str, list[str]],
    chained_features: list[str],
    level0_to_level1: dict[str, list[str]],
    level1_to_level2: dict[str, list[str]],
    copy: bool,
) -> None:
    """copy=True returns a new DataFrame; copy=False mutates the input in place."""
    auto_carver, x_discretized, _, features = _fit_binary_carver(
        x_train,
        x_dev_1,
        qualitative_features,
        quantitative_features,
        values_orders,
        chained_features,
        level0_to_level1,
        level1_to_level2,
        CramervCombinations(),
        copy=copy,
    )
    feature_names = features.names
    if copy:
        assert any(
            x_discretized[feature_names].fillna(Constants.NAN) != x_train[feature_names].fillna(Constants.NAN)
        ), "Not applied discretization inplace correctly"
    else:
        assert all(
            x_discretized[feature_names].fillna(Constants.NAN) == x_train[feature_names].fillna(Constants.NAN)
        ), "Not copied correctly"


def test_binary_carver_serialization_roundtrip(
    tmp_path: Path,
    x_train: pd.DataFrame,
    x_dev_1: pd.DataFrame,
    quantitative_features: list[str],
    qualitative_features: list[str],
    values_orders: dict[str, list[str]],
    chained_features: list[str],
    level0_to_level1: dict[str, list[str]],
    level1_to_level2: dict[str, list[str]],
    evaluator: CombinationEvaluator,
) -> None:
    """Save/load on a fitted carver preserves summary and transform output."""
    raw_x_train = x_train.copy()
    auto_carver, _, x_dev_discretized, features = _fit_binary_carver(
        x_train,
        x_dev_1,
        qualitative_features,
        quantitative_features,
        values_orders,
        chained_features,
        level0_to_level1,
        level1_to_level2,
        evaluator,
    )
    feature_names = features.names

    carver_file = tmp_path / "test.json"
    auto_carver.save(str(carver_file))
    loaded_carver = BinaryCarver.load(str(carver_file))

    assert all(loaded_carver.summary == auto_carver.summary), "Non-identical summaries when loading from JSON"
    loaded_x_train = loaded_carver.transform(raw_x_train)
    assert all(x_dev_discretized[feature_names] == loaded_x_train[loaded_carver.features.names]), (
        "Non-identical discretized values when loading from JSON"
    )


def test_binary_carver_wrong_dev_transform(
    x_train: pd.DataFrame,
    x_dev_1: pd.DataFrame,
    x_dev_wrong_1: pd.DataFrame,
    x_dev_wrong_2: pd.DataFrame,
    x_dev_wrong_3: pd.DataFrame,
    quantitative_features: list[str],
    qualitative_features: list[str],
    values_orders: dict[str, list[str]],
    chained_features: list[str],
    level0_to_level1: dict[str, list[str]],
    level1_to_level2: dict[str, list[str]],
) -> None:
    """transform on wrong-dev variants raises (or not) per has_default semantics."""
    auto_carver, _, _, _ = _fit_binary_carver(
        x_train,
        x_dev_1,
        qualitative_features,
        quantitative_features,
        values_orders,
        chained_features,
        level0_to_level1,
        level1_to_level2,
        CramervCombinations(),
    )

    # unexpected modal on a feature that has_default — does not raise
    auto_carver.transform(x_dev_wrong_1)

    # unexpected nans (even though it has_default) — raises
    with raises(ValueError):
        auto_carver.transform(x_dev_wrong_2)

    # unexpected modal on a feature that does not have default — raises
    with raises(ValueError):
        auto_carver.transform(x_dev_wrong_3)


def test_binary_carver_unknown_ordinal_values_raises(
    x_train_wrong_2: pd.DataFrame,
    values_orders: dict[str, list[str]],
) -> None:
    """fit_transform with an ordinal feature containing unknown values raises ValueError."""
    values_orders.update(
        {
            "Qualitative_Ordinal_lownan": [
                "Low+",
                "Medium-",
                "Medium",
                "Medium+",
                "High-",
                "High",
                "High+",
            ],
            "Qualitative_Ordinal_highnan": [
                "Low-",
                "Low",
                "Low+",
                "Medium-",
                "Medium",
                "Medium+",
                "High-",
                "High",
                "High+",
            ],
            "Discrete_Qualitative_highnan": ["1", "2", "3", "4", "5", "6", "7"],
        }
    )

    features = Features(ordinals=values_orders)
    for feature_name in ["nan", "ones", "ones_nan"]:
        features.remove(feature_name)

    auto_carver = BinaryCarver(
        min_freq=0.15,
        max_n_mod=5,
        combination_evaluator=CramervCombinations(),
        features=features,
        config=DiscretizerConfig(verbose=False),
    )
    with raises(ValueError):
        auto_carver.fit_transform(x_train_wrong_2, x_train_wrong_2["binary_target"])
