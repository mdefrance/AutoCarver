"""Set of tests for continuous_carver module."""

from pathlib import Path

import numpy as np
import pandas as pd
from pytest import FixtureRequest, fixture, raises

from AutoCarver import ContinuousCarver
from AutoCarver.carvers.continuous_carver import get_target_values_by_modality
from AutoCarver.carvers.utils.base_carver import Sample, Samples
from AutoCarver.combinations import (
    CombinationEvaluator,
    CramervCombinations,
    KruskalCombinations,
    TschuprowtCombinations,
)
from AutoCarver.config import Constants
from AutoCarver.discretizers import ProcessingConfig
from AutoCarver.features import Features, OrdinalFeature


def test_get_target_values_by_modality_basic():
    X = pd.DataFrame({"feature": ["A", "B", "A", "B", "C"]})
    y = pd.Series([1, 2, 3, 4, 5])
    feature = OrdinalFeature("feature", ["A", "B", "C"])
    result = get_target_values_by_modality(X, y, feature)
    expected = pd.Series({"A": [1, 3], "B": [2, 4], "C": [5]})
    assert result.equals(expected)


def test_get_target_values_by_modality_with_nan():
    X = pd.DataFrame({"feature": ["A", "B", "A", "B", None]})
    y = pd.Series([1, 2, 3, 4, 5])
    feature = OrdinalFeature("feature", ["A", "B", "C"])
    result = get_target_values_by_modality(X, y, feature)
    expected = pd.Series({"A": [1, 3], "B": [2, 4], "C": []})
    assert result.equals(expected)


def test_get_target_values_by_modality_unordered_labels():
    X = pd.DataFrame({"feature": ["A", "B", "A", "B", "C"]})
    y = pd.Series([1, 2, 3, 4, 5])
    feature = OrdinalFeature("feature", ["C", "A", "B"])
    result = get_target_values_by_modality(X, y, feature)
    expected = pd.Series({"C": [5], "A": [1, 3], "B": [2, 4]})
    assert result.equals(expected)


def test_get_target_values_by_modality_missing_labels():
    X = pd.DataFrame({"feature": ["A", "B", "A", "B", "C"]})
    y = pd.Series([1, 2, 3, 4, 5])
    feature = OrdinalFeature("feature", ["A", "B"])
    result = get_target_values_by_modality(X, y, feature)
    expected = pd.Series({"A": [1, 3], "B": [2, 4]})
    assert result.equals(expected)


def test_get_target_values_by_modality_extra_labels():
    X = pd.DataFrame({"feature": ["A", "B", "A", "B", "C"]})
    y = pd.Series([1, 2, 3, 4, 5])
    feature = OrdinalFeature("feature", ["A", "B", "C", "D"])
    result = get_target_values_by_modality(X, y, feature)
    expected = pd.Series({"A": [1, 3], "B": [2, 4], "C": [5], "D": []})
    assert result.equals(expected)


@fixture(params=[KruskalCombinations])
def evaluator(request: FixtureRequest) -> CombinationEvaluator:
    """Evaluator instance fixture, passed as combination_evaluator to the carver."""
    return request.param()


def test_continuous_carver_initialization():
    """Test ContinuousCarver initialization."""
    features = Features(
        categoricals=["feature1"],
        ordinals={"feature2": ["low", "medium", "high"]},
        numericals=["feature3"],
    )
    min_freq = 0.1
    carver = ContinuousCarver(min_freq=min_freq, max_n_mod=5, features=features)
    assert carver.min_freq == min_freq
    assert carver.features == features
    assert carver.config.dropna is True
    assert isinstance(carver.combination_evaluator, KruskalCombinations)
    assert carver.max_n_mod == 5

    max_n_mod = 8
    carver = ContinuousCarver(
        min_freq=0.1,
        features=features,
        max_n_mod=max_n_mod,
        combination_evaluator=KruskalCombinations(),
    )
    assert isinstance(carver.combination_evaluator, KruskalCombinations)
    assert carver.max_n_mod == max_n_mod

    carver = ContinuousCarver(
        min_freq=0.1,
        features=features,
        max_n_mod=max_n_mod,
        combination_evaluator=KruskalCombinations(),
    )
    assert isinstance(carver.combination_evaluator, KruskalCombinations)
    assert carver.max_n_mod == max_n_mod

    with raises(ValueError):
        ContinuousCarver(
            min_freq=0.1,
            features=features,
            max_n_mod=max_n_mod,
            combination_evaluator=CramervCombinations(),
        )

    with raises(ValueError):
        ContinuousCarver(
            min_freq=0.1,
            features=features,
            max_n_mod=max_n_mod,
            combination_evaluator=TschuprowtCombinations(),
        )


def test_continuous_carver_prepare_samples(evaluator: CombinationEvaluator):
    """Test ContinuousCarver _prepare_samples method."""
    features = Features(
        categoricals=["feature1"],
        ordinals={"feature2": ["low", "medium", "high"]},
        numericals=["feature3"],
    )
    carver = ContinuousCarver(min_freq=0.1, max_n_mod=5, features=features, combination_evaluator=evaluator)
    X = pd.DataFrame({"feature1": ["A", "B", "A"], "feature2": ["low", "medium", "high"], "feature3": [1, 2, 3]})

    # with wrong target
    y = pd.Series([0, 1, 1])
    samples = Samples(train=Sample(X, y))

    with raises(ValueError):
        carver._prepare_samples(samples)

    # with wrong target
    y = pd.Series([0.2, 1.5, "1"])
    samples = Samples(train=Sample(X, y))

    with raises(ValueError):
        carver._prepare_samples(samples)

    # with right target
    y = pd.Series([0.1, 1.2, 0.5])
    samples = Samples(train=Sample(X, y))

    prepared_samples = carver._prepare_samples(samples)
    assert isinstance(prepared_samples, Samples)


def test_continuous_carver_aggregator(evaluator: CombinationEvaluator):
    """Test ContinuousCarver _aggregator method."""
    features = Features(
        categoricals=["feature1"],
        ordinals={"feature2": ["low", "medium", "high"]},
        numericals=["feature3"],
    )
    carver = ContinuousCarver(min_freq=0.1, max_n_mod=5, features=features, combination_evaluator=evaluator)
    X = pd.DataFrame({"feature1": ["A", "B", "A"], "feature2": ["low", "medium", "high"], "feature3": [1, 2, 3]})
    y = pd.Series([0.1, 1.2, 0.5])
    xtabs = carver._aggregator(X, y)
    print(xtabs)

    expected = {
        "feature1": pd.Series({"A": [0.1, 0.5], "B": [1.2]}),
        "feature2": pd.Series({"low": [0.1], "medium": [1.2], "high": [0.5]}),
        "feature3": pd.Series({1: [0.1], 2: [1.2], 3: [0.5]}),
    }
    assert isinstance(xtabs, dict)
    assert "feature1" in xtabs
    assert "feature2" in xtabs
    assert "feature3" in xtabs
    for feature in features.names:
        assert all(xtabs[feature].index == expected[feature].index)
        assert (xtabs[feature].values == expected[feature].values).all()


def test_carve_feature_with_best_combination(evaluator):
    """Test ContinuousCarver _carve_feature method."""

    features = Features(
        categoricals=["feature1"],
        ordinals={"feature2": ["low", "medium", "high"]},
        numericals=["feature3"],
    )
    X = pd.DataFrame(
        {
            "feature1": ["A", "B", "A", "C"],
            "feature2": ["low", "medium", "high", "high"],
            "feature3": [1, 2, 3, float("nan")],
        }
    )
    y = pd.Series([0.1, 1.2, 0.5, 0.7])
    samples = Samples(train=Sample(X, y))

    # initializing carver
    carver = ContinuousCarver(
        features=features,
        min_freq=0.1,
        max_n_mod=5,
        combination_evaluator=evaluator,
        config=ProcessingConfig(dropna=True, verbose=False),
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
    assert feature.content == {"A": ["A"], "C": ["C"], "B": ["B"]}

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
        numericals=["feature3"],
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
    y = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.2, 0.7, 0.5, 1.2, 0.7, 0.8])
    samples = Samples(train=Sample(X, y))

    # initializing carver
    carver = ContinuousCarver(
        features=features,
        min_freq=0.9,
        max_n_mod=5,
        combination_evaluator=evaluator,
        config=ProcessingConfig(dropna=True, verbose=False),
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
    """Test ContinuousCarver fit method with best combination."""

    features = Features(
        categoricals=["feature1"],
        ordinals={"feature2": ["low", "medium", "high"]},
        numericals=["feature3"],
    )
    X = pd.DataFrame(
        {
            "feature1": ["A", "B", "A", "C"],
            "feature2": ["low", "medium", "high", "high"],
            "feature3": [1, 2, 3, float("nan")],
        }
    )
    y = pd.Series([0.1, 1.2, 0.5, 0.7])

    # initializing carver
    carver = ContinuousCarver(
        features=features,
        min_freq=0.1,
        max_n_mod=5,
        combination_evaluator=evaluator,
        config=ProcessingConfig(dropna=True, verbose=False),
    )

    # fitting carver
    carver.fit(X, y)

    feature = features[0]
    print(feature.content)
    assert feature in carver.features
    assert feature.content == {"A": ["A"], "C": ["C"], "B": ["B"]}

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
    """Test ContinuousCarver fit method without best combination."""

    features = Features(
        categoricals=["feature1"],
        ordinals={"feature2": ["low", "medium", "high"]},
        numericals=["feature3"],
    )
    X = pd.DataFrame(
        {
            "feature1": ["A", "B", "A", "C"],
            "feature2": ["low", "medium", "high", "high"],
            "feature3": [1, 2, 3, float("nan")],
        }
    )
    y = pd.Series([0.1, 1.2, 0.5, 0.7])

    # initializing carver
    carver = ContinuousCarver(
        features=features,
        min_freq=0.9,
        max_n_mod=5,
        combination_evaluator=evaluator,
        config=ProcessingConfig(dropna=True, verbose=False),
    )

    # fitting carver
    carver.fit(X, y)

    # carving a feature
    assert len(features) == 0


def test_continuous_carver_fit_transform_with_small_data_not_ordinal(
    evaluator: CombinationEvaluator,
):
    """Test ContinuousCarver fit_transform method."""
    features = Features(
        categoricals=["feature1"],
        ordinals={"feature2": ["low", "medium", "high"]},
        numericals=["feature3"],
    )
    carver = ContinuousCarver(
        min_freq=0.1,
        max_n_mod=5,
        features=features,
        combination_evaluator=evaluator,
        config=ProcessingConfig(dropna=True, ordinal_encoding=False, copy=False),
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
    y = pd.Series([0.1, 1.2, 0.5, 0.7], index=idx)
    X_transformed = carver.fit_transform(X, y)

    print(carver.features("feature1").content)
    print(X_transformed)
    expected = pd.DataFrame(
        {
            "feature1": ["A", "B", "A", "C"],
            "feature2": ["low", "medium", "high", "high"],
            "feature3": ["(-inf, 1.00e+00]", "(1.00e+00, 2.00e+00]", "(2.00e+00, inf)", "__NAN__"],
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


def test_continuous_carver_fit_transform_with_small_data_ordinal(evaluator: CombinationEvaluator):
    """Test ContinuousCarver fit_transform method."""
    features = Features(
        categoricals=["feature1"],
        ordinals={"feature2": ["low", "medium", "high"]},
        numericals=["feature3"],
    )
    carver = ContinuousCarver(
        min_freq=0.1,
        max_n_mod=5,
        features=features,
        combination_evaluator=evaluator,
        config=ProcessingConfig(dropna=True, ordinal_encoding=True, copy=False),
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
    y = pd.Series([0.1, 1.2, 0.5, 0.7], index=idx)
    X_transformed = carver.fit_transform(X, y)

    print(carver.features("feature1").content)
    print(X_transformed)
    expected = pd.DataFrame(
        {
            "feature1": [0, 2, 0, 1],
            "feature2": [0, 1, 2, 2],
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


def test_continuous_carver_fit_transform_with_large_data(evaluator: CombinationEvaluator):
    """Test ContinuousCarver fit_transform method."""
    features = Features(
        categoricals=["feature1"],
        ordinals={"feature2": ["low", "medium", "high"]},
        numericals=["feature3"],
    )
    carver = ContinuousCarver(
        min_freq=0.1,
        max_n_mod=5,
        features=features,
        combination_evaluator=evaluator,
        config=ProcessingConfig(dropna=True, ordinal_encoding=False, copy=False),
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
    y = pd.Series(
        [0.1, 1.2, 0.5, 0.7, 1.1, 1.1, 1.2, 1.3, 1.4, 1.45, 1.6, 1.7, 1.8, 2.4, 2.5, 2.9, 3.9],
        index=idx,
    )
    X_transformed = carver.fit_transform(X, y)

    print(carver.features("feature1").content)
    print(X_transformed)
    # Under Wilson-CI gating on n=17, neither "B" nor "medium"/"high" are
    # significantly below min_freq, so the carver doesn't merge them.
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
                "(-inf, 1.00e+00]",
                "(1.00e+00, 2.00e+00]",
                "(2.00e+00, inf)",
                "__NAN__",
                "(2.00e+00, inf)",
                "(-inf, 1.00e+00]",
                "(1.00e+00, 2.00e+00]",
                "(2.00e+00, inf)",
                "(-inf, 1.00e+00]",
                "(1.00e+00, 2.00e+00]",
                "__NAN__",
                "(2.00e+00, inf)",
                "(-inf, 1.00e+00]",
                "(1.00e+00, 2.00e+00]",
                "(2.00e+00, inf)",
                "(-inf, 1.00e+00]",
                "(1.00e+00, 2.00e+00]",
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


def test_continuous_carver_fit_transform_with_wrong_dev(evaluator: CombinationEvaluator):
    """Test ContinuousCarver fit_transform method."""
    features = Features(
        categoricals=["feature1"],
        ordinals={"feature2": ["low", "medium", "high"]},
        numericals=["feature3"],
    )
    carver = ContinuousCarver(
        min_freq=0.1,
        max_n_mod=5,
        features=features,
        combination_evaluator=evaluator,
        config=ProcessingConfig(dropna=True, ordinal_encoding=True, copy=False),
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
    y = pd.Series([0.1, 1.2, 0.5, 0.7], index=idx)
    X_dev = pd.DataFrame(
        {
            "feature1": ["A", "B", "A", "C"],
            "feature2": ["low", "medium", "high", "high"],
            "feature3": [1, 2, 3, float("nan")],
        },
        index=idx,
    )
    y_dev = pd.Series([5.2, 0.1, 8.7, 0.5], index=idx)
    X_transformed = carver.fit_transform(X, y, X_dev=X_dev, y_dev=y_dev)

    print(X_transformed)
    expected = X
    assert isinstance(X_transformed, pd.DataFrame)
    assert all(X_transformed.index == expected.index)
    assert all(X_transformed.index == X.index)
    assert all(X_transformed.columns == expected.columns)
    assert all(X.columns == expected.columns)
    assert X.equals(expected)
    assert X_transformed.equals(expected)

    assert all(X_dev.index == expected.index)
    assert all(X_dev.columns == expected.columns)
    assert X_dev.equals(expected)
    assert len(carver.features) == 0


def test_continuous_carver_save_load(tmp_path, evaluator: CombinationEvaluator):
    """Test ContinuousCarver save and load methods."""
    features = Features(
        categoricals=["feature1"],
        ordinals={"feature2": ["low", "medium", "high"]},
        numericals=["feature3"],
    )
    carver = ContinuousCarver(min_freq=0.1, max_n_mod=5, features=features, combination_evaluator=evaluator)
    carver_file = tmp_path / "binary_carver.json"
    carver.save(str(carver_file))
    loaded_carver = ContinuousCarver.load(str(carver_file))
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


def _fit_continuous_carver(
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
) -> tuple[ContinuousCarver, pd.DataFrame, pd.DataFrame, Features]:
    """Build features, fit ContinuousCarver, transform train and dev.

    Common setup factored out of the original mega-test; each focused test below
    calls it with only the parameter combinations it actually exercises.
    """
    features = Features(
        categoricals=qualitative_features,
        ordinals=values_orders,
        numericals=quantitative_features,
    )
    for feature_name in ["nan", "ones", "ones_nan"]:
        features.remove(feature_name)

    # (chained_features / level0_to_level1 / level1_to_level2 are retained as parameters for
    # fixture compatibility; rare-modality rollup is now handled inside the carver pipeline.)
    auto_carver = ContinuousCarver(
        min_freq=0.1,
        max_n_mod=4,
        features=features,
        combination_evaluator=evaluator,
        config=ProcessingConfig(dropna=dropna, ordinal_encoding=ordinal_encoding, copy=copy, verbose=False),
    )
    x_discretized = auto_carver.fit_transform(
        x_train,
        x_train["continuous_target"],
        X_dev=x_dev_1,
        y_dev=x_dev_1["continuous_target"],
    )
    x_dev_discretized = auto_carver.transform(x_dev_1)
    return auto_carver, x_discretized, x_dev_discretized, features


def test_continuous_carver_end_to_end_invariants(
    x_train: pd.DataFrame,
    x_dev_1: pd.DataFrame,
    quantitative_features: list[str],
    qualitative_features: list[str],
    values_orders: dict[str, list[str]],
    chained_features: list[str],
    level0_to_level1: dict[str, list[str]],
    level1_to_level2: dict[str, list[str]],
    dropna: bool,
) -> None:
    """Modality counts, NaN handling, train/dev robustness, value preservation."""
    raw_x_train = x_train.copy()
    target = "continuous_target"
    auto_carver, x_discretized, x_dev_discretized, features = _fit_continuous_carver(
        x_train,
        x_dev_1,
        qualitative_features,
        quantitative_features,
        values_orders,
        chained_features,
        level0_to_level1,
        level1_to_level2,
        KruskalCombinations(),
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
            f"Missing value in output! Some values are been dropped for qualitative feature: {feature}"
        )


def test_continuous_carver_copy_semantics(
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
    """copy=True leaves x_train comparable to x_discretized (original assertion preserved)."""
    _, x_discretized, _, features = _fit_continuous_carver(
        x_train,
        x_dev_1,
        qualitative_features,
        quantitative_features,
        values_orders,
        chained_features,
        level0_to_level1,
        level1_to_level2,
        KruskalCombinations(),
        copy=copy,
    )
    feature_names = features.names
    # original mega-test only asserts on copy=True; copy=False makes no assertion
    if copy:
        assert all(
            x_discretized[feature_names].fillna(Constants.NAN) == x_train[feature_names].fillna(Constants.NAN)
        ), "Not copied correctly"


def test_continuous_carver_serialization_roundtrip(
    tmp_path: Path,
    x_train: pd.DataFrame,
    x_dev_1: pd.DataFrame,
    quantitative_features: list[str],
    qualitative_features: list[str],
    values_orders: dict[str, list[str]],
    chained_features: list[str],
    level0_to_level1: dict[str, list[str]],
    level1_to_level2: dict[str, list[str]],
) -> None:
    """Save/load on a fitted carver preserves summary and transform output."""
    raw_x_train = x_train.copy()
    auto_carver, x_discretized, _, features = _fit_continuous_carver(
        x_train,
        x_dev_1,
        qualitative_features,
        quantitative_features,
        values_orders,
        chained_features,
        level0_to_level1,
        level1_to_level2,
        KruskalCombinations(),
    )
    feature_names = features.names

    carver_file = tmp_path / "test.json"
    auto_carver.save(str(carver_file))
    loaded_carver = ContinuousCarver.load(str(carver_file))

    assert all(loaded_carver.summary == auto_carver.summary), "Non-identical summaries when loading from JSON"
    loaded_x_train = loaded_carver.transform(raw_x_train)
    assert all(x_discretized[feature_names] == loaded_x_train[loaded_carver.features.names]), (
        "Non-identical discretized values when loading from JSON"
    )


def test_continuous_carver_wrong_dev_transform(
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
    auto_carver, _, _, _ = _fit_continuous_carver(
        x_train,
        x_dev_1,
        qualitative_features,
        quantitative_features,
        values_orders,
        chained_features,
        level0_to_level1,
        level1_to_level2,
        KruskalCombinations(),
    )

    # unexpected modal on a feature that has_default — does not raise
    auto_carver.transform(x_dev_wrong_1)

    # unexpected nans (even though it has_default) — raises
    with raises(ValueError):
        auto_carver.transform(x_dev_wrong_2)

    # unexpected modal on a feature that does not have default — raises
    with raises(ValueError):
        auto_carver.transform(x_dev_wrong_3)


def test_continuous_carver_unknown_ordinal_values_raises(
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

    auto_carver = ContinuousCarver(
        min_freq=0.15,
        max_n_mod=5,
        features=features,
        combination_evaluator=KruskalCombinations(),
        config=ProcessingConfig(verbose=False),
    )
    with raises(ValueError):
        auto_carver.fit_transform(x_train_wrong_2, x_train_wrong_2["continuous_target"])


def _fit_continuous_for_parity(n_jobs: int) -> dict[str, dict]:
    """Fits a ContinuousCarver on a small synthetic frame and returns
    ``{version: feature.content}`` so the parallel path can be compared to the
    sequential one without depending on object identity."""

    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame(
        {
            "feature1": rng.choice(["A", "B", "C", "D"], size=n),
            "feature2": rng.choice(["low", "medium", "high"], size=n),
            "feature3": rng.normal(size=n),
            "feature4": rng.integers(0, 20, size=n).astype(float),
        }
    )
    y = pd.Series(rng.normal(size=n) + X["feature3"] * 0.5)
    features = Features(
        categoricals=["feature1"],
        ordinals={"feature2": ["low", "medium", "high"]},
        numericals=["feature3", "feature4"],
    )
    carver = ContinuousCarver(
        features=features,
        min_freq=0.1,
        max_n_mod=4,
        combination_evaluator=KruskalCombinations(),
        config=ProcessingConfig(dropna=True, verbose=False, n_jobs=n_jobs),
    )
    carver.fit(X, y)
    return {f.version: f.content for f in carver.features}


def test_continuous_carver_parallel_features_parity():
    """n_jobs=2 must produce the same carved features as n_jobs=1 (Step 5)."""
    sequential = _fit_continuous_for_parity(n_jobs=1)
    parallel = _fit_continuous_for_parity(n_jobs=2)
    assert parallel == sequential
