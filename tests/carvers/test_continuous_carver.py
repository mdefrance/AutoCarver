"""Set of tests for continuous_carver module.
"""

from pathlib import Path

from pandas import DataFrame, Series
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
from AutoCarver.discretizers import ChainedDiscretizer
from AutoCarver.features import Features, OrdinalFeature


def test_get_target_values_by_modality_basic():
    X = DataFrame({"feature": ["A", "B", "A", "B", "C"]})
    y = Series([1, 2, 3, 4, 5])
    feature = OrdinalFeature("feature", ["A", "B", "C"])
    result = get_target_values_by_modality(X, y, feature)
    expected = Series({"A": [1, 3], "B": [2, 4], "C": [5]})
    assert result.equals(expected)


def test_get_target_values_by_modality_with_nan():
    X = DataFrame({"feature": ["A", "B", "A", "B", None]})
    y = Series([1, 2, 3, 4, 5])
    feature = OrdinalFeature("feature", ["A", "B", "C"])
    result = get_target_values_by_modality(X, y, feature)
    expected = Series({"A": [1, 3], "B": [2, 4], "C": []})
    assert result.equals(expected)


def test_get_target_values_by_modality_unordered_labels():
    X = DataFrame({"feature": ["A", "B", "A", "B", "C"]})
    y = Series([1, 2, 3, 4, 5])
    feature = OrdinalFeature("feature", ["C", "A", "B"])
    result = get_target_values_by_modality(X, y, feature)
    expected = Series({"C": [5], "A": [1, 3], "B": [2, 4]})
    assert result.equals(expected)


def test_get_target_values_by_modality_missing_labels():
    X = DataFrame({"feature": ["A", "B", "A", "B", "C"]})
    y = Series([1, 2, 3, 4, 5])
    feature = OrdinalFeature("feature", ["A", "B"])
    result = get_target_values_by_modality(X, y, feature)
    expected = Series({"A": [1, 3], "B": [2, 4]})
    assert result.equals(expected)


def test_get_target_values_by_modality_extra_labels():
    X = DataFrame({"feature": ["A", "B", "A", "B", "C"]})
    y = Series([1, 2, 3, 4, 5])
    feature = OrdinalFeature("feature", ["A", "B", "C", "D"])
    result = get_target_values_by_modality(X, y, feature)
    expected = Series({"A": [1, 3], "B": [2, 4], "C": [5], "D": []})
    assert result.equals(expected)


@fixture(params=[KruskalCombinations])
def evaluator(request: FixtureRequest) -> CombinationEvaluator:
    """CombinationEvaluator fixture."""
    return request.param()


def test_continuous_carver_initialization():
    """Test ContinuousCarver initialization."""
    features = Features(
        categoricals=["feature1"],
        ordinals={"feature2": ["low", "medium", "high"]},
        quantitatives=["feature3"],
    )
    min_freq = 0.1
    carver = ContinuousCarver(min_freq=min_freq, features=features, dropna=True)
    assert carver.min_freq == min_freq
    assert carver.features == features
    assert carver.dropna is True
    assert isinstance(carver.combinations, KruskalCombinations)
    assert carver.combinations.max_n_mod == 5

    max_n_mod = 8
    carver = ContinuousCarver(
        min_freq=0.1, features=features, dropna=True, combinations=KruskalCombinations(max_n_mod)
    )
    assert isinstance(carver.combinations, KruskalCombinations)
    assert carver.combinations.max_n_mod == max_n_mod

    carver = ContinuousCarver(
        min_freq=0.1, features=features, combinations=KruskalCombinations(max_n_mod)
    )
    assert isinstance(carver.combinations, KruskalCombinations)
    assert carver.combinations.max_n_mod == max_n_mod

    with raises(ValueError):
        ContinuousCarver(
            min_freq=0.1, features=features, combinations=CramervCombinations(max_n_mod)
        )

    with raises(ValueError):
        ContinuousCarver(
            min_freq=0.1, features=features, combinations=TschuprowtCombinations(max_n_mod)
        )


def test_continuous_carver_prepare_data(evaluator: CombinationEvaluator):
    """Test ContinuousCarver _prepare_data method."""
    features = Features(
        categoricals=["feature1"],
        ordinals={"feature2": ["low", "medium", "high"]},
        quantitatives=["feature3"],
    )
    carver = ContinuousCarver(min_freq=0.1, features=features, dropna=True, combinations=evaluator)
    X = DataFrame(
        {"feature1": ["A", "B", "A"], "feature2": ["low", "medium", "high"], "feature3": [1, 2, 3]}
    )

    # with wrong target
    y = Series([0, 1, 1])
    samples = Samples(train=Sample(X, y))

    with raises(ValueError):
        carver._prepare_data(samples)

    # with wrong target
    y = Series([0.2, 1.5, "1"])
    samples = Samples(train=Sample(X, y))

    with raises(ValueError):
        carver._prepare_data(samples)

    # with right target
    y = Series([0.1, 1.2, 0.5])
    samples = Samples(train=Sample(X, y))

    prepared_samples = carver._prepare_data(samples)
    assert isinstance(prepared_samples, Samples)


def test_continuous_carver_aggregator(evaluator: CombinationEvaluator):
    """Test ContinuousCarver _aggregator method."""
    features = Features(
        categoricals=["feature1"],
        ordinals={"feature2": ["low", "medium", "high"]},
        quantitatives=["feature3"],
    )
    carver = ContinuousCarver(min_freq=0.1, features=features, dropna=True, combinations=evaluator)
    X = DataFrame(
        {"feature1": ["A", "B", "A"], "feature2": ["low", "medium", "high"], "feature3": [1, 2, 3]}
    )
    y = Series([0.1, 1.2, 0.5])
    xtabs = carver._aggregator(X, y)
    print(xtabs)

    expected = {
        "feature1": Series({"A": [0.1, 0.5], "B": [1.2]}),
        "feature2": Series({"low": [0.1], "medium": [1.2], "high": [0.5]}),
        "feature3": Series({1: [0.1], 2: [1.2], 3: [0.5]}),
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
        quantitatives=["feature3"],
    )
    X = DataFrame(
        {
            "feature1": ["A", "B", "A", "C"],
            "feature2": ["low", "medium", "high", "high"],
            "feature3": [1, 2, 3, float("nan")],
        }
    )
    y = Series([0.1, 1.2, 0.5, 0.7])
    samples = Samples(train=Sample(X, y))

    # initializing carver
    carver = ContinuousCarver(
        features=features,
        min_freq=0.1,
        combinations=evaluator,
        dropna=True,
        verbose=False,
    )
    carver._prepare_data(samples)

    # getting aggregated data
    xaggs = carver._aggregator(samples.train.X, samples.train.y)
    xaggs_dev = carver._aggregator(samples.dev.X, samples.dev.y)

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
        quantitatives=["feature3"],
    )
    X = DataFrame(
        {
            "feature1": ["A", "B", "A", "C"],
            "feature2": ["low", "medium", "high", "high"],
            "feature3": [1, 2, 3, float("nan")],
        }
    )
    y = Series([0.1, 1.2, 0.5, 0.7])
    samples = Samples(train=Sample(X, y))

    # initializing carver
    carver = ContinuousCarver(
        features=features,
        min_freq=0.9,
        combinations=evaluator,
        dropna=True,
        verbose=False,
    )
    carver._prepare_data(samples)

    # getting aggregated data
    xaggs = carver._aggregator(samples.train.X, samples.train.y)
    xaggs_dev = carver._aggregator(samples.dev.X, samples.dev.y)

    # carving a feature
    feature = features[0]
    carver._carve_feature(feature, xaggs, xaggs_dev, "1/1")
    print(feature.content)
    assert feature not in carver.features
    assert feature.content == {"A": ["A"], "__OTHER__": ["C", "B", "__OTHER__"]}


def test_fit_with_best_combination(evaluator):
    """Test ContinuousCarver fit method with best combination."""

    features = Features(
        categoricals=["feature1"],
        ordinals={"feature2": ["low", "medium", "high"]},
        quantitatives=["feature3"],
    )
    X = DataFrame(
        {
            "feature1": ["A", "B", "A", "C"],
            "feature2": ["low", "medium", "high", "high"],
            "feature3": [1, 2, 3, float("nan")],
        }
    )
    y = Series([0.1, 1.2, 0.5, 0.7])

    # initializing carver
    carver = ContinuousCarver(
        features=features,
        min_freq=0.1,
        combinations=evaluator,
        dropna=True,
        verbose=False,
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
        quantitatives=["feature3"],
    )
    X = DataFrame(
        {
            "feature1": ["A", "B", "A", "C"],
            "feature2": ["low", "medium", "high", "high"],
            "feature3": [1, 2, 3, float("nan")],
        }
    )
    y = Series([0.1, 1.2, 0.5, 0.7])

    # initializing carver
    carver = ContinuousCarver(
        features=features,
        min_freq=0.9,
        combinations=evaluator,
        dropna=True,
        verbose=False,
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
        quantitatives=["feature3"],
    )
    carver = ContinuousCarver(
        min_freq=0.1,
        features=features,
        dropna=True,
        combinations=evaluator,
        copy=False,
        ordinal_encoding=False,
    )
    idx = ["a", "b", "c", "d"]
    X = DataFrame(
        {
            "feature1": ["A", "B", "A", "C"],
            "feature2": ["low", "medium", "high", "high"],
            "feature3": [1, 2, 3, float("nan")],
        },
        index=idx,
    )
    y = Series([0.1, 1.2, 0.5, 0.7], index=idx)
    X_transformed = carver.fit_transform(X, y)

    print(carver.features("feature1").content)
    print(X_transformed)
    expected = DataFrame(
        {
            "feature1": ["A", "B", "A", "C"],
            "feature2": ["low", "medium", "high", "high"],
            "feature3": ["x <= 1.00e+00", "1.00e+00 < x <= 2.00e+00", "2.00e+00 < x", "__NAN__"],
        },
        index=idx,
    )
    assert isinstance(X_transformed, DataFrame)
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
        quantitatives=["feature3"],
    )
    carver = ContinuousCarver(
        min_freq=0.1, features=features, dropna=True, combinations=evaluator, copy=False
    )
    idx = ["a", "b", "c", "d"]
    X = DataFrame(
        {
            "feature1": ["A", "B", "A", "C"],
            "feature2": ["low", "medium", "high", "high"],
            "feature3": [1, 2, 3, float("nan")],
        },
        index=idx,
    )
    y = Series([0.1, 1.2, 0.5, 0.7], index=idx)
    X_transformed = carver.fit_transform(X, y)

    print(carver.features("feature1").content)
    print(X_transformed)
    expected = DataFrame(
        {
            "feature1": [0, 2, 0, 1],
            "feature2": [0, 1, 2, 2],
            "feature3": [0, 1, 2, 3],
        },
        index=idx,
    )
    assert isinstance(X_transformed, DataFrame)
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


def test_continuous_carver_fit_transform_with_large_data(evaluator: CombinationEvaluator):
    """Test ContinuousCarver fit_transform method."""
    features = Features(
        categoricals=["feature1"],
        ordinals={"feature2": ["low", "medium", "high"]},
        quantitatives=["feature3"],
    )
    carver = ContinuousCarver(
        min_freq=0.1,
        features=features,
        dropna=True,
        combinations=evaluator,
        copy=False,
        ordinal_encoding=False,
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
    X = DataFrame(
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
    y = Series(
        [0.1, 1.2, 0.5, 0.7, 1.1, 1.1, 1.2, 1.3, 1.4, 1.45, 1.6, 1.7, 1.8, 2.4, 2.5, 2.9, 3.9],
        index=idx,
    )
    X_transformed = carver.fit_transform(X, y)

    print(carver.features("feature1").content)
    print(X_transformed)
    expected = DataFrame(
        {
            "feature1": [
                "A",
                "B, C",
                "A",
                "B, C",
                "B, C",
                "B, C",
                "B, C",
                "B, C",
                "B, C",
                "B, C",
                "B, C",
                "B, C",
                "B, C",
                "B, C",
                "B, C",
                "B, C",
                "B, C",
            ],
            "feature2": [
                "low",
                "medium to high",
                "medium to high",
                "medium to high",
                "low",
                "low",
                "low",
                "low",
                "low",
                "low",
                "medium to high",
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
                "__NAN__",
                "2.00e+00 < x",
                "x <= 1.00e+00",
                "1.00e+00 < x <= 2.00e+00",
                "2.00e+00 < x",
                "x <= 1.00e+00",
                "1.00e+00 < x <= 2.00e+00",
                "__NAN__",
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
    assert isinstance(X_transformed, DataFrame)
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
        quantitatives=["feature3"],
    )
    carver = ContinuousCarver(
        min_freq=0.1, features=features, dropna=True, combinations=evaluator, copy=False
    )
    idx = ["a", "b", "c", "d"]
    X = DataFrame(
        {
            "feature1": ["A", "B", "A", "C"],
            "feature2": ["low", "medium", "high", "high"],
            "feature3": [1, 2, 3, float("nan")],
        },
        index=idx,
    )
    y = Series([0.1, 1.2, 0.5, 0.7], index=idx)
    X_dev = DataFrame(
        {
            "feature1": ["A", "B", "A", "C"],
            "feature2": ["low", "medium", "high", "high"],
            "feature3": [1, 2, 3, float("nan")],
        },
        index=idx,
    )
    y_dev = Series([5.2, 0.1, 8.7, 0.5], index=idx)
    X_transformed = carver.fit_transform(X, y, X_dev=X_dev, y_dev=y_dev)

    print(X_transformed)
    expected = X
    assert isinstance(X_transformed, DataFrame)
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
        quantitatives=["feature3"],
    )
    carver = ContinuousCarver(min_freq=0.1, features=features, dropna=True, combinations=evaluator)
    carver_file = tmp_path / "binary_carver.json"
    carver.save(str(carver_file))
    loaded_carver = ContinuousCarver.load(str(carver_file))
    assert carver.min_freq == loaded_carver.min_freq
    for feature in carver.features:
        assert feature in loaded_carver.features
        assert feature.content == loaded_carver.features[feature.name].content
    assert carver.dropna == loaded_carver.dropna
    assert carver.verbose == loaded_carver.verbose
    assert carver.copy == loaded_carver.copy
    assert carver.combinations.__class__ == loaded_carver.combinations.__class__
    assert carver.combinations.max_n_mod == loaded_carver.combinations.max_n_mod
    assert carver.combinations.sort_by == loaded_carver.combinations.sort_by
    assert carver.combinations.dropna == loaded_carver.combinations.dropna
    assert carver.combinations.verbose == loaded_carver.combinations.verbose


def test_continuous_carver(
    tmp_path: Path,
    x_train: DataFrame,
    x_train_wrong_2: DataFrame,
    x_dev_1: DataFrame,
    x_dev_wrong_1: DataFrame,
    x_dev_wrong_2: DataFrame,
    x_dev_wrong_3: DataFrame,
    quantitative_features: list[str],
    qualitative_features: list[str],
    ordinal_features: list[str],
    values_orders: dict[str, list[str]],
    chained_features: list[str],
    level0_to_level1: dict[str, list[str]],
    level1_to_level2: dict[str, list[str]],
    discretizer_min_freq: float,
    ordinal_encoding: bool,
    dropna: bool,
    evaluator: CombinationEvaluator,
    copy: bool,
) -> None:
    """Tests ContinuousCarver

    Parameters
    ----------
    x_train : DataFrame
        Simulated Train DataFrame
    x_train_wrong_1 : DataFrame
        Simulated Train DataFrame with unknown values (without nans)
    x_train_wrong_2 : DataFrame
        Simulated Train DataFrame with unknown values (with nans)
    x_dev_1 : DataFrame
        Simulated Dev DataFrame
    x_dev_wrong_1 : DataFrame
        Simulated wrong Dev DataFrame with unexpected modality
    x_dev_wrong_2 : DataFrame
        Simulated wrong Dev DataFrame with unexpected nans
    x_dev_wrong_3 : DataFrame
        Simulated wrong Dev DataFrame
    quantitative_features : list[str]
        List of quantitative raw features to be carved
    qualitative_features : list[str]
        List of qualitative raw features to be carved
    ordinal_features : list[str]
        List of ordinal raw features to be carved
    values_orders : dict[str, list[str]]
        values_orders of raw features to be carved
    chained_features : list[str]
        List of chained raw features to be chained
    level0_to_level1 : dict[str, list[str]]
        Chained orders level0 to level1 of features to be chained
    level1_to_level2 : dict[str, list[str]]
        Chained orders level1 to level2 of features to be chained
    discretizer_min_freq : float
        Minimum frequency per carved modalities
    ordinal_encoding : str
        Output type 'str' or 'float'
    dropna : bool
        Whether or note to drop nans
    sort_by : str
        Sorting measure 'tschuprowt' or 'cramerv'
    copy : bool
        Whether or not to copy the input dataset
    """
    # copying x_train for comparison purposes
    raw_x_train = x_train.copy()

    # continuous_target for continuous carver
    target = "continuous_target"

    # minimum frequency per value
    min_freq = 0.15

    # defining features
    features = Features(
        categoricals=qualitative_features,
        ordinals=values_orders,
        quantitatives=quantitative_features,
    )

    # removing wrong features
    features.remove("nan")
    features.remove("ones")
    features.remove("ones_nan")

    # fitting chained discretizer
    chained_discretizer = ChainedDiscretizer(
        min_freq=min_freq,
        features=features[chained_features],
        chained_orders=[level0_to_level1, level1_to_level2],
        copy=copy,
    )
    chained_discretizer.fit(x_train)

    # minimum frequency and maximum number of modality
    min_freq = 0.1
    evaluator.max_n_mod = 4

    # fitting continuous carver
    auto_carver = ContinuousCarver(
        min_freq=min_freq,
        features=features,
        combinations=evaluator,
        ordinal_encoding=ordinal_encoding,
        discretizer_min_freq=discretizer_min_freq,
        dropna=dropna,
        copy=copy,
        verbose=False,
    )
    x_discretized = auto_carver.fit_transform(
        x_train,
        x_train[target],
        X_dev=x_dev_1,
        y_dev=x_dev_1[target],
    )
    x_dev_discretized = auto_carver.transform(x_dev_1)

    # getting kept features
    feature_names = features.names

    # testing that attributes where correctly used
    assert all(
        x_discretized[feature_names].nunique() <= evaluator.max_n_mod
    ), "Too many buckets after carving of train sample"
    assert all(
        x_dev_discretized[feature_names].nunique() <= evaluator.max_n_mod
    ), "Too many buckets after carving of test sample"

    # checking that nans were not dropped if not requested
    if not dropna:
        # testing output of nans
        assert all(
            raw_x_train[feature_names].isna().mean() == x_discretized[feature_names].isna().mean()
        ), "Some Nans are being dropped (grouped) or more nans than expected"

    # checking that nans were dropped if requested
    else:
        assert all(
            x_discretized[feature_names].isna().mean() == 0
        ), "Some Nans are not dropped (grouped)"

    # testing for differences between train and dev
    assert all(
        x_discretized[feature_names].nunique() == x_dev_discretized[feature_names].nunique()
    ), "More buckets in train or test samples"
    for feature in feature_names:
        train_target_rate = x_discretized.groupby(feature)[target].mean().sort_values()
        dev_target_rate = x_dev_discretized.groupby(feature)[target].mean().sort_values()
        assert all(
            train_target_rate.index == dev_target_rate.index
        ), f"Not robust feature {feature} was not dropped, or robustness test not working"

        # checking for final modalities less frequent than discretizer_min_freq
        train_frequency = x_discretized[feature].value_counts(normalize=True, dropna=True)
        assert not any(
            train_frequency.values < auto_carver.discretizer_min_freq
        ), f"Some modalities of {feature} are less frequent than discretizer_min_freq in train"

        dev_frequency = x_dev_discretized[feature].value_counts(normalize=True, dropna=True)
        assert not any(
            dev_frequency.values < auto_carver.discretizer_min_freq
        ), f"Some modalities {feature} are less frequent than discretizer_min_freq in dev"

    # test that all values still are in the values_orders
    for feature in features.qualitatives:
        fitted_values = feature.values.values
        # adding nan to list of initial values
        init_values = raw_x_train[feature.name].fillna(feature.nan).unique()
        if not dropna:  # removing nan from list of initial values
            init_values = [value for value in init_values if value != feature.nan]

        assert all(value in fitted_values for value in init_values), (
            "Missing value in output! Some values are been dropped for qualitative "
            f"feature: {feature}"
        )

    # testing copy functionnality
    if copy:
        assert all(
            x_discretized[feature_names].fillna(Constants.NAN)
            == x_train[feature_names].fillna(Constants.NAN)
        ), "Not copied correctly"

    # testing json serialization
    carver_file = tmp_path / "test.json"
    auto_carver.save(str(carver_file))
    loaded_carver = ContinuousCarver.load(str(carver_file))

    # checking that reloading worked exactly the same
    print("\n\ninput carver", auto_carver.summary)
    print("\n\nLoaded carver", loaded_carver.summary)
    assert all(
        loaded_carver.summary == auto_carver.summary
    ), "Non-identical summaries when loading from JSON"
    loaded_x_train = loaded_carver.transform(raw_x_train)
    assert all(
        x_discretized[feature_names] == loaded_x_train[loaded_carver.features.names]
    ), "Non-identical discretized values when loading from JSON"

    # transform dev with unexpected modal for a feature that has_default
    auto_carver.transform(x_dev_wrong_1)

    # transform dev with unexpected nans for a feature that has_default
    with raises(ValueError):
        auto_carver.transform(x_dev_wrong_2)

    # transform dev with unexpected modal for a feature that does not have default
    with raises(ValueError):
        auto_carver.transform(x_dev_wrong_3)

    # testing with unknown values
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

    # minimum frequency per value
    min_freq = 0.15

    # defining features
    print(ordinal_features, values_orders)
    features = Features(
        ordinals=values_orders,
    )

    # removing wrong features
    features.remove("nan")
    features.remove("ones")
    features.remove("ones_nan")

    # initiating continuous carver
    auto_carver = ContinuousCarver(
        min_freq=min_freq,
        features=features,
        combinations=evaluator,
        ordinal_encoding=ordinal_encoding,
        discretizer_min_freq=discretizer_min_freq,
        dropna=dropna,
        copy=copy,
        verbose=False,
    )

    # trying to carve an ordinal feature with unexpected values
    with raises(ValueError):
        x_discretized = auto_carver.fit_transform(x_train_wrong_2, x_train_wrong_2[target])
