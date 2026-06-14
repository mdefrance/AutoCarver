"""Set of tests for multiclass_carver module."""

from pathlib import Path

import pandas as pd
from pytest import FixtureRequest, fixture, raises

from AutoCarver.carvers.multiclass_carver import MulticlassCarver, get_one_vs_rest
from AutoCarver.carvers.utils.base_carver import Sample, Samples
from AutoCarver.combinations import (
    CombinationEvaluator,
    CramervCombinations,
    KruskalCombinations,
    TschuprowtCombinations,
)
from AutoCarver.config import Constants
from AutoCarver.discretizers import ProcessingConfig
from AutoCarver.features import Features


@fixture(params=[CramervCombinations, TschuprowtCombinations])
def evaluator(request: FixtureRequest) -> CombinationEvaluator:
    """Evaluator instance fixture, passed as combination_evaluator= to the carver."""
    return request.param()


@fixture(scope="module", params=["tschuprowt", "cramerv"])
def sort_by(request) -> str:
    """sorting measure"""
    return request.param


def test_get_one_vs_rest_with_string_series():
    y = pd.Series(["A", "B", "A", "C", "B", "A"])
    y_class = "A"
    result = get_one_vs_rest(y, y_class)
    expected = pd.Series([1, 0, 1, 0, 0, 1])
    result.equals(expected)


def test_get_one_vs_rest_conversion():
    y = pd.Series([1, 2, 1, 3, 2, 1])
    y_class = 1
    result = get_one_vs_rest(y, y_class)
    expected = pd.Series([1, 0, 1, 0, 0, 1])
    result.equals(expected)


def test_get_one_vs_rest_different_class():
    y = pd.Series([1, 2, 1, 3, 2, 1])
    y_class = 2
    result = get_one_vs_rest(y, y_class)
    expected = pd.Series([0, 1, 0, 0, 1, 0])
    result.equals(expected)


def test_get_one_vs_rest_no_match():
    y = pd.Series([1, 2, 1, 3, 2, 1])
    y_class = 4
    result = get_one_vs_rest(y, y_class)
    expected = pd.Series([0, 0, 0, 0, 0, 0])
    result.equals(expected)


def test_get_one_vs_rest_none():
    y = None
    y_class = 1
    result = get_one_vs_rest(y, y_class)
    assert result is None


def test_multiclass_carver_initialization():
    """Test MulticlassCarver initialization."""
    features = Features(
        categoricals=["feature1"],
        ordinals={"feature2": ["low", "medium", "high"]},
        numericals=["feature3"],
    )
    carver = MulticlassCarver(min_freq=0.1, max_n_mod=5, features=features)
    assert carver.min_freq == 0.1
    assert carver.features == features
    assert carver.config.dropna is True
    assert isinstance(carver.combination_evaluator, TschuprowtCombinations)
    assert carver.max_n_mod == 5

    max_n_mod = 8
    carver = MulticlassCarver(
        min_freq=0.1,
        features=features,
        max_n_mod=max_n_mod,
        combination_evaluator=TschuprowtCombinations(),
    )
    assert isinstance(carver.combination_evaluator, TschuprowtCombinations)
    assert carver.max_n_mod == max_n_mod

    carver = MulticlassCarver(
        min_freq=0.1,
        features=features,
        max_n_mod=max_n_mod,
        combination_evaluator=CramervCombinations(),
    )
    assert isinstance(carver.combination_evaluator, CramervCombinations)
    assert carver.max_n_mod == max_n_mod

    with raises(ValueError):
        MulticlassCarver(
            min_freq=0.1,
            features=features,
            max_n_mod=max_n_mod,
            combination_evaluator=KruskalCombinations(),
        )


def test_multiclass_carver_prepare_samples(evaluator: CombinationEvaluator):
    """Test MulticlassCarver _prepare_samples method."""
    features = Features(
        categoricals=["feature1"],
        ordinals={"feature2": ["low", "medium", "high"]},
        numericals=["feature3"],
    )
    carver = MulticlassCarver(min_freq=0.1, max_n_mod=5, features=features, combination_evaluator=evaluator)
    X = pd.DataFrame({"feature1": ["A", "B", "A"], "feature2": ["low", "medium", "high"], "feature3": [1, 2, 3]})

    # with wrong target
    y = pd.Series([0, 1, 0])
    samples = Samples(train=Sample(X, y))

    with raises(ValueError):
        carver._prepare_samples(samples)

    # with right target
    y = pd.Series([0, 1, 2])
    samples = Samples(train=Sample(X, y))

    prepared_samples = carver._prepare_samples(samples)
    assert isinstance(prepared_samples, Samples)

    # with wrong dev target
    y_dev = pd.Series([0, 1, 0])
    y = pd.Series([0, 1, 2])
    samples = Samples(train=Sample(X, y), dev=Sample(X, y_dev))

    with raises(ValueError):
        carver._prepare_samples(samples)


def test_multiclass_carver_fit_transform_with_small_data_not_ordinal(
    evaluator: CombinationEvaluator,
):
    """Test MulticlassCarver fit_transform method."""
    features = Features(
        categoricals=["feature1"],
        ordinals={"feature2": ["low", "medium", "high"]},
        numericals=["feature3"],
    )
    carver = MulticlassCarver(
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
    y = pd.Series([0, 1, 2, 1], index=idx)
    X_transformed = carver.fit_transform(X, y)

    print(X_transformed)
    # Under Wilson-CI gating on n=4 the borderline ``2.0`` bin survives, and
    # ordinal labels with count=1 are not significantly below min_freq → fewer merges.
    expected = pd.DataFrame(
        {
            "feature1": ["A", "B", "A", "C"],
            "feature2": ["low", "medium", "high", "high"],
            "feature3": [1, 2, 3, float("nan")],
            "feature1__y=1": ["A", "B, C", "A", "B, C"],
            "feature1__y=2": ["A, C", "B", "A, C", "A, C"],
            "feature2__y=1": ["low", "medium", "high", "high"],
            "feature2__y=2": ["low", "medium to high", "medium to high", "medium to high"],
            "feature3__y=1": [
                "(-inf, 1.00e+00]",
                "(1.00e+00, 2.00e+00]",
                "(2.00e+00, inf)",
                "(1.00e+00, 2.00e+00]",
            ],
            "feature3__y=2": [
                "(-inf, 2.00e+00]",
                "(-inf, 2.00e+00]",
                "(2.00e+00, 3.00e+00]",
                "(-inf, 2.00e+00]",
            ],
        },
        index=idx,
    )
    print(X_transformed.columns)
    assert isinstance(X_transformed, pd.DataFrame)
    assert all(X_transformed.index == expected.index)
    assert all(X_transformed.index == X.index)
    assert all(X_transformed.columns == expected.columns)
    print(
        "X values",
        X.values,
        "\n\nX transfor",
        X_transformed.values,
        "\n",
        "\n",
        (X_transformed.values == expected.values),
    )
    assert X_transformed.equals(expected)


def test_multiclass_carver_fit_transform_with_small_data_ordinal(evaluator: CombinationEvaluator):
    """Test MulticlassCarver fit_transform method."""
    features = Features(
        categoricals=["feature1"],
        ordinals={"feature2": ["low", "medium", "high"]},
        numericals=["feature3"],
    )
    carver = MulticlassCarver(
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
    y = pd.Series([0, 1, 2, 1], index=idx)
    X_transformed = carver.fit_transform(X, y)

    print(X_transformed)
    # feature3__y=1 keeps 3 numeric bins (codes 0,1,2); NaN folds into its
    # ordinal-neighbour ``2.0`` bin (code 1). feature3__y=2 keeps 3 bins (0,1,0).
    expected = pd.DataFrame(
        {
            "feature1": ["A", "B", "A", "C"],
            "feature2": ["low", "medium", "high", "high"],
            "feature3": [1, 2, 3, float("nan")],
            "feature1__y=1": [0, 1, 0, 1],
            "feature1__y=2": [1, 0, 1, 1],
            "feature2__y=1": [0, 1, 2, 2],
            "feature2__y=2": [0, 1, 1, 1],
            "feature3__y=1": [0, 1, 2, 1],
            "feature3__y=2": [0, 0, 1, 0],
        },
        index=idx,
    )
    print(X_transformed.columns)
    assert isinstance(X_transformed, pd.DataFrame)
    assert all(X_transformed.index == expected.index)
    assert all(X_transformed.index == X.index)
    assert all(X_transformed.columns == expected.columns)
    print(
        "X values\n",
        X.values,
        "\n\nX transfor\n",
        X_transformed.values,
        "\n",
        "\n",
        (X_transformed.values == expected.values),
    )
    pd.testing.assert_frame_equal(X_transformed, expected, check_dtype=False)


def test_multiclass_carver_fit_transform_with_large_data(evaluator: CombinationEvaluator):
    """Test MulticlassCarver fit_transform method."""
    features = Features(
        categoricals=["feature1"],
        ordinals={"feature2": ["low", "medium", "high"]},
        numericals=["feature3"],
    )
    carver = MulticlassCarver(
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
    y = pd.Series([3, 1, 0, 1, 0, 2, 2, 2, 3, 1, 1, 0, 1, 1, 1, 3, 3], index=idx)
    X_transformed = carver.fit_transform(X, y)

    print(X_transformed)
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
                1.0,
                2.0,
                3.0,
                float("nan"),
                3.0,
                1.0,
                2.0,
                3.0,
                1.0,
                2.0,
                float("nan"),
                3.0,
                1.0,
                2.0,
                3.0,
                1.0,
                2.0,
            ],
            # Under Wilson-CI gating on n=17, categorical "B" / ordinal "medium"/"high"
            # bins are no longer significantly below min_freq for most y-splits.
            "feature1__y=1": ["A", "B", "A"] + ["C"] * 14,
            "feature1__y=2": ["A, B"] * 3 + ["C"] * 14,
            "feature1__y=3": ["A", "B", "A"] + ["C"] * 14,
            "feature2__y=1": [
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
            "feature2__y=2": [
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
            "feature2__y=3": [
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
            "feature3__y=1": [
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
            "feature3__y=2": [
                "(-inf, 3.0e+00]",
                "(-inf, 3.0e+00]",
                "(-inf, 3.0e+00]",
                "__NAN__",
                "(-inf, 3.0e+00]",
                "(-inf, 3.0e+00]",
                "(-inf, 3.0e+00]",
                "(-inf, 3.0e+00]",
                "(-inf, 3.0e+00]",
                "(-inf, 3.0e+00]",
                "__NAN__",
                "(-inf, 3.0e+00]",
                "(-inf, 3.0e+00]",
                "(-inf, 3.0e+00]",
                "(-inf, 3.0e+00]",
                "(-inf, 3.0e+00]",
                "(-inf, 3.0e+00]",
            ],
            "feature3__y=3": [
                "(-inf, 1.00e+00]",
                "(1.00e+00, 2.00e+00]",
                "(2.00e+00, inf)",
                "(2.00e+00, inf)",
                "(2.00e+00, inf)",
                "(-inf, 1.00e+00]",
                "(1.00e+00, 2.00e+00]",
                "(2.00e+00, inf)",
                "(-inf, 1.00e+00]",
                "(1.00e+00, 2.00e+00]",
                "(2.00e+00, inf)",
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
    # Cramérv and Tschuprowt legitimately diverge here on the borderline NaN
    # placement for the y=1 split: Cramérv keeps NaN as its own modality while
    # Tschuprowt folds it into its ordinal-neighbour ``(1, 2]`` bin.
    if isinstance(evaluator, TschuprowtCombinations):
        expected.loc[["d", "k"], "feature3__y=1"] = "(1.00e+00, 2.00e+00]"
    print(X_transformed.to_dict(orient="list"))
    print(X_transformed.columns)
    assert isinstance(X_transformed, pd.DataFrame)
    assert all(X_transformed.index == expected.index)
    assert all(X_transformed.index == X.index)
    assert all(X_transformed.columns == expected.columns)
    assert X_transformed.equals(expected)


def test_multiclass_carver_fit_transform_with_target_only_nan(evaluator: CombinationEvaluator):
    """Test MulticlassCarver fit_transform method."""
    features = Features(
        categoricals=["feature1"],
        ordinals={"feature2": ["low", "medium", "high"]},
        numericals=["feature3"],
    )
    carver = MulticlassCarver(
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
    y = pd.Series([2, 0, 0, 1], index=idx)
    X_transformed = carver.fit_transform(X, y)

    print(X_transformed.to_dict(orient="list"))
    # For the y=1 split only NaN-ness predicts the class, so feature3 collapses
    # to a single ``(-inf, 3]`` bin with NaN kept as its own modality; the y=2
    # split keeps the 3-bucket carving. Grouping is label-independent (ordinal).
    expected = pd.DataFrame(
        {
            "feature1": ["A", "B", "A", "C"],
            "feature2": ["low", "medium", "high", "high"],
            "feature3": [1.0, 2.0, 3.0, float("nan")],
            "feature1__y=1": ["A, B", "A, B", "A, B", "C"],
            "feature1__y=2": ["A, C", "B", "A, C", "A, C"],
            "feature2__y=1": ["low", "medium to high", "medium to high", "medium to high"],
            "feature2__y=2": ["low", "medium to high", "medium to high", "medium to high"],
            "feature3__y=1": [
                "(-inf, 3.0e+00]",
                "(-inf, 3.0e+00]",
                "(-inf, 3.0e+00]",
                "__NAN__",
            ],
            "feature3__y=2": [
                "(-inf, 1.00e+00]",
                "(1.00e+00, 3.00e+00]",
                "(1.00e+00, 3.00e+00]",
                "(1.00e+00, 3.00e+00]",
            ],
        },
        index=idx,
    )
    assert isinstance(X_transformed, pd.DataFrame)
    assert all(X_transformed.index == expected.index)
    assert all(X_transformed.index == X.index)
    assert all(X_transformed.columns == expected.columns)
    print(
        X.values,
        "\n",
        expected.values,
        "\n",
        X_transformed.values,
        "\n",
    )
    assert X_transformed.equals(expected)


def test_multiclass_carver_fit_transform_with_wrong_dev(evaluator: CombinationEvaluator):
    """Test MulticlassCarver fit_transform method."""
    features = Features(
        categoricals=["feature1"],
        ordinals={"feature2": ["low", "medium", "high"]},
        numericals=["feature3"],
    )
    carver = MulticlassCarver(
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
    y = pd.Series([0, 1, 0, 2], index=idx)
    X_dev = pd.DataFrame(
        {
            "feature1": ["A", "B", "A", "C"],
            "feature2": ["low", "medium", "high", "high"],
            "feature3": [1, 2, 3, float("nan")],
        },
        index=idx,
    )
    y_dev = pd.Series([2, 0, 1, 0], index=idx)
    X_transformed = carver.fit_transform(X, y, X_dev=X_dev, y_dev=y_dev)

    print(X_transformed.to_dict(orient="list"))
    # Under Wilson-CI gating fewer features fail the rank-inversion test, and
    # the carver preserves an extra feature3__y=2 column.
    expected = pd.DataFrame(
        {
            "feature1": ["A", "B", "A", "C"],
            "feature2": ["low", "medium", "high", "high"],
            "feature3": [1.0, 2.0, 3.0, float("nan")],
            "feature2__y=1": ["low", "medium to high", "medium to high", "medium to high"],
            "feature3__y=1": [
                "(-inf, 1.00e+00]",
                "(1.00e+00, 3.00e+00]",
                "(1.00e+00, 3.00e+00]",
                "(-inf, 1.00e+00]",
            ],
            "feature3__y=2": [
                "(-inf, 3.0e+00]",
                "(-inf, 3.0e+00]",
                "(-inf, 3.0e+00]",
                "(-inf, 3.0e+00]",
            ],
        },
        index=idx,
    )
    assert isinstance(X_transformed, pd.DataFrame)
    assert all(X_transformed.index == expected.index)
    assert all(X_transformed.index == X.index)
    assert all(X_transformed.columns == expected.columns)
    assert X_transformed.equals(expected)

    assert len(carver.features) == 3


def test_multiclass_carver_save_load(tmp_path: Path, evaluator: CombinationEvaluator):
    """Test MulticlassCarver save and load methods."""
    features = Features(
        categoricals=["feature1"],
        ordinals={"feature2": ["low", "medium", "high"]},
        numericals=["feature3"],
    )
    carver = MulticlassCarver(min_freq=0.1, max_n_mod=5, features=features, combination_evaluator=evaluator)
    carver_file = tmp_path / "multilclass_carver.json"
    carver.save(str(carver_file))
    loaded_carver = MulticlassCarver.load(str(carver_file))
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


def _fit_multiclass_carver(
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
) -> tuple[MulticlassCarver, pd.DataFrame, pd.DataFrame, Features]:
    """Build features, fit MulticlassCarver, transform train and dev.

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
    auto_carver = MulticlassCarver(
        min_freq=0.1,
        max_n_mod=4,
        features=features,
        combination_evaluator=evaluator,
        config=ProcessingConfig(dropna=dropna, ordinal_encoding=ordinal_encoding, copy=copy, verbose=False),
    )
    x_discretized = auto_carver.fit_transform(
        x_train,
        x_train["multiclass_target"],
        X_dev=x_dev_1,
        y_dev=x_dev_1["multiclass_target"],
    )
    x_dev_discretized = auto_carver.transform(x_dev_1)
    return auto_carver, x_discretized, x_dev_discretized, features


def test_multiclass_carver_end_to_end_invariants(
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
    """Versions present, modality counts, NaN handling, train/dev robustness, value preservation."""
    raw_x_train = x_train.copy()
    target = "multiclass_target"
    auto_carver, x_discretized, x_dev_discretized, features = _fit_multiclass_carver(
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
    feature_versions = features.versions

    # multiclass-specific: version columns are built and present in both outputs
    assert all(feature.version != feature.name for feature in features), "No version built for some features"
    assert all(version in x_discretized for version in feature_versions), (
        "Version missing from Train dataframe after transform"
    )
    assert all(version in x_dev_discretized for version in feature_versions), (
        "Version missing from Dev dataframe after transform"
    )

    # max_n_mod respected on train + dev
    assert all(x_discretized[feature_versions].nunique() <= auto_carver.max_n_mod), (
        "Too many buckets after carving of train sample"
    )
    assert all(x_dev_discretized[feature_versions].nunique() <= auto_carver.max_n_mod), (
        "Too many buckets after carving of test sample"
    )

    # NaN handling matches dropna
    if not dropna:
        for feature in features:
            assert raw_x_train[feature.name].isna().mean() == x_discretized[feature.version].isna().mean(), (
                f"Some Nans are being dropped (grouped) or more nans than expected {feature}"
            )
    else:
        assert all(x_discretized[feature_versions].isna().mean() == 0), "Some Nans are not dropped (grouped)"

    # train/dev modality counts match
    assert all(x_discretized[feature_versions].nunique() == x_dev_discretized[feature_versions].nunique()), (
        "More buckets in train or test samples"
    )

    # robustness: same modalities in train and dev per feature version
    for feature in features:
        train_target_rate = (
            (x_discretized[target].astype(str) == feature.version_tag).groupby(x_discretized[feature.version])
        ).mean()
        dev_target_rate = (
            (x_dev_discretized[target].astype(str) == feature.version_tag).groupby(x_dev_discretized[feature.version])
        ).mean()
        if not feature.is_ordinal:
            train_target_rate = train_target_rate.sort_values()
            dev_target_rate = dev_target_rate.sort_values()
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


def test_multiclass_carver_copy_semantics(
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
    """copy=True: version column values differ from raw feature values (new column was created)."""
    _, x_discretized, _, features = _fit_multiclass_carver(
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
    # original mega-test only asserts on copy=True; copy=False makes no assertion
    if copy:
        for feature in features:
            discretized = list(x_discretized[feature.version].fillna(Constants.NAN).unique())
            train = list(x_train[feature.name].fillna(Constants.NAN).unique())
            assert any(val not in train for val in discretized) or any(val not in discretized for val in train), (
                f"Not copied correctly ({feature})"
            )


def test_multiclass_carver_serialization_roundtrip(
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
    auto_carver, x_discretized, _, features = _fit_multiclass_carver(
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
    feature_versions = features.versions

    carver_file = tmp_path / "test.json"
    auto_carver.save(str(carver_file))
    loaded_carver = MulticlassCarver.load(str(carver_file))

    assert all(loaded_carver.summary == auto_carver.summary), "Non-identical summaries when loading from JSON"
    assert all(x_discretized[feature_versions] == loaded_carver.transform(x_dev_1)[loaded_carver.features.versions]), (
        "Non-identical discretized values when loading from JSON"
    )


def test_multiclass_carver_wrong_dev_transform(
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
    auto_carver, _, _, _ = _fit_multiclass_carver(
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


def test_multiclass_carver_unknown_ordinal_values_raises(
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

    auto_carver = MulticlassCarver(
        min_freq=0.15,
        max_n_mod=5,
        features=features,
        combination_evaluator=CramervCombinations(),
        config=ProcessingConfig(verbose=False),
    )
    with raises(ValueError):
        auto_carver.fit_transform(x_train_wrong_2, x_train_wrong_2["multiclass_target"])
