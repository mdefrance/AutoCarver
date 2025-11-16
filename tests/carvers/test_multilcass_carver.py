"""Set of tests for multiclass_carver module.
"""

from pathlib import Path

from pandas import DataFrame, Series
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
from AutoCarver.discretizers import ChainedDiscretizer
from AutoCarver.features import Features


@fixture(params=[CramervCombinations, TschuprowtCombinations])
def evaluator(request: FixtureRequest) -> CombinationEvaluator:
    """CombinationEvaluator fixture."""
    return request.param()


@fixture(scope="module", params=["tschuprowt", "cramerv"])
def sort_by(request) -> str:
    """sorting measure"""
    return request.param


def test_get_one_vs_rest_with_string_series():
    y = Series(["A", "B", "A", "C", "B", "A"])
    y_class = "A"
    result = get_one_vs_rest(y, y_class)
    expected = Series([1, 0, 1, 0, 0, 1])
    result.equals(expected)


def test_get_one_vs_rest_conversion():
    y = Series([1, 2, 1, 3, 2, 1])
    y_class = 1
    result = get_one_vs_rest(y, y_class)
    expected = Series([1, 0, 1, 0, 0, 1])
    result.equals(expected)


def test_get_one_vs_rest_different_class():
    y = Series([1, 2, 1, 3, 2, 1])
    y_class = 2
    result = get_one_vs_rest(y, y_class)
    expected = Series([0, 1, 0, 0, 1, 0])
    result.equals(expected)


def test_get_one_vs_rest_no_match():
    y = Series([1, 2, 1, 3, 2, 1])
    y_class = 4
    result = get_one_vs_rest(y, y_class)
    expected = Series([0, 0, 0, 0, 0, 0])
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
        quantitatives=["feature3"],
    )
    carver = MulticlassCarver(min_freq=0.1, features=features, dropna=True)
    assert carver.min_freq == 0.1
    assert carver.features == features
    assert carver.dropna is True
    assert isinstance(carver.combinations, TschuprowtCombinations)
    assert carver.combinations.max_n_mod == 5

    max_n_mod = 8
    carver = MulticlassCarver(
        min_freq=0.1, features=features, dropna=True, combinations=TschuprowtCombinations(max_n_mod)
    )
    assert isinstance(carver.combinations, TschuprowtCombinations)
    assert carver.combinations.max_n_mod == max_n_mod

    carver = MulticlassCarver(
        min_freq=0.1, features=features, combinations=CramervCombinations(max_n_mod)
    )
    assert isinstance(carver.combinations, CramervCombinations)
    assert carver.combinations.max_n_mod == max_n_mod

    with raises(ValueError):
        MulticlassCarver(
            min_freq=0.1, features=features, combinations=KruskalCombinations(max_n_mod)
        )


def test_multiclass_carver_prepare_data(evaluator: CombinationEvaluator):
    """Test MulticlassCarver _prepare_data method."""
    features = Features(
        categoricals=["feature1"],
        ordinals={"feature2": ["low", "medium", "high"]},
        quantitatives=["feature3"],
    )
    carver = MulticlassCarver(min_freq=0.1, features=features, dropna=True, combinations=evaluator)
    X = DataFrame(
        {"feature1": ["A", "B", "A"], "feature2": ["low", "medium", "high"], "feature3": [1, 2, 3]}
    )

    # with wrong target
    y = Series([0, 1, 0])
    samples = Samples(train=Sample(X, y))

    with raises(ValueError):
        carver._prepare_data(samples)

    # with right target
    y = Series([0, 1, 2])
    samples = Samples(train=Sample(X, y))

    prepared_samples = carver._prepare_data(samples)
    assert isinstance(prepared_samples, Samples)

    # with wrong dev target
    y_dev = Series([0, 1, 0])
    y = Series([0, 1, 2])
    samples = Samples(train=Sample(X, y), dev=Sample(X, y_dev))

    with raises(ValueError):
        carver._prepare_data(samples)


def test_multiclass_carver_fit_transform_with_small_data_not_ordinal(
    evaluator: CombinationEvaluator,
):
    """Test MulticlassCarver fit_transform method."""
    features = Features(
        categoricals=["feature1"],
        ordinals={"feature2": ["low", "medium", "high"]},
        quantitatives=["feature3"],
    )
    carver = MulticlassCarver(
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
    y = Series([0, 1, 2, 1], index=idx)
    X_transformed = carver.fit_transform(X, y)

    print(X_transformed)
    expected = DataFrame(
        {
            "feature1": ["A", "B", "A", "C"],
            "feature2": ["low", "medium", "high", "high"],
            "feature3": [1, 2, 3, float("nan")],
            "feature1__y=1": ["A", "B, C", "A", "B, C"],
            "feature1__y=2": ["A, C", "B", "A, C", "A, C"],
            "feature2__y=1": ["low", "medium", "high", "high"],
            "feature2__y=2": ["low", "medium to high", "medium to high", "medium to high"],
            "feature3__y=1": ["x <= 1.0e+00", "1.0e+00 < x", "1.0e+00 < x", "__NAN__"],
            "feature3__y=2": [
                "x <= 1.00e+00",
                "1.00e+00 < x <= 2.00e+00",
                "2.00e+00 < x",
                "x <= 1.00e+00",
            ],
        },
        index=idx,
    )
    print(X_transformed.columns)
    assert isinstance(X_transformed, DataFrame)
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
        quantitatives=["feature3"],
    )
    carver = MulticlassCarver(
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
    y = Series([0, 1, 2, 1], index=idx)
    X_transformed = carver.fit_transform(X, y)

    print(X_transformed)
    expected = DataFrame(
        {
            "feature1": ["A", "B", "A", "C"],
            "feature2": ["low", "medium", "high", "high"],
            "feature3": [1, 2, 3, float("nan")],
            "feature1__y=1": [0, 1, 0, 1],
            "feature1__y=2": [1, 0, 1, 1],
            "feature2__y=1": [0, 1, 2, 2],
            "feature2__y=2": [0, 1, 1, 1],
            "feature3__y=1": [0, 1, 1, 2],
            "feature3__y=2": [0, 1, 2, 0],
        },
        index=idx,
    )
    print(X_transformed.columns)
    assert isinstance(X_transformed, DataFrame)
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
    assert X_transformed.equals(expected)


def test_multiclass_carver_fit_transform_with_large_data(evaluator: CombinationEvaluator):
    """Test MulticlassCarver fit_transform method."""
    features = Features(
        categoricals=["feature1"],
        ordinals={"feature2": ["low", "medium", "high"]},
        quantitatives=["feature3"],
    )
    carver = MulticlassCarver(
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
    y = Series([3, 1, 0, 1, 0, 2, 2, 2, 3, 1, 1, 0, 1, 1, 1, 3, 3], index=idx)
    X_transformed = carver.fit_transform(X, y)

    print(X_transformed)
    expected = DataFrame(
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
            "feature1__y=1": [
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
            "feature1__y=2": [
                "A, B",
                "A, B",
                "A, B",
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
            "feature1__y=3": [
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
            "feature2__y=1": [
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
                "x <= 1.0e+00",
                "1.0e+00 < x",
                "1.0e+00 < x",
                "__NAN__",
                "1.0e+00 < x",
                "x <= 1.0e+00",
                "1.0e+00 < x",
                "1.0e+00 < x",
                "x <= 1.0e+00",
                "1.0e+00 < x",
                "__NAN__",
                "1.0e+00 < x",
                "x <= 1.0e+00",
                "1.0e+00 < x",
                "1.0e+00 < x",
                "x <= 1.0e+00",
                "1.0e+00 < x",
            ],
            "feature3__y=3": [
                "x <= 1.00e+00",
                "1.00e+00 < x <= 2.00e+00",
                "2.00e+00 < x",
                "2.00e+00 < x",
                "2.00e+00 < x",
                "x <= 1.00e+00",
                "1.00e+00 < x <= 2.00e+00",
                "2.00e+00 < x",
                "x <= 1.00e+00",
                "1.00e+00 < x <= 2.00e+00",
                "2.00e+00 < x",
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
    print(X_transformed.to_dict(orient="list"))
    print(X_transformed.columns)
    assert isinstance(X_transformed, DataFrame)
    assert all(X_transformed.index == expected.index)
    assert all(X_transformed.index == X.index)
    assert all(X_transformed.columns == expected.columns)
    assert X_transformed.equals(expected)


def test_multiclass_carver_fit_transform_with_target_only_nan(evaluator: CombinationEvaluator):
    """Test MulticlassCarver fit_transform method."""
    features = Features(
        categoricals=["feature1"],
        ordinals={"feature2": ["low", "medium", "high"]},
        quantitatives=["feature3"],
    )
    carver = MulticlassCarver(
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
    y = Series([2, 0, 0, 1], index=idx)
    X_transformed = carver.fit_transform(X, y)

    print(X_transformed.to_dict(orient="list"))
    expected = DataFrame(
        {
            "feature1": ["A", "B", "A", "C"],
            "feature2": ["low", "medium", "high", "high"],
            "feature3": [1.0, 2.0, 3.0, float("nan")],
            "feature1__y=1": ["A, B", "A, B", "A, B", "C"],
            "feature1__y=2": ["A, C", "B", "A, C", "A, C"],
            "feature2__y=1": ["low", "medium to high", "medium to high", "medium to high"],
            "feature2__y=2": ["low", "medium", "high", "high"],
            "feature3__y=2": ["x <= 1.0e+00", "1.0e+00 < x", "1.0e+00 < x", "1.0e+00 < x"],
        },
        index=idx,
    )
    assert isinstance(X_transformed, DataFrame)
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
        quantitatives=["feature3"],
    )
    carver = MulticlassCarver(
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
    y = Series([0, 1, 0, 2], index=idx)
    X_dev = DataFrame(
        {
            "feature1": ["A", "B", "A", "C"],
            "feature2": ["low", "medium", "high", "high"],
            "feature3": [1, 2, 3, float("nan")],
        },
        index=idx,
    )
    y_dev = Series([2, 0, 1, 0], index=idx)
    X_transformed = carver.fit_transform(X, y, X_dev=X_dev, y_dev=y_dev)

    print(X_transformed.to_dict(orient="list"))
    expected = DataFrame(
        {
            "feature1": ["A", "B", "A", "C"],
            "feature2": ["low", "medium", "high", "high"],
            "feature3": [1.0, 2.0, 3.0, float("nan")],
            "feature2__y=1": ["low", "medium to high", "medium to high", "medium to high"],
            "feature3__y=1": ["x <= 1.0e+00", "1.0e+00 < x", "1.0e+00 < x", "x <= 1.0e+00"],
        },
        index=idx,
    )
    assert isinstance(X_transformed, DataFrame)
    assert all(X_transformed.index == expected.index)
    assert all(X_transformed.index == X.index)
    assert all(X_transformed.columns == expected.columns)
    assert X_transformed.equals(expected)

    assert len(carver.features) == 2


def test_multiclass_carver_save_load(tmp_path: Path, evaluator: CombinationEvaluator):
    """Test MulticlassCarver save and load methods."""
    features = Features(
        categoricals=["feature1"],
        ordinals={"feature2": ["low", "medium", "high"]},
        quantitatives=["feature3"],
    )
    carver = MulticlassCarver(min_freq=0.1, features=features, dropna=True, combinations=evaluator)
    carver_file = tmp_path / "multilclass_carver.json"
    carver.save(str(carver_file))
    loaded_carver = MulticlassCarver.load(str(carver_file))
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


def test_multiclass_carver(
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
    """Tests MulticlassCarver

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

    # multiclass_target for multiclass carver
    target = "multiclass_target"

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

    # fitting with provided measure
    auto_carver = MulticlassCarver(
        min_freq=min_freq,
        features=features,
        combinations=evaluator,
        discretizer_min_freq=discretizer_min_freq,
        ordinal_encoding=ordinal_encoding,
        dropna=dropna,
        copy=copy,
        verbose=False,
    )
    print("combiiiiiiiiiiiiiiiiiiiiiiii", auto_carver.combinations, "\n")
    x_discretized = auto_carver.fit_transform(
        x_train,
        x_train[target],
        X_dev=x_dev_1,
        y_dev=x_dev_1[target],
    )
    x_dev_discretized = auto_carver.transform(x_dev_1)

    # getting kept features
    feature_versions = features.versions

    # checking that there were some versions built
    assert all(
        feature.version != feature.name for feature in features
    ), "No version built for some features"
    assert all(
        version in x_discretized for version in feature_versions
    ), "Version missing from Train dataframe after transform"
    assert all(
        version in x_dev_discretized for version in feature_versions
    ), "Version missing from Dev dataframe after transform"

    # testing that attributes where correctly used
    assert all(
        x_discretized[feature_versions].nunique() <= evaluator.max_n_mod
    ), "Too many buckets after carving of train sample"
    assert all(
        x_dev_discretized[feature_versions].nunique() <= evaluator.max_n_mod
    ), "Too many buckets after carving of test sample"

    # checking that nans were not dropped if not requested
    if not dropna:
        # iterating over each feature
        for feature in features:
            assert (
                raw_x_train[feature.name].isna().mean()
                == x_discretized[feature.version].isna().mean()
            ), f"Some Nans are being dropped (grouped) or more nans than expected {feature}"

    # checking that nans were dropped if requested
    else:
        assert all(
            x_discretized[feature_versions].isna().mean() == 0
        ), "Some Nans are not dropped (grouped)"

    # testing for differences between train and dev
    assert all(
        x_discretized[feature_versions].nunique() == x_dev_discretized[feature_versions].nunique()
    ), "More buckets in train or test samples"
    for feature in features:
        # getting target rate for version tag
        train_target_rate = (
            (x_discretized[target].astype(str) == feature.version_tag).groupby(
                x_discretized[feature.version]
            )
        ).mean()
        dev_target_rate = (
            (x_dev_discretized[target].astype(str) == feature.version_tag).groupby(
                x_dev_discretized[feature.version]
            )
        ).mean()
        # sorting by target for non-ordinal features
        if not feature.is_ordinal:
            train_target_rate = train_target_rate.sort_values()
            dev_target_rate = dev_target_rate.sort_values()
        assert all(
            train_target_rate.index == dev_target_rate.index
        ), f"Not robust feature {feature} was not dropped, or robustness test not working"

    # test that all values still are in the values_orders
    for feature in features.qualitatives:
        fitted_values = feature.values.values
        # adding nan to list of initial values
        init_values = raw_x_train[feature.name].fillna(feature.nan).unique()
        if not dropna:  # removing nan from list of initial values
            init_values = [value for value in init_values if value != feature.nan]
        assert all(value in fitted_values for value in init_values), (
            "Missing value in output! Some values have been dropped for qualitative feature"
            f": {feature.name}"
        )

    # trying out copy
    if copy:
        for feature in features:
            # unique values per feature
            discretized = list(x_discretized[feature.version].fillna(Constants.NAN).unique())
            train = list(x_train[feature.name].fillna(Constants.NAN).unique())

            assert any(val not in train for val in discretized) or any(
                val not in discretized for val in train
            ), f"Not copied correctly ({feature})"

    # testing json serialization
    carver_file = tmp_path / "test.json"
    print(auto_carver.to_json())
    auto_carver.save(str(carver_file))
    loaded_carver = MulticlassCarver.load(str(carver_file))

    # checking that reloading worked exactly the same
    assert all(
        loaded_carver.summary == auto_carver.summary
    ), "Non-identical summaries when loading from JSON"
    assert all(
        x_discretized[feature_versions]
        == loaded_carver.transform(x_dev_1)[loaded_carver.features.versions]
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
    features = Features(ordinals=values_orders)

    # removing wrong features
    features.remove("nan")
    features.remove("ones")
    features.remove("ones_nan")

    # fitting carver
    auto_carver = MulticlassCarver(
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
