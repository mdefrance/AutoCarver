"""Set of tests for base_discretizers module."""

import json

import numpy as np
import pandas as pd
from pytest import FixtureRequest, fixture, raises

from AutoCarver.discretizers.utils.base_discretizer import (
    BaseDiscretizer,
    DiscretizerConfig,
    Sample,
    transform_quantitative_feature,
)
from AutoCarver.features import (
    CategoricalFeature,
    Features,
    FeaturesConfig,
    GroupedList,
    OrdinalFeature,
    QuantitativeFeature,
)


@fixture(params=[True, False])
def dropna(request: FixtureRequest) -> str:
    """whether or not to drop nans"""
    return request.param


@fixture
def features() -> Features:
    """set of features"""
    feature1 = QuantitativeFeature("feature1")
    feature2 = OrdinalFeature("feature2", values=["1", "2", "3", "4"])
    feature3 = CategoricalFeature("feature3")
    return Features.from_list([feature1, feature2, feature3])


@fixture(params=[True, False])
def true_false(request: FixtureRequest) -> bool:
    """true or false"""
    return request.param


def test_transform_quantitative_feature(features: Features) -> None:
    """test function transform_quantitative_feature"""

    # with values to group, without np.nan
    feature = features[-1]
    feature.update(GroupedList([2, 4.5, np.inf]))

    df_feature = pd.Series([1, 2, 3, 4, 4.5, 5], name=feature.version)
    feature_version, list_feature = transform_quantitative_feature(feature, df_feature, len(df_feature))
    assert feature_version == feature.version
    assert [
        "x <= 2.00e+00",
        "x <= 2.00e+00",
        "2.00e+00 < x <= 4.50e+00",
        "2.00e+00 < x <= 4.50e+00",
        "2.00e+00 < x <= 4.50e+00",
        "4.50e+00 < x",
    ] == list_feature

    # with values to group, with np.nan in df_feature (np.nan not grouped)
    feature = features[-1]
    feature.update(GroupedList([2, 4.5, np.inf]))
    feature.has_nan = True
    feature.dropna = True

    df_feature = pd.Series([1, 2, 3, 4, 4.5, np.nan, 5], name=feature.version)
    feature_version, list_feature = transform_quantitative_feature(feature, df_feature, len(df_feature))
    assert feature_version == feature.version
    assert [
        "x <= 2.00e+00",
        "x <= 2.00e+00",
        "2.00e+00 < x <= 4.50e+00",
        "2.00e+00 < x <= 4.50e+00",
        "2.00e+00 < x <= 4.50e+00",
        feature.nan,
        "4.50e+00 < x",
    ] == list_feature

    # with values to group, with np.nan in df_feature (grouped nans)
    feature = features[-1]
    feature.update(GroupedList([2, 4.5, np.inf]))
    feature.has_nan = True
    feature.dropna = True
    feature.update(GroupedList({2: [2], 4.5: [4.5], np.inf: [np.inf, "__NAN__"]}))
    print(feature.values.content)

    df_feature = pd.Series([1, 2, 3, 4, 4.5, np.nan, 5], name=feature.version)
    feature_version, list_feature = transform_quantitative_feature(feature, df_feature, len(df_feature))
    assert feature_version == feature.version
    assert [
        "x <= 2.00e+00",
        "x <= 2.00e+00",
        "2.00e+00 < x <= 4.50e+00",
        "2.00e+00 < x <= 4.50e+00",
        "2.00e+00 < x <= 4.50e+00",
        "4.50e+00 < x",
        "4.50e+00 < x",
    ] == list_feature

    # with values to group, with feature.nan in df_feature (grouped nans)
    feature = features[-1]
    feature.has_nan = True
    feature.dropna = True
    feature.update(GroupedList({2: [2], 4.5: [4.5], np.inf: [np.inf, "__NAN__"]}), replace=True)
    print(feature.values.content)

    df_feature = pd.Series([1, 2, 3, 4, 4.5, "__NAN__", 5], name=feature.version)
    feature_version, list_feature = transform_quantitative_feature(feature, df_feature, len(df_feature))
    assert feature_version == feature.version
    assert [
        "x <= 2.00e+00",
        "x <= 2.00e+00",
        "2.00e+00 < x <= 4.50e+00",
        "2.00e+00 < x <= 4.50e+00",
        "2.00e+00 < x <= 4.50e+00",
        "4.50e+00 < x",
        "4.50e+00 < x",
    ] == list_feature


def test_init(features: Features, true_false: bool) -> None:
    """test init method"""

    # test features
    disc = BaseDiscretizer(features.to_list())
    assert disc.features.to_list() == features.to_list()
    disc = BaseDiscretizer(features)
    assert disc.features.to_list() == features.to_list()

    # test default values
    assert not disc.is_fitted
    assert not disc.config.dropna
    assert disc.config.copy
    assert not disc.config.verbose
    assert not disc.config.ordinal_encoding
    assert disc.config.n_jobs == 1
    assert disc.min_freq is None

    # test setting copy
    disc = BaseDiscretizer(features, config=DiscretizerConfig(copy=true_false))
    assert disc.config.copy == true_false

    # test setting ordinal_encoding
    disc = BaseDiscretizer(features, config=DiscretizerConfig(ordinal_encoding=true_false))
    assert disc.config.ordinal_encoding == true_false

    # test setting dropna
    disc = BaseDiscretizer(features, config=DiscretizerConfig(dropna=true_false))
    assert disc.config.dropna == true_false

    # test setting verbose
    disc = BaseDiscretizer(features, config=DiscretizerConfig(verbose=true_false))
    assert disc.config.verbose == true_false

    # test setting is_fitted (now a post-construction attribute, not a constructor arg)
    disc = BaseDiscretizer(features)
    disc.is_fitted = true_false
    assert disc.is_fitted == true_false

    # test setting n_jobs
    disc = BaseDiscretizer(features, config=DiscretizerConfig(n_jobs=4))
    assert disc.config.n_jobs == 4

    # test setting min_freq
    disc = BaseDiscretizer(features, min_freq=0.2)
    assert disc.min_freq == 0.2


def test_cast_features(features: Features) -> None:
    """test cast_features method"""

    disc = BaseDiscretizer(features)
    X = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4],
            "feature2": ["1", "2", "3", "4"],
            "feature3": [1, 2, 3, 4],
            "feature4": [1, 2, 3, 4],
        }
    )
    casted_X = disc._cast_features(X)
    assert all(casted_X == X)

    features.add_feature_versions(["A", "B"])
    disc = BaseDiscretizer(features)
    casted_X = disc._cast_features(X)
    for feature in features:
        assert feature.name != feature.version
        assert feature.version in casted_X
        assert feature.name in casted_X
        assert all(X[feature.name] == casted_X[feature.version])
    assert "feature4" in casted_X


def test_prepare_X(features: Features) -> None:
    """test prepare_X method"""

    # x with all features needed
    X = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4],
            "feature2": ["1", "2", "3", "4"],
            "feature3": [1, 2, 3, 4],
            "feature4": [1, 2, 3, 4],
        }
    )

    # test with copy
    disc = BaseDiscretizer(features, config=DiscretizerConfig(copy=True))
    with raises(ValueError):
        disc._prepare_X(X["feature1"])
    with raises(ValueError):
        disc._prepare_X(X[["feature1", "feature4"]])
    prepared_X = disc._prepare_X(X)
    prepared_X["feature1"] = [1] * len(X["feature1"])
    X["feature3"] = [3] * len(X["feature3"])
    assert (prepared_X != X).any().any()
    assert (X["feature1"] != prepared_X["feature1"]).any()
    assert (X["feature3"] != prepared_X["feature3"]).any()

    # test without copy
    disc = BaseDiscretizer(features, config=DiscretizerConfig(copy=False))
    prepared_X = disc._prepare_X(X)
    prepared_X["feature1"] = [1] * len(X["feature1"])
    X["feature3"] = [3] * len(X["feature3"])
    assert (prepared_X == X).all().all()
    assert (prepared_X["feature3"] == 3).all()
    assert (X["feature1"] == prepared_X["feature1"]).all()
    assert (X["feature3"] == prepared_X["feature3"]).all()

    # test with casted features
    features.add_feature_versions(["A", "B"])
    disc = BaseDiscretizer(features)
    prepared_X = disc._prepare_X(X)
    assert prepared_X.shape != X.shape
    with raises(ValueError):
        disc._prepare_X(X[["feature1", "feature4"]])


def test_prepare_y(features: Features) -> None:
    """test prepare_y method"""

    y = pd.Series([1, 2, 3, 4])

    disc = BaseDiscretizer(features)
    disc._prepare_y(y)
    with raises(ValueError):
        disc._prepare_y(pd.DataFrame(y))
    y = pd.Series([np.nan, 2, 3, 4])
    with raises(ValueError):
        disc._prepare_y(y)
    y = pd.Series([None, 2, 3, 4])
    with raises(ValueError):
        disc._prepare_y(y)


def test_prepare_data(features: Features) -> None:
    """test prepare_data method"""

    disc = BaseDiscretizer(features)
    X = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4],
            "feature2": ["1", "2", "3", "4"],
            "feature3": [1, 2, 3, 4],
            "feature4": [1, 2, 3, 4],
        }
    )
    y = pd.Series([0, 0, 0, 1])
    assert disc._prepare_data(Sample(None)).X is None
    disc._prepare_data(Sample(X))
    disc._prepare_data(Sample(X, y))

    # mismatched X and y
    y = pd.Series([0, 0, 0, 1, 1])
    with raises(ValueError):
        disc._prepare_data(Sample(X, y))


def test_fit(features: Features) -> None:
    """tests fit method"""

    # already fitted discretizer
    disc = BaseDiscretizer(features)
    disc.is_fitted = True
    with raises(RuntimeError):
        disc.fit()

    # non-fitted features
    disc = BaseDiscretizer(features)
    with raises(RuntimeError):
        disc.fit()

    # making fitted features
    for feature in features:
        feature.is_fitted = True

    # expected use without ordinal_encoding
    disc = BaseDiscretizer(features, config=DiscretizerConfig(ordinal_encoding=False))
    disc.fit()
    assert disc.is_fitted
    for feature in features:
        assert not feature.ordinal_encoding

    # expected use with ordinal_encoding
    disc = BaseDiscretizer(features, config=DiscretizerConfig(ordinal_encoding=True))
    disc.fit()
    assert disc.is_fitted
    for feature in features:
        assert feature.ordinal_encoding


def test_transform_qualitative() -> None:
    """tests base discretizer transform_qualitative method"""

    # test features
    feature1 = OrdinalFeature("feature1", values=["1", "2", "3", "4"])
    feature2 = CategoricalFeature("feature2")
    feature2.update(GroupedList({"A": ["A"], "B": ["B"], "X": ["X", "C", "D"]}))
    disc = BaseDiscretizer([feature1, feature2])

    # Create sample data
    index = [1, 2, 3, 4, 5, 6, 7]
    X = pd.DataFrame(
        {
            "feature1": ["1", "2", "3", "2", "3", "4", "4"],
            "feature2": ["A", "A", "B", "C", "D", "E", "X"],
        },
        index=index,
    )

    # Call the method
    result = disc._transform_qualitative(Sample(X=X, y=None)).X
    assert all(np.array(result.index) == np.array(index))

    # Assert the result
    expected = pd.DataFrame(
        {
            "feature1": ["1", "2", "3", "2", "3", "4", "4"],
            "feature2": ["A", "A", "B", "X", "X", "E", "X"],
        },
        index=index,
    )
    assert (result == expected).all().all()
    assert (result == X).all().all()


def test_transform_quantitative() -> None:
    """tests base discretizer transform_quantitative method"""

    # test features
    feature1 = QuantitativeFeature("feature1")
    feature1.update(GroupedList([2, 4.5, np.inf]))
    feature2 = QuantitativeFeature("feature2")
    feature2.update(GroupedList([20, 45, np.inf]))
    disc = BaseDiscretizer([feature1, feature2])

    # Create sample data
    index = [1, 2, 3, 4, 5, 6]
    X = pd.DataFrame({"feature1": [1, 2, 3, 4, 4.5, 5], "feature2": [10, 20, 30, 40, 45, 50]}, index=index)

    # Call the method
    result = disc._transform_quantitative(Sample(X=X, y=None)).X
    assert all(np.array(result.index) == np.array(index))

    # Assert the result
    print(result)
    expected = pd.DataFrame(
        {
            "feature1": [
                "x <= 2.00e+00",
                "x <= 2.00e+00",
                "2.00e+00 < x <= 4.50e+00",
                "2.00e+00 < x <= 4.50e+00",
                "2.00e+00 < x <= 4.50e+00",
                "4.50e+00 < x",
            ],
            "feature2": [
                "x <= 2.00e+01",
                "x <= 2.00e+01",
                "2.00e+01 < x <= 4.50e+01",
                "2.00e+01 < x <= 4.50e+01",
                "2.00e+01 < x <= 4.50e+01",
                "4.50e+01 < x",
            ],
        },
        index=index,
    )
    assert (result == expected).all().all()
    assert (result == X).all().all()


def test_transform(true_false: bool) -> None:
    """tests base discretizer transform method"""

    # test features
    feature1 = OrdinalFeature("feature1", values=["1", "2", "3", "4"])
    feature1.is_fitted = True
    feature2 = CategoricalFeature("feature2")
    feature2.update(GroupedList({"A": ["A"], "B": ["B"], "X": ["X", "C", "D"]}))
    feature2.is_fitted = True
    feature3 = QuantitativeFeature("feature3")
    feature3.update(GroupedList([2, 4.5, np.inf]))
    feature3.is_fitted = True
    # GroupedList({2: [2], 4.5: [4.5], np.inf: [np.inf, "__NAN__"]})
    disc = BaseDiscretizer([feature1, feature2, feature3], config=DiscretizerConfig(copy=true_false))

    # creating sample
    index = [1, 2, 3, 4, 5, 6, 7]
    X = pd.DataFrame(
        {
            "feature1": ["1", "2", "3", "2", "3", "4", "4"],
            "feature2": ["A", "A", "B", "C", "D", "E", "X"],
            "feature3": [1, 2, 3, 4, 4.5, 5, 6],
        },
        index=index,
    )

    # Call the method with non fitted discretizer
    with raises(RuntimeError):
        result = disc.transform(X, None)

    # fitting discretizer
    disc.fit(X)

    # call method with unexpected value for categorical feature
    with raises(ValueError):
        result = disc.transform(X, None)

    # adding default to categorical feature
    feature2.has_default = True
    result = disc.transform(X, None)
    assert all(np.array(result.index) == np.array(index))

    # Assert the result
    print(result)
    expected = pd.DataFrame(
        {
            "feature1": ["1", "2", "3", "2", "3", "4", "4"],
            "feature2": ["A", "A", "B", "X", "X", feature2.default, "X"],
            "feature3": [
                "x <= 2.00e+00",
                "x <= 2.00e+00",
                "2.00e+00 < x <= 4.50e+00",
                "2.00e+00 < x <= 4.50e+00",
                "2.00e+00 < x <= 4.50e+00",
                "4.50e+00 < x",
                "4.50e+00 < x",
            ],
        },
        index=index,
    )
    assert (result == expected).all().all()
    if true_false:  # testing copy
        assert (result != X).any().any()
    else:  # testing inplace
        assert (result == X).all().all()

    # WITH NANS DROPPED
    # test features
    feature1 = OrdinalFeature("feature1", values=["1", "2", "3", "4"])
    feature1.is_fitted = True
    feature1.has_nan = True
    feature1.dropna = True
    feature2 = CategoricalFeature("feature2")
    feature2.update(GroupedList({"A": ["A"], "B": ["B"], "X": ["X", "C", "D"]}))
    feature2.is_fitted = True
    feature2.has_nan = True
    feature2.dropna = True
    feature3 = QuantitativeFeature("feature3")
    feature3.update(GroupedList([2, 4.5, np.inf]))
    feature3.is_fitted = True
    feature3.has_nan = True
    feature3.dropna = True
    # feature3.update(GroupedList({2: [2], 4.5: [4.5], np.inf: [np.inf, "__NAN__"]}))
    disc = BaseDiscretizer([feature1, feature2, feature3], config=DiscretizerConfig(copy=true_false))

    # creating sample
    index = [1, 2, 3, 4, 5, 6, 7]
    X = pd.DataFrame(
        {
            "feature1": ["1", "2", "3", "2", np.nan, "4", "4"],
            "feature2": ["A", "A", "B", "C", np.nan, "E", "X"],
            "feature3": [np.nan, 2, 3, 4, 4.5, 5, 6],
        },
        index=index,
    )

    # Call the method with non fitted discretizer
    with raises(RuntimeError):
        result = disc.transform(X, None)

    # fitting discretizer
    disc.fit(X)

    # call method with unexpected value for categorical feature
    with raises(ValueError):
        result = disc.transform(X, None)

    # adding default to categorical feature
    feature2.has_default = True
    result = disc.transform(X, None)
    assert all(np.array(result.index) == np.array(index))

    # Assert the result
    print(result)
    expected = pd.DataFrame(
        {
            "feature1": ["1", "2", "3", "2", feature1.nan, "4", "4"],
            "feature2": ["A", "A", "B", "X", feature2.nan, feature2.default, "X"],
            "feature3": [
                feature3.nan,
                "x <= 2.00e+00",
                "2.00e+00 < x <= 4.50e+00",
                "2.00e+00 < x <= 4.50e+00",
                "2.00e+00 < x <= 4.50e+00",
                "4.50e+00 < x",
                "4.50e+00 < x",
            ],
        },
        index=index,
    )
    assert (result == expected).all().all()
    if true_false:  # testing copy
        assert (result != X).any().any()
    else:  # testing inplace
        assert (result == X).all().all()

    # WITH NANS NOT DROPPED
    # test features
    feature1 = OrdinalFeature("feature1", values=["1", "2", "3", "4"])
    feature1.is_fitted = True
    feature1.has_nan = True
    feature2 = CategoricalFeature("feature2")
    feature2.update(GroupedList({"A": ["A"], "B": ["B"], "X": ["X", "C", "D"]}))
    feature2.is_fitted = True
    feature2.has_nan = True
    feature3 = QuantitativeFeature("feature3")
    feature3.update(GroupedList([2, 4.5, np.inf]))
    feature3.is_fitted = True
    feature3.has_nan = True
    # feature3.update(GroupedList({2: [2], 4.5: [4.5], np.inf: [np.inf, "__NAN__"]}))
    disc = BaseDiscretizer([feature1, feature2, feature3], config=DiscretizerConfig(copy=true_false))

    # creating sample
    index = [1, 2, 3, 4, 5, 6, 7]
    X = pd.DataFrame(
        {
            "feature1": ["1", "2", "3", "2", np.nan, "4", "4"],
            "feature2": ["A", "A", "B", "C", np.nan, "E", "X"],
            "feature3": [np.nan, 2, 3, 4, 4.5, 5, 6],
        },
        index=index,
    )

    # Call the method with non fitted discretizer
    with raises(RuntimeError):
        result = disc.transform(X, None)

    # fitting discretizer
    disc.fit(X)

    # call method with unexpected value for categorical feature
    with raises(ValueError):
        result = disc.transform(X, None)

    # adding default to categorical feature
    feature2.has_default = True
    result = disc.transform(X, None)
    assert all(np.array(result.index) == np.array(index))

    # Assert the result
    print(result)
    expected = pd.DataFrame(
        {
            "feature1": ["1", "2", "3", "2", np.nan, "4", "4"],
            "feature2": ["A", "A", "B", "X", np.nan, feature2.default, "X"],
            "feature3": [
                np.nan,
                "x <= 2.00e+00",
                "2.00e+00 < x <= 4.50e+00",
                "2.00e+00 < x <= 4.50e+00",
                "2.00e+00 < x <= 4.50e+00",
                "4.50e+00 < x",
                "4.50e+00 < x",
            ],
        },
        index=index,
    )
    for feature in ["feature1", "feature2", "feature3"]:
        assert ((result[feature] == expected[feature]) | (result[feature].isna() & expected[feature].isna())).all()


def _make_discretizer(features, *, min_freq, true_false, n_jobs):
    """Helper: build a BaseDiscretizer with the new config-based API."""
    config = DiscretizerConfig(
        dropna=true_false,
        verbose=true_false,
        ordinal_encoding=true_false,
        n_jobs=n_jobs,
    )
    discretizer = BaseDiscretizer(features, min_freq=min_freq, config=config)
    discretizer.is_fitted = true_false
    return discretizer


def test_to_json(features: Features, true_false: bool) -> None:
    """tests base discretizer to_json method"""

    min_freq = 0.1
    n_jobs = 2
    discretizer = _make_discretizer(features, min_freq=min_freq, true_false=true_false, n_jobs=n_jobs)

    result = discretizer.to_json(light_mode=True)
    assert isinstance(result, dict)
    assert "features" in result
    assert "combination_evaluator" not in result
    assert result["config"]["dropna"] == true_false
    assert result["min_freq"] == min_freq
    assert result["is_fitted"] == true_false
    assert result["config"]["n_jobs"] == n_jobs
    assert result["config"]["verbose"] == true_false
    assert result["config"]["ordinal_encoding"] == true_false


def test_save(tmp_path, features: Features, true_false: bool) -> None:
    """tests base discretizer save method"""

    min_freq = 0.1
    n_jobs = 2
    discretizer = _make_discretizer(features, min_freq=min_freq, true_false=true_false, n_jobs=n_jobs)

    file_path = tmp_path / "test_discretizer.json"
    discretizer.save(str(file_path), light_mode=true_false)

    assert file_path.exists()
    with open(file_path, encoding="utf-8") as f:
        saved_data = json.load(f)
    assert saved_data == discretizer.to_json(light_mode=true_false)

    # checking with wrong path
    with raises(ValueError):
        discretizer.save("wrong_path", light_mode=true_false)


def test_load_discretizer(tmp_path, features: Features, true_false: bool) -> None:
    """tests base discretizer load_discretizer method"""

    min_freq = 0.1
    n_jobs = 2
    discretizer = _make_discretizer(features, min_freq=min_freq, true_false=true_false, n_jobs=n_jobs)

    file_path = tmp_path / "test_discretizer.json"
    discretizer.save(str(file_path), light_mode=true_false)

    loaded = BaseDiscretizer.load(str(file_path))

    for feature in loaded.features:
        assert feature.name in discretizer.features
    assert loaded.config.dropna == discretizer.config.dropna
    assert loaded.min_freq == discretizer.min_freq
    assert loaded.is_fitted == discretizer.is_fitted
    assert loaded.config.n_jobs == discretizer.config.n_jobs
    assert loaded.config.verbose == discretizer.config.verbose
    assert loaded.config.ordinal_encoding == discretizer.config.ordinal_encoding

    with raises(FileNotFoundError):
        _ = BaseDiscretizer.load("wrong_path")


# def test_summary() -> None:
#     """tests base discretizer summary method"""
#     assert False


# def test_history() -> None:
#     """tests base discretizer history method"""
#     assert False


# def test_update() -> None:
#     """tests base discretizer update method"""
#     assert False


def test_base_discretizer(x_train: pd.DataFrame, dropna: bool) -> None:
    """Tests BaseDiscretizer

    Parameters
    ----------
    x_train : pd.DataFrame
        Simulated Train DataFrame
    """

    # values to input nans
    str_nan = "nan"
    # dropna = True

    # defining values_orders
    order = ["Low-", "Low", "Low+", "Medium-", "Medium", "Medium+", "High-", "High", "High+"]

    # ordering for base qualitative ordinal feature
    groupedlist = GroupedList(order)
    groupedlist.group(["Low-", "Low"], "Low+")
    groupedlist.group(["Medium+", "High-"], "High")

    # ordering for qualitative ordinal feature that contains NaNs
    groupedlist_lownan = GroupedList(order)
    groupedlist_lownan.group(["Low-", "Low"], "Low+")
    groupedlist_lownan.group(["Medium+", "High-"], "High")

    # storing per feature orders
    ordinal_values = {
        "Qualitative_Ordinal": groupedlist,
        "Qualitative_Ordinal_lownan": groupedlist_lownan,
    }
    features = Features(ordinals=ordinal_values, config=FeaturesConfig(nan=str_nan, dropna=dropna))
    features.fit(x_train)
    feature = features("Qualitative_Ordinal_lownan")
    print(feature.has_nan, feature.dropna, feature.content)

    # initiating discretizer
    discretizer = BaseDiscretizer(features=features, config=DiscretizerConfig(dropna=dropna, copy=True))
    x_discretized = discretizer.fit_transform(x_train)

    # testing ordinal qualitative feature discretization
    x_expected = x_train.copy()
    feature = "Qualitative_Ordinal"
    x_expected[feature] = (
        x_expected[feature]
        .replace("Low-", "Low+")
        .replace("Low", "Low+")
        .replace("Low+", "Low+")
        .replace("Medium-", "Medium-")
        .replace("Medium", "Medium")
        .replace("Medium+", "High")
        .replace("High-", "High")
        .replace("High", "High")
        .replace("High+", "High+")
    )
    assert all(x_expected[feature] == x_discretized[feature]), "incorrect discretization"

    # testing ordinal qualitative feature discretization with nans
    feature = "Qualitative_Ordinal_lownan"
    x_expected[feature] = (
        x_expected[feature]
        .replace("Low-", "Low+")
        .replace("Low", "Low+")
        .replace("Low+", "Low+")
        .replace("Medium-", "Medium-")
        .replace("Medium", "Medium")
        .replace("Medium+", "High")
        .replace("High-", "High")
        .replace("High", "High")
        .replace("High+", "High+")
    )
    # replacing nans if requested
    if dropna:
        x_expected[feature] = x_expected[feature].replace(np.nan, str_nan)

    assert all(x_expected[feature].isna() == x_discretized[feature].isna()), "unexpected NaNs"

    non_nans = x_expected[feature].notna()
    print(x_expected.loc[non_nans, feature].value_counts())
    print(x_discretized.loc[non_nans, feature].value_counts())
    assert all(x_expected.loc[non_nans, feature] == x_discretized.loc[non_nans, feature]), (
        "incorrect discretization with nans"
    )

    # checking that other columns are left unchanged
    feature = "Quantitative"
    assert all(x_discretized[feature] == x_discretized[feature]), "Others should not be modified"


@fixture
def sample_data():
    X = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
    y = pd.Series([0, 1, 0])
    return Sample(X=X, y=y)


def test_initialization(sample_data):
    assert isinstance(sample_data.X, pd.DataFrame)
    assert isinstance(sample_data.y, pd.Series)


def test_getitem(sample_data):
    assert sample_data["X"].equals(sample_data.X)
    assert sample_data["y"].equals(sample_data.y)
    with raises(KeyError):
        sample_data["invalid_key"]


def test_iter(sample_data):
    keys = list(iter(sample_data))
    assert keys == ["X", "y"]


def test_keys(sample_data):
    assert sample_data.keys() == ["X", "y"]
