"""Set of tests for base_discretizers module."""

from numpy import inf, nan
from pandas import DataFrame, Series
from pytest import FixtureRequest, fixture, raises

from AutoCarver.discretizers.utils.base_discretizer import (
    BaseDiscretizer,
    get_bool_attribute,
    transform_quantitative_feature,
)
from AutoCarver.features import (
    CategoricalFeature,
    Features,
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
    feature1 = QuantitativeFeature("feature1")
    feature2 = OrdinalFeature("feature2", values=["1", "2", "3", "4"])
    feature3 = CategoricalFeature("feature3")
    return Features([feature1, feature2, feature3])


@fixture(params=[True, False])
def true_false(request: FixtureRequest) -> bool:
    """true or false"""
    return request.param


def test_get_bool_attribute() -> None:
    """test get_bool_attribute checks"""

    kwargs = {"true": True, "false": False, "wrong": "value"}
    with raises(ValueError):
        get_bool_attribute(kwargs, "wrong", True)
    assert get_bool_attribute(kwargs, "true", True)
    assert get_bool_attribute(kwargs, "true", False)
    assert not get_bool_attribute(kwargs, "false", True)
    assert not get_bool_attribute(kwargs, "false", False)
    assert get_bool_attribute(kwargs, "missing", True)
    assert not get_bool_attribute(kwargs, "missing", False)


def test_transform_quantitative_feature(features: Features) -> None:
    """test function transform_quantitative_feature"""

    # with values to group, without nan
    feature = features[-1]
    feature.update(GroupedList([2, 4.5, inf]))

    df_feature = Series([1, 2, 3, 4, 4.5, 5], name=feature.version)
    feature_version, list_feature = transform_quantitative_feature(
        feature, df_feature, len(df_feature)
    )
    assert feature_version == feature.version
    assert [
        "x <= 2.0e+00",
        "x <= 2.0e+00",
        "2.0e+00 < x <= 4.5e+00",
        "2.0e+00 < x <= 4.5e+00",
        "2.0e+00 < x <= 4.5e+00",
        "4.5e+00 < x",
    ] == list_feature

    # with values to group, with nan in df_feature
    feature = features[-1]
    feature.update(GroupedList([2, 4.5, inf]))

    df_feature = Series([1, 2, 3, 4, 4.5, nan, 5], name=feature.version)
    feature_version, list_feature = transform_quantitative_feature(
        feature, df_feature, len(df_feature)
    )
    assert feature_version == feature.version

    print(list_feature)
    print(feature.label_per_value)
    print(feature)
    assert [
        "x <= 2.0e+00",
        "x <= 2.0e+00",
        "2.0e+00 < x <= 4.5e+00",
        "2.0e+00 < x <= 4.5e+00",
        "2.0e+00 < x <= 4.5e+00",
        nan,
        "4.5e+00 < x",
    ] == list_feature

    # with values to group, with nan in df_feature and grouped nans
    feature = features[-1]
    feature.update(GroupedList([2, 4.5, inf]))

    df_feature = Series([1, 2, 3, 4, 4.5, nan, 5], name=feature.version)
    feature_version, list_feature = transform_quantitative_feature(
        feature, df_feature, len(df_feature)
    )
    assert feature_version == feature.version

    print(list_feature)
    print(feature.label_per_value)
    print(feature)
    assert [
        "x <= 2.0e+00",
        "x <= 2.0e+00",
        "2.0e+00 < x <= 4.5e+00",
        "2.0e+00 < x <= 4.5e+00",
        "2.0e+00 < x <= 4.5e+00",
        nan,
        "4.5e+00 < x",
    ] == list_feature
    assert False


def test_init(features: Features, true_false: bool) -> None:
    """test init method"""

    # test features
    disc = BaseDiscretizer(features.to_list())
    assert disc.features.to_list() == features.to_list()
    disc = BaseDiscretizer(features)
    assert disc.features.to_list() == features.to_list()

    # test default values
    assert not disc.is_fitted
    assert not disc.dropna
    assert disc.copy
    assert disc.verbose
    assert not disc.ordinal_encoding
    assert disc.n_jobs == 1
    assert disc.min_freq is None
    assert disc.sort_by is None

    # test setting copy
    disc = BaseDiscretizer(features, copy=true_false)
    assert disc.copy == true_false

    # test setting ordinal_encoding
    disc = BaseDiscretizer(features, ordinal_encoding=true_false)
    assert disc.ordinal_encoding == true_false

    # test setting dropna
    disc = BaseDiscretizer(features, dropna=true_false)
    assert disc.dropna == true_false

    # test setting verbose
    disc = BaseDiscretizer(features, verbose=true_false)
    assert disc.verbose == true_false

    # test setting is_fitted
    disc = BaseDiscretizer(features, is_fitted=true_false)
    assert disc.is_fitted == true_false

    # test setting n_jobs
    disc = BaseDiscretizer(features, n_jobs=4)
    assert disc.n_jobs == 4

    # test setting min_freq
    disc = BaseDiscretizer(features, min_freq=0.2)
    assert disc.min_freq == 0.2

    # test setting sort_by
    disc = BaseDiscretizer(features, sort_by="test")
    assert disc.sort_by == "test"


def test_remove_feature(features: Features) -> None:
    """test remove_feature method"""

    disc = BaseDiscretizer(features)
    # existing feature
    disc._remove_feature("feature1")
    assert "feature1" not in disc.features
    # non-existing feature
    n_features = len(disc.features)
    disc._remove_feature("feature_x")
    assert len(disc.features) == n_features


def test_cast_features(features: Features) -> None:
    """test cast_features method"""

    disc = BaseDiscretizer(features)
    X = DataFrame(
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
    X = DataFrame(
        {
            "feature1": [1, 2, 3, 4],
            "feature2": ["1", "2", "3", "4"],
            "feature3": [1, 2, 3, 4],
            "feature4": [1, 2, 3, 4],
        }
    )

    # test with copy
    disc = BaseDiscretizer(features, copy=True)
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
    disc = BaseDiscretizer(features, copy=False)
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

    y = Series([1, 2, 3, 4])

    disc = BaseDiscretizer(features)
    disc._prepare_y(y)
    with raises(ValueError):
        disc._prepare_y(DataFrame(y))
    y = Series([nan, 2, 3, 4])
    with raises(ValueError):
        disc._prepare_y(y)
    y = Series([None, 2, 3, 4])
    with raises(ValueError):
        disc._prepare_y(y)


def test_prepare_data(features: Features) -> None:
    """test prepare_data method"""

    disc = BaseDiscretizer(features)
    X = DataFrame(
        {
            "feature1": [1, 2, 3, 4],
            "feature2": ["1", "2", "3", "4"],
            "feature3": [1, 2, 3, 4],
            "feature4": [1, 2, 3, 4],
        }
    )
    y = Series([0, 0, 0, 1])
    assert disc._prepare_data(None) is None
    disc._prepare_data(X)
    disc._prepare_data(X, y)

    # mismatched X and y
    y = Series([0, 0, 0, 1, 1])
    with raises(ValueError):
        disc._prepare_data(X, y)


def test_fit(features: Features) -> None:
    """tests fit method"""

    # already fitted discretizer
    disc = BaseDiscretizer(features, is_fitted=True)
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
    disc = BaseDiscretizer(features, ordinal_encoding=False)
    disc.fit()
    assert disc.is_fitted
    for feature in features:
        assert not feature.ordinal_encoding

    # expected use with ordinal_encoding
    disc = BaseDiscretizer(features, ordinal_encoding=True)
    disc.fit()
    assert disc.is_fitted
    for feature in features:
        assert feature.ordinal_encoding


def test_transform_qualitative() -> None:
    """tests base discretizer transform_qualitative method"""
    assert False


def test_transform_quantitative() -> None:
    """tests base discretizer transform_quantitative method"""
    assert False


def test_transform() -> None:
    """tests base discretizer transform method"""
    assert False


def test_to_json() -> None:
    """tests base discretizer to_json method"""
    assert False


def test_save() -> None:
    """tests base discretizer save method"""
    assert False


def test_load_discretizer() -> None:
    """tests base discretizer load_discretizer method"""
    assert False


def test_summary() -> None:
    """tests base discretizer summary method"""
    assert False


def test_history() -> None:
    """tests base discretizer history method"""
    assert False


def test_update() -> None:
    """tests base discretizer update method"""
    assert False


# TODO: test quantitative discretization
def test_base_discretizer(x_train: DataFrame, dropna: bool) -> None:
    """Tests BaseDiscretizer

    Parameters
    ----------
    x_train : DataFrame
        Simulated Train DataFrame
    """

    # values to input nans
    str_nan = "NAN"
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
    ordinals = ["Qualitative_Ordinal", "Qualitative_Ordinal_lownan"]
    features = Features(
        ordinals=ordinals, ordinal_values=ordinal_values, nan=str_nan, dropna=dropna
    )
    features.fit(x_train)
    feature = features("Qualitative_Ordinal_lownan")
    print(feature.has_nan, feature.dropna, feature.content)

    # initiating discretizer
    discretizer = BaseDiscretizer(features=features, dropna=dropna, copy=True)
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
        x_expected[feature] = x_expected[feature].replace(nan, str_nan)

    assert all(x_expected[feature].isna() == x_discretized[feature].isna()), "unexpected NaNs"

    non_nans = x_expected[feature].notna()
    print(x_expected.loc[non_nans, feature].value_counts())
    print(x_discretized.loc[non_nans, feature].value_counts())
    assert all(
        x_expected.loc[non_nans, feature] == x_discretized.loc[non_nans, feature]
    ), "incorrect discretization with nans"

    # checking that other columns are left unchanged
    feature = "Quantitative"
    assert all(x_discretized[feature] == x_discretized[feature]), "Others should not be modified"
