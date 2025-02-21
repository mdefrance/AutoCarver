"""Set of tests for quantitative_discretizers module."""

from numpy import nan
from pandas import DataFrame, Series

from AutoCarver.discretizers.dtypes.string_discretizer import StringDiscretizer, fit_feature
from AutoCarver.features import CategoricalFeature, Features, OrdinalFeature


def test_fit_feature() -> None:
    """test string discretizer fitting feature"""

    # without existing values
    feature = CategoricalFeature("feature1")
    df_feature = Series(["1", "2", 3.0, 3.1, 4, "6", nan, "6", 1, 2])

    version, order = fit_feature(feature, df_feature)

    assert version == feature.version
    assert order == ["6", "1", "2", "3", "3.1", "4"]
    assert order.content["6"] == ["6"]
    assert order.content["1"] == [1, "1"]
    assert order.content["2"] == [2, "2"]
    assert order.content["3"] == [3.0, "3"]
    assert order.content["3.1"] == [3.1, "3.1"]
    assert order.content["4"] == [4, "4"]

    # with existing values
    feature = OrdinalFeature("feature2", values=["1", "2", "3"])
    df_feature = Series(["1", "2", 3, 2.0, 1, nan])

    version, order = fit_feature(feature, df_feature)

    assert version == feature.version
    assert order == ["1", "2", "3"]
    assert order.content["1"] == [1, "1"]
    assert order.content["2"] == [2, "2"]
    assert order.content["3"] == [3, "3"]


def test_stringdiscretizer_initialization() -> None:
    """Tests the initialization of the StringDiscretizer class"""
    feature1 = CategoricalFeature("feature1")
    feature2 = OrdinalFeature("feature2", values=["1", "2", "3"])
    string_discretizer = StringDiscretizer([feature1, feature2])
    assert isinstance(string_discretizer.features, Features)
    assert string_discretizer.features["feature2"].values == ["1", "2", "3"]


def test_stringdiscretizer_fit() -> None:
    """Tests the fit method of the StringDiscretizer class"""
    feature1 = CategoricalFeature("feature1")
    feature2 = OrdinalFeature("feature2", values=["1", "2", "3"])
    string_discretizer = StringDiscretizer([feature1, feature2])

    X = DataFrame(
        {
            "feature1": ["1", "2", 3.0, 3.1, 4, "6", nan, "6", 1, 2],
            "feature2": ["1", "2", 3, 2.0, 1, nan, nan, "1", "2", "3"],
        }
    )

    # fitting the string discretizer
    string_discretizer.fit(X)

    assert string_discretizer.features["feature1"].values == ["6", "1", "2", "3", "3.1", "4"]
    assert string_discretizer.features["feature2"].values == ["1", "2", "3"]
    assert string_discretizer.features["feature1"].is_fitted
    assert string_discretizer.features["feature2"].is_fitted

    # transforming X
    transformed_x = string_discretizer.transform(X)

    assert transformed_x["feature1"].tolist() == [
        "1",
        "2",
        "3",
        "3.1",
        "4",
        "6",
        nan,
        "6",
        "1",
        "2",
    ]
    assert transformed_x["feature2"].tolist() == ["1", "2", "3", "2", "1", nan, nan, "1", "2", "3"]
