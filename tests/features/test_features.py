"""Set of tests for features module.
"""

# from pytest import raises

# from AutoCarver.features import Features, GroupedList


# def _features(
#     quantitative_features: list[str],
#     qualitative_features: list[str],
#     ordinal_features: list[str],
#     values_orders: dict[str, list[str]],
# ) -> None:
#     """Tests Features

#     Parameters
#     ----------
#     quantitative_features : list[str]
#         List of quantitative raw features
#     qualitative_features : list[str]
#         List of qualitative raw features
#     ordinal_features : list[str]
#         List of ordinal raw features
#     values_orders : dict[str, list[str]]
#         values_orders of raw features
#     """
#     features = Features(
#         categoricals=qualitative_features,
#         quantitatives=quantitative_features,
#         ordinals=ordinal_features,
#         ordinal_values=values_orders,
#     )
#     # checking for initiation of ordinal features
#     assert len(features(ordinal_features[0]).values) > 0, "non initiated ordinal values"
#     assert (
#         "High+" in features("Qualitative_Ordinal").label_per_value
#     ), "non initiated ordinal labels"

#     # checking for updates of values
#     features.update(
#         {"Qualitative_Ordinal": GroupedList(["High-", "High", "High+", "High++++"])}, replace=True
#     )
#     assert (
#         features.ordinals[0].values == features("Qualitative_Ordinal").values
#     ), "reference issue, not same Feature object"
#     assert (
#         "High++++" in features("Qualitative_Ordinal").values
#     ), "reference issue, not same Feature object"

#     # checking that an ordinal feature needs its values
#     with raises(ValueError):
#         Features(ordinals=["test"])

#     # checking that a feature can not be both ordinal and categorical
#     with raises(ValueError):
#         Features(categoricals=["test"], ordinals=["test"], ordinal_values={"test": ["test2"]})


from pytest import fixture, raises
from AutoCarver.features import (
    BaseFeature,
    CategoricalFeature,
    OrdinalFeature,
    QuantitativeFeature,
)
from AutoCarver.features.features import (
    make_versions,
    make_version,
    make_version_name,
    cast_features,
    get_names,
    get_versions,
)  # Replace 'your_module' with the actual module name


@fixture
def mock_features():
    # Replace these with actual mock or real features as needed
    return [
        CategoricalFeature("feature1"),
        OrdinalFeature("feature2", values=["1", "2", "3", "4", "5", "a", "b", "c", "d", "e", "f"]),
        QuantitativeFeature("feature3"),
    ]


def test_make_version_name():
    """test funtion make_version_name"""
    feature_name = "feature"
    y_class = "classA"
    expected = "feature__y=classA"
    assert make_version_name(feature_name, y_class) == expected


def test_make_version(mock_features):
    """test funtion make_version"""
    for feature in mock_features:
        y_class = "classB"
        ordinal_encoding = False

        new_feature = make_version(feature, y_class, ordinal_encoding)

        assert new_feature.version_tag == y_class
        assert new_feature.version == make_version_name(new_feature.name, y_class)
        assert new_feature.name == feature.name
        assert isinstance(new_feature, type(feature))


def test_make_versions(mock_features):
    """test funtion make_versions"""
    y_class = "classB"
    ordinal_encoding = True

    new_features = make_versions(mock_features, y_class, ordinal_encoding)

    assert len(new_features) == len(mock_features)
    assert all(f.version_tag == y_class for f in new_features)
    assert all(f.version == make_version_name(f.name, y_class) for f in new_features)
    assert isinstance(new_features[0], CategoricalFeature)
    assert isinstance(new_features[1], OrdinalFeature)
    assert isinstance(new_features[2], QuantitativeFeature)


def test_get_names(mock_features):
    """test funtion get_names"""
    names = get_names(mock_features)
    assert names == ["feature1", "feature2", "feature3"]


def test_get_versions(mock_features):
    """test funtion get_versions"""
    versions = get_versions(mock_features)
    assert versions == ["feature1", "feature2", "feature3"]
    feature_versions = make_versions(mock_features, "A", False)
    versions = get_versions(feature_versions)
    assert versions == ["feature1__y=A", "feature2__y=A", "feature3__y=A"]


def test_cast_features(mock_features):
    """test funtion cast_features"""

    # ordinal with strings
    feature_names = ["feature1", "feature2"]
    ordinal_values = {"feature1": ["low", "medium", "high"], "feature2": ["low", "medium", "high"]}
    casted_features = cast_features(
        feature_names, target_class=OrdinalFeature, ordinal_values=ordinal_values
    )
    assert len(casted_features) == 2
    assert isinstance(casted_features[0], OrdinalFeature)
    assert isinstance(casted_features[1], OrdinalFeature)
    assert casted_features[0].values == ["low", "medium", "high"]
    assert casted_features[1].values == ["low", "medium", "high"]

    # categorical with strings
    feature_names = ["feature1", "feature2"]
    casted_features = cast_features(feature_names, target_class=CategoricalFeature)
    assert len(casted_features) == 2
    assert isinstance(casted_features[0], CategoricalFeature)
    assert isinstance(casted_features[1], CategoricalFeature)

    # categorical with strings
    feature_names = ["feature1", "feature2"]
    casted_features = cast_features(feature_names, target_class=QuantitativeFeature)
    assert len(casted_features) == 2
    assert isinstance(casted_features[0], QuantitativeFeature)
    assert isinstance(casted_features[1], QuantitativeFeature)

    # with existing features
    casted_features = cast_features(mock_features)

    assert len(casted_features) == len(mock_features)

    assert all(isinstance(f, BaseFeature) for f in casted_features)
    assert casted_features[0].name == mock_features[0].name
    assert isinstance(casted_features[0], CategoricalFeature)
    assert isinstance(casted_features[1], OrdinalFeature)
    assert isinstance(casted_features[2], QuantitativeFeature)

    # with invalid type
    with raises(TypeError):
        cast_features([123], target_class=BaseFeature)

    # deduplication
    feature1 = QuantitativeFeature(name="feature1")
    feature2 = QuantitativeFeature(name="feature1")
    feature3 = QuantitativeFeature(name="feature1")
    feature1.version = "feature1_v0"

    casted_features = cast_features(
        [feature1, feature2, feature3], target_class=QuantitativeFeature
    )

    assert len(casted_features) == 2
    assert feature1.version in get_versions(casted_features)
