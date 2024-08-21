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
    QualitativeFeature,
    CategoricalFeature,
    OrdinalFeature,
    QuantitativeFeature,
)
from AutoCarver.features.features import (
    Features,
    make_versions,
    make_version,
    make_version_name,
    remove_version,
    keep_versions,
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

        new_feature = make_version(feature, y_class)

        assert new_feature.version_tag == y_class
        assert new_feature.version == make_version_name(new_feature.name, y_class)
        assert new_feature.name == feature.name
        assert isinstance(new_feature, type(feature))


def test_make_versions(mock_features):
    """test funtion make_versions"""
    y_classes = ["A", "B", "C"]
    features.ordinal_encoding = True

    initial_names = get_names(mock_features)

    new_features = make_versions(mock_features, y_classes)

    assert len(new_features) == len(mock_features) * len(y_classes)
    new_features_names = get_names(new_features)
    assert all(new_name in initial_names for new_name in new_features_names)
    new_features_versions = get_versions(new_features)
    assert all(
        make_version_name(f.name, y_class) in new_features_versions
        for f in mock_features
        for y_class in y_classes
    )
    assert (
        sum(isinstance(new_feature, QualitativeFeature) for new_feature in new_features)
        == len(y_classes) * 2  # ordinal features also are categorical
    )
    assert sum(isinstance(new_feature, CategoricalFeature) for new_feature in new_features) == len(
        y_classes
    )  # ordinal features also are categorical
    assert sum(isinstance(new_feature, OrdinalFeature) for new_feature in new_features) == len(
        y_classes
    )
    assert sum(isinstance(new_feature, QuantitativeFeature) for new_feature in new_features) == len(
        y_classes
    )
    for y_class in y_classes:
        assert sum(new_feature.version_tag == y_class for new_feature in new_features) == len(
            mock_features
        )


def test_get_names(mock_features):
    """test funtion get_names"""
    names = get_names(mock_features)
    assert names == ["feature1", "feature2", "feature3"]


def test_get_versions(mock_features):
    """test funtion get_versions"""
    versions = get_versions(mock_features)
    assert versions == ["feature1", "feature2", "feature3"]
    feature_versions = make_versions(mock_features, "A")
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


def test_remove_version(mock_features):
    """test function remove_version"""
    y_classes = ["A", "B", "C"]
    new_features = make_versions(mock_features, y_classes)

    # removing each feature version
    for feature in new_features:
        result = remove_version(feature.version, new_features)
        assert len(result) == len(mock_features) * len(y_classes) - 1
        assert all(f.version != feature.version for f in result)

    # Edge case: Remove a version that doesn't exist
    removed_version = "feature_D"
    result = remove_version(removed_version, new_features)
    assert len(result) == len(mock_features) * len(y_classes)


def test_keep_versions(mock_features):
    """test function keep_versions"""
    y_classes = ["A", "B", "C"]
    ordinal_encoding = False
    new_features = make_versions(mock_features, y_classes)

    for kept_version1 in new_features:
        for kept_version2 in new_features:
            result = keep_versions([kept_version1.version, kept_version2.version], new_features)
            assert len(result) == 1 + int(kept_version1.version != kept_version2.version)
            assert any(f.version == kept_version1.version for f in result)
            assert any(f.version == kept_version2.version for f in result)

    # Edge case: Keep a version that doesn't exist
    kept_versions = ["feature_D"]
    result = keep_versions(kept_versions, new_features)
    assert len(result) == 0


from pandas import DataFrame, Series


@fixture
def mock_categoricals():
    return [
        CategoricalFeature(name="cat1"),
        CategoricalFeature(name="cat2"),
    ]


@fixture
def mock_ordinals():
    return [
        OrdinalFeature(name="ord1", values=["low", "medium", "high"]),
        OrdinalFeature(name="ord2", values=["low", "medium", "high"]),
    ]


@fixture
def mock_quantitatives():
    return [
        QuantitativeFeature(name="quant1"),
        QuantitativeFeature(name="quant2"),
    ]


@fixture
def features(mock_categoricals, mock_ordinals, mock_quantitatives):
    return Features(
        categoricals=[f.name for f in mock_categoricals],
        quantitatives=[f.name for f in mock_quantitatives],
        ordinals=[f.name for f in mock_ordinals],
        ordinal_values={"ord1": ["low", "medium", "high"], "ord2": ["low", "medium", "high"]},
    )


def test_features_initialization(features, mock_categoricals, mock_ordinals, mock_quantitatives):
    assert len(features.categoricals) == len(mock_categoricals)
    assert len(features.ordinals) == len(mock_ordinals)
    assert len(features.quantitatives) == len(mock_quantitatives)


def test_features_call(features, mock_categoricals, mock_ordinals, mock_quantitatives):
    assert features("cat1") == features.categoricals[0]
    assert features("ord1") == features.ordinals[0]
    assert features("quant1") == features.quantitatives[0]

    with raises(ValueError):
        features("nonexistent")


def test_features_len(features):
    assert len(features) == len(features.categoricals) + len(features.ordinals) + len(
        features.quantitatives
    )


def test_features_iter(features):
    feature_list = list(iter(features))
    assert feature_list == features.to_list()


def test_features_getitem(features):
    # list mode
    assert features[0] == features.categoricals[0]
    with raises(IndexError):
        features[100]
    assert features[[0, -1]] == [features.categoricals[0], features.quantitatives[-1]]

    # dict mode
    assert features["cat1"] == features.categoricals[0]
    with raises(ValueError):
        features["nonexistent"]
    assert features[["cat1", "ord1"]] == [features("cat1"), features("ord1")]


def test_features_get_names(features):
    names = features.names
    expected_names = get_names(features)
    assert names == expected_names


def test_features_get_versions(features):
    versions = features.versions
    expected_versions = get_versions(features)
    assert versions == expected_versions


def test_features_remove_by_name(features):

    # removing a categorical feature by name
    features.remove("cat1")
    assert len(features.categoricals) == 1
    assert features.categoricals[0].name != "cat1"
    assert len(features) == 5

    # # removing a categorical feature by version
    # features.categoricals[0].version = "cat2_v2"
    # features.remove("cat2_v2")
    # assert len(features.categoricals) == 1
    # assert features.categoricals[0].version != "cat2_v2"
    # assert len(features) == 5

    # removing a ordinal feature by name
    features.remove("ord1")
    assert len(features.ordinals) == 1
    assert features.ordinals[0].name != "ord1"
    assert len(features) == 4

    # removing a quantitative feature by name
    features.remove("quant1")
    assert len(features.quantitatives) == 1
    assert features.quantitatives[0].name != "quant1"
    assert len(features) == 3


def test_features_remove_by_version(features):

    # removing a categorical feature by version
    features.categoricals[0].version = "cat1_v2"
    features.remove("cat1_v2")
    assert len(features.categoricals) == 1
    assert features.categoricals[0].version != "cat1_v2"
    assert len(features) == 5

    # removing a ordinal feature by version
    features.ordinals[0].version = "ord1_v2"
    features.remove("ord1_v2")
    assert len(features.ordinals) == 1
    assert features.ordinals[0].version != "ord1_v2"
    assert len(features) == 4

    # removing a quantitative feature by version
    features.quantitatives[0].version = "quant1_v2"
    features.remove("quant1_v2")
    assert len(features.quantitatives) == 1
    assert features.quantitatives[0].version != "quant1_v2"
    assert len(features) == 3


def test_features_keep(features):

    # keeping a categorical feature by name
    features_copy = Features(features)
    features_copy.keep(["cat1"])
    assert len(features_copy.categoricals) == 1
    assert features_copy.categoricals[0].name == "cat1"
    assert len(features_copy.ordinals) == 0
    assert len(features_copy.quantitatives) == 0

    # keeping a ordinal feature by name
    features_copy = Features(features)
    features_copy.keep(["ord1"])
    assert len(features_copy.ordinals) == 1
    assert features_copy.ordinals[0].name == "ord1"
    assert len(features_copy.categoricals) == 0
    assert len(features_copy.quantitatives) == 0

    # keeping a quantitative feature by name
    features_copy = Features(features)
    features_copy.keep(["quant1"])
    assert len(features_copy.quantitatives) == 1
    assert features_copy.quantitatives[0].name == "quant1"
    assert len(features_copy.categoricals) == 0
    assert len(features_copy.ordinals) == 0

    # keeping a categorical feature by version
    features_copy = Features(features)
    features_copy.categoricals[0].version = "cat1_v2"
    features_copy.keep(["cat1_v2"])
    assert len(features_copy.categoricals) == 1
    assert features_copy.categoricals[0].version == "cat1_v2"
    assert len(features_copy.ordinals) == 0
    assert len(features_copy.quantitatives) == 0

    # keeping a ordinal feature by version
    features_copy = Features(features)
    features_copy.ordinals[0].version = "ord1_v2"
    features_copy.keep(["ord1_v2"])
    assert len(features_copy.ordinals) == 1
    assert features_copy.ordinals[0].version == "ord1_v2"
    assert len(features_copy.categoricals) == 0
    assert len(features_copy.quantitatives) == 0

    # keeping a quantitative feature by version
    features_copy = Features(features)
    features_copy.quantitatives[0].version = "quant1_v2"
    features_copy.keep(["quant1_v2"])
    assert len(features_copy.quantitatives) == 1
    assert features_copy.quantitatives[0].version == "quant1_v2"
    assert len(features_copy.categoricals) == 0
    assert len(features_copy.ordinals) == 0

    # Edge case, keep unexpected version
    features_copy = Features(features)
    features_copy.keep(["unexpected"])
    assert len(features_copy.ordinals) == 0
    assert len(features_copy.categoricals) == 0
    assert len(features_copy.ordinals) == 0


# def test_features_check_values(features):
#     X = DataFrame({"cat1": ["a", "b"], "ord1": ["low", "medium"], "quant1": [1, 2]})

#     with raises(RuntimeError):
#         features.check_values(X)

#     for feature in features:
#         feature.is_fitted = True

#     features.check_values(X)


# def test_features_fit(features):
#     X = DataFrame({"cat1": ["a", "b"], "ord1": ["low", "medium"], "quant1": [1, 2]})

#     for feature in features:
#         feature.is_fitted = False

#     features.fit(X)

#     for feature in features:
#         assert feature.is_fitted is True


# def test_features_fillna(features):
#     X = DataFrame({"cat1": ["a", None], "ord1": ["low", None], "quant1": [1, None]})

#     for feature in features:
#         feature.has_nan = True
#         feature.dropna = True
#         feature.nan = -1

#     filled_X = features.fillna(X)
#     assert filled_X["cat1"].iloc[1] == -1
#     assert filled_X["ord1"].iloc[1] == -1
#     assert filled_X["quant1"].iloc[1] == -1


# def test_features_unfillna(features):
#     X = DataFrame({"cat1": ["a", -1], "ord1": ["low", -1], "quant1": [1, -1]})

#     for feature in features:
#         feature.has_nan = True
#         feature.dropna = False
#         feature.nan = -1
#         feature.label_per_value = {-1: "NA"}

#     unfilled_X = features.unfillna(X)
#     assert unfilled_X["cat1"].iloc[1] == pytest.approx(float("nan"))
#     assert unfilled_X["ord1"].iloc[1] == pytest.approx(float("nan"))
#     assert unfilled_X["quant1"].iloc[1] == pytest.approx(float("nan"))


# def test_features_update(features):
#     mock_values = {
#         "cat1": GroupedList(content={"a": ["a1", "a2"]}),
#         "ord1": GroupedList(content={"low": ["low1", "low2"]}),
#     }

#     for feature in features:
#         feature.update = lambda *args, **kwargs: None

#     features.update(mock_values)

#     for feature in features.categoricals + features.ordinals:
#         assert feature.version in mock_values


# def test_features_update_labels(features):
#     for feature in features:
#         feature.update_labels = lambda *args, **kwargs: None

#     features.update_labels()

#     for feature in features:
#         assert hasattr(feature, "labels")


def test_features_get_qualitatives(features):
    qualitatives = features.qualitatives
    assert len(qualitatives) == 4
    assert all(isinstance(feature, QualitativeFeature) for feature in qualitatives)


def test_features_get_quantitatives(features):
    quantitatives = features.quantitatives
    assert len(quantitatives) == 2
    assert all(isinstance(feature, QuantitativeFeature) for feature in quantitatives)


def test_features_get_ordinals(features):
    ordinals = features.ordinals
    assert len(ordinals) == 2
    assert all(isinstance(feature, OrdinalFeature) for feature in ordinals)


def test_features_get_categoricals(features):
    categoricals = features.categoricals
    assert len(categoricals) == 2
    assert all(isinstance(feature, CategoricalFeature) for feature in categoricals)


# def test_features_set_dropna(features):
#     features.set_dropna(True)
#     for feature in features:
#         assert feature.dropna is True


# def test_features_get_content(features):
#     content = features.get_content()
#     assert isinstance(content, dict)
#     assert all(isinstance(value, dict) for value in content.values())


# def test_features_to_json(features):
#     json_data = features.to_json()
#     assert isinstance(json_data, dict)
#     assert all(isinstance(value, dict) for value in json_data.values())


# def test_features_to_list(features):
#     feature_list = features.to_list()
#     assert isinstance(feature_list, list)
#     assert len(feature_list) == len(features.categoricals) + len(features.ordinals) + len(
#         features.quantitatives
#     )


# def test_features_to_dict(features):
#     feature_dict = features.to_dict()
#     assert isinstance(feature_dict, dict)
#     assert len(feature_dict) == len(features.categoricals) + len(features.ordinals) + len(
#         features.quantitatives
#     )


# def test_features_load(features):
#     json_data = features.to_json()
#     loaded_features = Features.load(json_data, ordinal_encoding=False)
#     assert isinstance(loaded_features, Features)
#     assert len(loaded_features) == len(features)


# def test_features_get_summaries(features):
#     summaries = features.get_summaries()
#     assert isinstance(summaries, DataFrame)
#     assert "feature" in summaries.columns
#     assert "label" in summaries.columns


# def test_features_add_feature_versions(features):
#     features.add_feature_versions(["classA", "classB"], ordinal_encoding=False)
#     assert len(features.categoricals) > 0
#     assert len(features.ordinals) > 0
#     assert len(features.quantitatives) > 0


# def test_features_get_version_group(features):
#     group = features.get_version_group("classA")
#     assert isinstance(group, list)
#     assert all(feature.version_tag == "classA" for feature in group)
