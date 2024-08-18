

# def test_base_feature_update_qualitative() -> None:
#     """test method update convert_labels for qualitative features"""

#     # testing convert_labels=False and feature.values=None
#     feature = BaseFeature(name="test_feature")
#     feature.update(GroupedList({"a": ["a", "b"], "c": ["c", "d"]}), convert_labels=False)
#     assert feature.values == ["a", "c"]
#     assert feature.get_content() == {"a": ["a", "b"], "c": ["c", "d"]}

#     # testing convert_labels=False and feature.values not None
#     with raises(ValueError):
#         feature.update(GroupedList({"e": ["f", "g"], "a": ["a", "b"]}), convert_labels=False)
#     feature.update(GroupedList({"a": ["a", "c"]}), convert_labels=False)
#     assert feature.values == ["a"]
#     assert feature.get_content() == {"a": ["c", "d", "a", "b"]}

#     # testing convert_labels=True and feature.values=None
#     feature = BaseFeature(name="test_feature")
#     with raises(AttributeError):  # only for already valued features
#         feature.update(GroupedList({"a": ["a", "b"], "c": ["c", "d"]}), convert_labels=True)

#     # testing convert_labels=True and feature.values not None without ordinal_encoding
#     feature = BaseFeature(name="test_feature")
#     ordinal_encoding = False
#     feature.values = GroupedList({"a": ["a", "b"], "c": ["c", "d"], "e": ["e", "f", "g"]})
#     feature.update_labels(ordinal_encoding=ordinal_encoding)  # setting up labels
#     feature.is_quantitative = False
#     feature.update(
#         GroupedList({"a": ["a", "c", "b"]}),
#         convert_labels=True,
#         ordinal_encoding=ordinal_encoding,
#     )
#     assert feature.values == ["a", "e"]
#     assert feature.get_content() == {"a": ["c", "d", "a", "b"], "e": ["e", "f", "g"]}
#     assert feature.labels == ["a", "e"]
#     assert feature.label_per_value == {
#         "a": "a",
#         "b": "a",
#         "c": "a",
#         "d": "a",
#         "e": "e",
#         "f": "e",
#         "g": "e",
#     }
#     assert feature.value_per_label == {"a": "a", "e": "e"}
#     # test updating nothing
#     feature.update(
#         GroupedList({"a": ["a", "c", "b"]}),
#         convert_labels=True,
#         ordinal_encoding=ordinal_encoding,
#     )
#     assert feature.values == ["a", "e"]
#     assert feature.get_content() == {"a": ["c", "d", "a", "b"], "e": ["e", "f", "g"]}
#     assert feature.labels == ["a", "e"]
#     assert feature.label_per_value == {
#         "a": "a",
#         "b": "a",
#         "c": "a",
#         "d": "a",
#         "e": "e",
#         "f": "e",
#         "g": "e",
#     }
#     assert feature.value_per_label == {"a": "a", "e": "e"}

#     # testing convert_labels=True and feature.values not None with ordinal_encoding
#     feature = BaseFeature(name="test_feature")
#     ordinal_encoding = True
#     feature.values = GroupedList({"a": ["a", "b"], "c": ["c", "d"], "e": ["e", "f", "g"]})
#     feature.update_labels(ordinal_encoding=ordinal_encoding)  # setting up labels
#     feature.is_quantitative = False
#     feature.update(
#         GroupedList({"a": ["a", "c", "b"]}),
#         convert_labels=True,
#         ordinal_encoding=ordinal_encoding,
#     )
#     assert feature.values == ["a", "e"]
#     assert feature.get_content() == {"a": ["c", "d", "a", "b"], "e": ["e", "f", "g"]}
#     assert feature.labels == [0, 1]
#     assert feature.label_per_value == {
#         "a": 0,
#         "b": 0,
#         "c": 0,
#         "d": 0,
#         "e": 1,
#         "f": 1,
#         "g": 1,
#     }
#     assert feature.value_per_label == {0: "a", 1: "e"}
