""" Defines a set of features"""

from pandas import DataFrame, Series

from .utils.base_feature import BaseFeature
from .utils.grouped_list import GroupedList
from .qualitative_features import CategoricalFeature, OrdinalFeature
from .quantitative_features import QuantitativeFeature

from numpy import nan
from typing import Union, Type
from copy import deepcopy

# class AutoFeatures(Features):
#     """TODO"""

#     __name__ = "AutoFeatures"

#     def __init__(self):
#         raise EnvironmentError(
#             f" - [{self.__name__}] Should be instantiated with AutoFeatures.from_dataframe()"
#         )

#     def from_dataframe(self, X: DataFrame) -> None:
#         """Automatically generates Features from an input DataFrame based on there data types"""
#         # initiating features
#         categoricals, ordinals, quantitatives, datetimes = ([],) * 4

#         # getting data types
#         for feature, dtype in X.dtypes:
#             str_dtype = str(dtype).lower()
#             # categorical feature
#             if dtype == "object":
#                 categoricals += [feature]
#             # quantitative feature
#             elif str_dtype.startswith("int") or str_dtype.startswith("float"):
#                 quantitatives += [feature]
#             # datetime feature
#             elif "date" in str_dtype or "time" in str_dtype:
#                 datetimes += [feature]
#             # unknown data type
#             else:
#                 warn(
#                     f" - [{self.__name__}] Ommited column {feature}, unknown data type {dtype}",
#                     UserWarning,
#                 )


# # def get_dtype()


# class MultiFeatures:
#     """TODO"""

#     def __init__(
#         self,
#         labels: list[str],
#         categoricals: list[str] = None,
#         quantitatives: list[str] = None,
#         ordinals: list[str] = None,
#         ordinal_values: dict[str, list[str]] = None,
#         **kwargs: dict,
#     ) -> None:
#         self.features = {
#             label: Features(categoricals, quantitatives, ordinals, ordinal_values, **kwargs)
#             for label in labels
#         }
#         self.labels = labels


class Features:

    __name__ = "Features"

    def __init__(
        self,
        categoricals: list[str] = None,
        quantitatives: list[str] = None,
        ordinals: list[str] = None,
        ordinal_values: dict[str, list[str]] = None,
        **kwargs: dict,
    ) -> None:
        # ordered values per ordinal features
        self.ordinal_values = ordinal_values

        # casting features accordingly
        self.categoricals = cast_features(categoricals, CategoricalFeature, **kwargs)
        self.quantitatives = cast_features(quantitatives, QuantitativeFeature, **kwargs)
        self.ordinals = cast_features(
            ordinals, OrdinalFeature, ordinal_values=self.ordinal_values, **kwargs
        )

        # ensuring features are grouped accordingly (already initiated features)
        all_features = self.categoricals + self.ordinals + self.quantitatives
        self.categoricals = [feature for feature in all_features if feature.is_categorical]
        self.ordinals = [feature for feature in all_features if feature.is_ordinal]
        self.quantitatives = [feature for feature in all_features if feature.is_quantitative]

        # checking that features were passed as input
        if len(self.categoricals) == 0 and len(self.quantitatives) == 0 and len(self.ordinals) == 0:
            raise ValueError(
                f" - [{self.__name__}] No feature passed as input. Please provide column names"
                "by setting categoricals, quantitatives or ordinals."
            )

        # checking that qualitatitve and quantitative features are distinct
        ordinal_names = get_versions(self.ordinals)  # unique version but non-unique names
        categorcial_names = get_versions(self.categoricals)
        quantitative_names = get_versions(self.quantitatives)
        if (
            any(feature in ordinal_names + quantitative_names for feature in categorcial_names)
            or any(feature in categorcial_names + ordinal_names for feature in quantitative_names)
            or any(feature in categorcial_names + quantitative_names for feature in ordinal_names)
        ):
            raise ValueError(
                f" - [{self.__name__}] One of provided features is in quantitatives and in "
                "categoricals and/or in ordinals. Please, check inputs!"
            )

    def __repr__(self) -> str:
        """Returns names of all features"""
        return f"{self.__name__}({str(self.get_names())})"

    def __call__(self, feature_name: str) -> BaseFeature:
        """Returns specified feature by name"""
        # checking that feature exists among versions
        self_dict = self.to_dict()
        if feature_name in self_dict:
            return self_dict.get(feature_name)

        # not found feature
        raise ValueError(f" - [{self.__name__}] '{feature_name}' not in features.")

    def __len__(self) -> int:
        """Returns number of features"""
        return len(self.to_list())

    def __iter__(self):
        """Returns an iterator of all features"""
        return iter(self.to_list())

    def __getitem__(
        self, index: Union[int, str, list[int], list[str]]
    ) -> Union[BaseFeature, list[BaseFeature]]:
        """Get item by index in list of features, by feature name or with a list of
        indices/feature names
        """
        # list index request
        if isinstance(index, int):
            return self.to_list()[index]

        # feature name request
        if isinstance(index, str):
            return self(index)

        # list request and element to search for
        if isinstance(index, list) and len(index) > 0:

            # list index request
            if isinstance(index[0], int):
                self_list = self.to_list()
                return [self_list[idx] for idx in index]

            # feature name request
            if isinstance(index[0], str):
                return [self(name) for name in index]

        return None

    def get_names(self) -> list[str]:
        """Returns names of all features"""
        return get_names(self.to_list())

    def get_versions(self) -> list[str]:
        """Returns names of all features"""
        return get_versions(self.to_list())

    def remove(self, feature_name: str) -> None:
        """Removes a feature by name"""

        # removing from list of typed features
        self.categoricals = [
            feature for feature in self.categoricals if feature.name != feature_name
        ]
        self.ordinals = [feature for feature in self.ordinals if feature.name != feature_name]
        self.quantitatives = [
            feature for feature in self.quantitatives if feature.name != feature_name
        ]

    def keep(self, kept_features: list[str]):
        """list of features to keep (removes the others)"""

        # keeping from list of typed features
        self.categoricals = [
            feature for feature in self.categoricals if feature.name in kept_features
        ]
        self.ordinals = [feature for feature in self.ordinals if feature.name in kept_features]
        self.quantitatives = [
            feature for feature in self.quantitatives if feature.name in kept_features
        ]

    def check_values(self, X: DataFrame) -> None:
        """Cheks for unexpected values for each feature in columns of DataFrame X"""
        # iterating over all features
        for feature in self:

            # checking for non-fitted features
            if not feature.is_fitted:
                raise RuntimeError(f" - [{self.__name__}] '{feature}' not yet fitted!")

            # checking for unexpected values
            feature.check_values(X)

    def fit(self, X: DataFrame, y: Series = None) -> None:
        """fits all features to there respective column in DataFrame X"""
        # iterating over all features
        for feature in self:

            # checking for fitted features
            if feature.is_fitted:
                feature.check_values(X)

            # fitting feature
            else:
                feature.fit(X, y)

    def fillna(self, X: DataFrame) -> DataFrame:
        """fills nans of a DataFrame"""

        # fills features with nans when dropna is True
        X.fillna(
            {
                feature.version: feature.nan
                for feature in self
                if feature.has_nan and feature.dropna
            },
            inplace=True,
        )

        return X

    def unfillna(self, X: DataFrame) -> DataFrame:
        """unfills nans when not supposed to have filled them"""

        # reinstating nans of features for which nans should not have been dropped
        X.replace(
            {
                feature.version: {feature.nan: nan}
                for feature in self
                if feature.has_nan and not feature.dropna
            },
            inplace=True,
        )

        return X

    def update(
        self,
        feature_values: dict[str, GroupedList],
        convert_labels: bool = False,
        sorted_values: bool = False,
        replace: bool = False,
    ) -> None:
        """Updates all features using provided feature_values"""
        # iterating over each feature
        for feature, values in feature_values.items():
            self(feature).update(values, convert_labels, sorted_values, replace)

    def update_labels(self, ordinal_encoding: bool = False) -> None:
        """Updates all feature labels"""
        # iterating over each feature
        for feature in self:
            feature.update_labels(ordinal_encoding=ordinal_encoding)

    def get_qualitatives(self, return_names: bool = False) -> list[BaseFeature]:
        """Returns all qualitative features"""
        # returning feature names
        if return_names:
            return get_versions(self.categoricals + self.ordinals)

        # returning feature objects
        return self.categoricals + self.ordinals

    def get_quantitatives(self, return_names: bool = False) -> list[BaseFeature]:
        """Returns all quantitative features"""
        # returning feature names
        if return_names:
            return get_versions(self.quantitatives)

        # returning feature objects
        return self.quantitatives

    def get_ordinals(self, return_names: bool = False) -> list[BaseFeature]:
        """Returns all ordinal features"""
        # returning feature names
        if return_names:
            return get_versions(self.ordinals)

        # returning feature objects
        return self.ordinals

    def get_categoricals(self, return_names: bool = False) -> list[BaseFeature]:
        """Returns all categorical features"""
        # returning feature names
        if return_names:
            return get_versions(self.categoricals)

        # returning feature objects
        return self.categoricals

    def set_dropna(self, dropna: bool = True) -> None:
        """Sets feature in dropna mode"""
        # iterating over each feature
        for feature in self:
            feature.set_dropna(dropna)

    def get_content(self, feature: str = None) -> dict:
        """Returns per feature content"""
        # returning specific feature's content
        if feature is not None:
            return self(feature).get_content()

        # returning all features' content
        return {feature.version: feature.get_content() for feature in self}

    def to_json(self, light_mode: bool = False) -> dict:
        """Converts a feature to JSON format"""
        return {feature.name: feature.to_json(light_mode) for feature in self}

    def to_list(self) -> list[BaseFeature]:
        """Returns a list of all features"""
        return self.categoricals + self.ordinals + self.quantitatives

    def to_dict(self) -> dict[str, BaseFeature]:
        """Returns a dict of all versionned features"""
        return {feature.version: feature for feature in self.to_list()}

    @classmethod
    def load(cls: Type["Features"], features_json: dict, ordinal_encoding: bool) -> "Features":
        """Loads a set of features"""

        # casting each feature to there corresponding type
        unpacked_features: list[BaseFeature] = []
        for _, feature in features_json.items():
            # categorical feature
            if feature.get("is_categorical"):
                unpacked_features += [CategoricalFeature.load(feature, ordinal_encoding)]
            # ordinal feature
            elif feature.get("is_ordinal"):
                unpacked_features += [OrdinalFeature.load(feature, ordinal_encoding)]
            # ordinal feature
            elif feature.get("is_quantitative"):
                unpacked_features += [QuantitativeFeature.load(feature, ordinal_encoding)]

        # initiating features
        return cls(unpacked_features)

    def get_summaries(self):
        """returns summaries of features' values' content"""
        # iterating over each feature
        summaries = []
        for feature in self:
            summaries += feature.get_summary()

        return DataFrame(summaries).set_index(["feature", "label"])

    def get_multiclass_features(self, y_classes: list[str]) -> "Features":
        """Returns multiclass version of accordingly renamed features"""

        return {
            y_class: Features([feature.rename(f"{feature.name}_y={y_class}") for feature in self])
            for y_class in y_classes
        }


def cast_features(
    features: list[str],
    target_class: type = BaseFeature,
    ordinal_values: dict[str, list[str]] = None,
    **kwargs: dict,
) -> list[BaseFeature]:
    """converts a list of string feature names to there corresponding Feature class"""

    # inititating features if not provided
    if features is None:
        features: list[str] = []

    # inititating ordered values for ordinal features if not provided
    if ordinal_values is None:
        ordinal_values: dict[str, list[str]] = {}

    # initiating list of converted features
    converted_features: list[target_class] = []

    # iterating over each feature
    for feature in features:
        # string case, initiating feature
        if isinstance(feature, str):
            converted_features += [
                target_class(feature, values=ordinal_values.get(feature), **kwargs)
            ]
        # already a BaseFeature
        elif isinstance(feature, BaseFeature):
            converted_features += [feature]

        else:
            raise TypeError(
                f" - [Features] feature {feature} is neither a str, nor a {target_class.__name__}."
            )

    # deduplicating features by name
    return [
        feature
        for n, feature in enumerate(converted_features)
        if feature.name not in get_names(converted_features[n + 1 :])
    ]


def get_names(features: list[BaseFeature]) -> list[str]:
    """Gives names from Features"""
    return [feature.name for feature in features]


def get_versions(features: list[BaseFeature]) -> list[str]:
    """Gives version names from Features"""
    return [feature.version for feature in features]


class MulticlassFeatures:

    def __init__(self, features: Features, y_classes: list[str]) -> None:

        self.raw_features = deepcopy(features)
        self.y_classes = y_classes
        self.features = {y_class: deepcopy(features) for y_class in self.y_classes}
