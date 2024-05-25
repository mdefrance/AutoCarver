""" Defines a set of features"""

from pandas import DataFrame, Series

from .utils.base_feature import BaseFeature
from .utils.grouped_list import GroupedList
from .qualitative_features import CategoricalFeature, OrdinalFeature
from .quantitative_features import QuantitativeFeature

from numpy import nan
from typing import Union, Type


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
                " - [Features] No feature passed as input. Please provided column names"
                "by setting categoricals, quantitatives or ordinals."
            )

        # checking that qualitatitve and quantitative features are distinct
        if (
            any(
                feature in get_names(self.ordinals) + get_names(self.quantitatives)
                for feature in get_names(self.categoricals)
            )
            or any(
                feature in get_names(self.categoricals) + get_names(self.ordinals)
                for feature in get_names(self.quantitatives)
            )
            or any(
                feature in get_names(self.categoricals) + get_names(self.quantitatives)
                for feature in get_names(self.ordinals)
            )
        ):
            raise ValueError(
                " - [AutoCarver] One of provided features is in quantitatives and in "
                "categoricals and/or in ordinals. Please, check inputs!"
            )

    def __repr__(self) -> str:
        # TODO add feature types
        # f'get_names(self.categoricals) + get_names(self.ordinals) + get_names(self.quantitatives)
        return f"Features({str(self.get_names())})"

    def __call__(self, feature_name: str) -> BaseFeature:
        # checking that feature exists
        if feature_name in self.to_dict():
            return self.to_dict().get(feature_name)
        else:
            raise ValueError(f" - [AutoCarver] '{feature_name}' not in features.")

    def get_names(self) -> list[str]:
        return (
            get_names(self.categoricals) + get_names(self.ordinals) + get_names(self.quantitatives)
        )

    def to_list(self) -> list[BaseFeature]:
        return self.categoricals + self.ordinals + self.quantitatives

    def __len__(self) -> int:
        return len(self.to_list())

    def to_dict(self) -> dict[str, BaseFeature]:
        return {feature.name: feature for feature in self.to_list()}

    def __iter__(self):
        return iter(self.to_list())

    def __getitem__(
        self, index: Union[int, str, list[int], list[str]]
    ) -> Union[BaseFeature, list[BaseFeature]]:
        # list index request
        if isinstance(index, int):
            return self.to_list()[index]
        # feature name request
        if isinstance(index, str):
            return self.to_dict().get(index)
        # list request
        if isinstance(index, list):
            # checking for element to search for
            if len(index) > 0:
                # list index request
                if isinstance(index[0], int):
                    self_list = self.to_list()
                    return [self_list[idx] for idx in index]
                # feature name request
                if isinstance(index[0], str):
                    self_dict = self.to_dict()
                    return [self_dict.get(name) for name in index]
        return None

    def remove(self, feature_name: str) -> None:
        self.categoricals = [feat for feat in self.categoricals if feat.name != feature_name]
        self.ordinals = [feat for feat in self.ordinals if feat.name != feature_name]
        self.quantitatives = [feat for feat in self.quantitatives if feat.name != feature_name]

    def keep_features(self, kept_features: list[str]):
        """list of features to keep (removes the others)"""
        self.categoricals = [
            feature for feature in self.categoricals if feature.name in kept_features
        ]
        self.ordinals = [feature for feature in self.ordinals if feature.name in kept_features]
        self.quantitatives = [
            feature for feature in self.quantitatives if feature.name in kept_features
        ]

    def check_values(self, X: DataFrame) -> None:
        """Cheks for unexpected values for each feature in columns of DataFrame X"""
        for feature in self:
            if not feature.is_fitted:  # checking for non-fitted features
                raise RuntimeError(f" - [Features] {feature} not yet fitted!")
            else:  # checking for unexpected values
                feature.check_values(X)

    def fit(self, X: DataFrame, y: Series = None) -> None:
        """fits all features to there respective column in DataFrame X"""
        for feature in self:
            if feature.is_fitted:  # checking for fitted features
                feature.check_values(X)
            else:  # fitting feature
                feature.fit(X, y)

    def fillna(self, X: DataFrame) -> DataFrame:
        """fills nans of a DataFrame"""

        # fills features with nans when dropna is True
        X.fillna(
            {feature.name: feature.nan for feature in self if feature.has_nan and feature.dropna},
            inplace=True,
        )

        return X

    def unfillna(self, X: DataFrame) -> DataFrame:
        """unfills nans when not supposed to have filled them"""

        # reinstating nans of features for which nans should not have been dropped
        X.replace(
            {
                feature.name: {feature.nan: nan}
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
        for feature, values in feature_values.items():
            self(feature).update(values, convert_labels, sorted_values, replace)

    def update_labels(self, ordinal_encoding: bool = False) -> None:
        for feature in self:
            feature.update_labels(ordinal_encoding=ordinal_encoding)

    def get_qualitatives(self, return_names: bool = False) -> list[BaseFeature]:
        if return_names:
            return get_names(self.categoricals + self.ordinals)
        return self.categoricals + self.ordinals

    def get_quantitatives(self, return_names: bool = False) -> list[BaseFeature]:
        if return_names:
            return get_names(self.quantitatives)
        return self.quantitatives

    def set_dropna(self, dropna: bool = True) -> None:
        """Sets feature in dropna mode"""
        for feature in self:
            feature.set_dropna(dropna)

    def get_content(self) -> dict:
        """Returns per feature content"""
        return {feature.name: feature.get_content() for feature in self}

    def to_json(self, light_mode: bool = False) -> dict:
        return {feature.name: feature.to_json(light_mode) for feature in self}

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
