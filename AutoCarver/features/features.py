"""Defines a set of features"""

from collections.abc import Sequence
from typing import TypeVar, overload

from numpy import nan
from pandas import DataFrame, Series

from AutoCarver.features.qualitatives import (
    CategoricalFeature,
    OrdinalFeature,
    QualitativeFeature,
    get_categorical_features,
    get_ordinal_features,
)
from AutoCarver.features.quantitatives import QuantitativeFeature, get_quantitative_features
from AutoCarver.features.utils.attributes import get_bool_attribute
from AutoCarver.features.utils.base_feature import BaseFeature
from AutoCarver.features.utils.grouped_list import GroupedList

# class AutoFeatures(Features):
#     """TODO"""

#     __name__ = "AutoFeatures"

#     def __init__(self):
#         raise EnvironmentError(
#             f"[{self.__name__}] Should be instantiated with AutoFeatures.from_dataframe()"
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
#                     f"[{self.__name__}] Ommited column {feature}, unknown data type {dtype}",
#                     UserWarning,
#                 )


@overload
def check_ordinal_features(ordinals: list[OrdinalFeature]) -> list[OrdinalFeature]: ...


@overload
def check_ordinal_features(ordinals: dict[str, list[str]]) -> list[str]: ...


def check_ordinal_features(ordinals: list[OrdinalFeature] | dict[str, list[str]]) -> list[OrdinalFeature] | list[str]:
    """Checks that ordinals are correctly formatted"""

    # checking for ordinal types
    if isinstance(ordinals, Features | list):
        for feature in ordinals:
            if not isinstance(feature, OrdinalFeature):
                raise TypeError("Ordinals should be a list of OrdinalFeature or a dict of ordinal values.")
        return ordinals
    elif isinstance(ordinals, dict):
        return list(ordinals.keys())
    raise TypeError("Ordinals should be a list of OrdinalFeature or a dict of ordinal values.")


class Features:
    """A set of typed features"""

    __name__ = "Features"

    def __init__(
        self,
        categoricals: list[CategoricalFeature] | None = None,
        quantitatives: list[QuantitativeFeature] | None = None,
        ordinals: list[OrdinalFeature] | None = None,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------

        categoricals : list[Union[CategoricalFeature, str]], optional
            List of categorical features or column names, by default ``None``

        quantitatives : list[Union[QuantitativeFeature, str]], optional
            List of quantitative features or column names, by default ``None``

        ordinals : Union[list[OrdinalFeature], dict[str, list[str]]], optional
            List of ordinal features or dict column names with associated value ordering,
            by default ``None``


        .. warning::
            At least one of categoricals, ordinals or quantitatives should be provided.


        Keyword Arguments
        -----------------

        ordinal_encoding : bool, optional
            Whether or not to ordinal encode labels, by default ``False``

        nan : str, optional
            Label for missing values, by default ``"__NAN__"``

        default : str, optional
            Label for default values, by default ``"__OTHER__"``
        """
        # ensuring features are grouped accordingly (already initiated features)
        self._categoricals = categoricals if categoricals is not None else []
        self._ordinals = ordinals if ordinals is not None else []
        self._quantitatives = quantitatives if quantitatives is not None else []

        # checking that features were passed as input
        if len(self.categoricals) == 0 and len(self.quantitatives) == 0 and len(self.ordinals) == 0:
            raise ValueError(
                f"[{self}] No feature passed as input. Please set quantitatives, categoricals or ordinals."
            )

        # checking that qualitatitve and quantitative features are distinct
        check_duplicate_features(self.ordinals, self.categoricals, self.quantitatives)

        self._dropna = False
        self._ordinal_encoding = False
        self.is_fitted = get_bool_attribute(kwargs, "is_fitted", False)

    @classmethod
    def from_str(
        cls,
        categoricals: list[str] | None = None,
        quantitatives: list[str] | None = None,
        ordinals: dict[str, list[str]] | None = None,
        **kwargs,
    ) -> "Features":
        """Initiates Features from string feature names only"""

        # initiating ordinal values if not provided
        if ordinals is None:
            ordinals = {}

        # getting list of ordinal features by name of BaseFeature
        ordinal_features = check_ordinal_features(ordinals)

        # casting features accordingly
        all_features: list[
            BaseFeature | QuantitativeFeature | QualitativeFeature | CategoricalFeature | OrdinalFeature
        ] = []
        all_features += cast_features(categoricals, CategoricalFeature, **kwargs)
        all_features += cast_features(quantitatives, QuantitativeFeature, **kwargs)
        all_features += cast_features(ordinal_features, OrdinalFeature, ordinal_values=ordinals, **kwargs)

        return cls.from_list(all_features)

    @classmethod
    def from_list(
        cls,
        features: list[BaseFeature | QuantitativeFeature | QualitativeFeature | CategoricalFeature | OrdinalFeature],
        **kwargs,
    ) -> "Features":
        """Initiates Features from a list of features only"""

        # ensuring features are grouped accordingly (already initiated features)
        categoricals = get_categorical_features(features)
        ordinals = get_ordinal_features(features)
        quantitatives = get_quantitative_features(features)

        return cls(
            categoricals=categoricals,
            ordinals=ordinals,
            quantitatives=quantitatives,
            **kwargs,
        )

    def __repr__(self) -> str:
        """Returns names of all features"""
        return f"{self.__name__}({str(self.versions)})"

    def __contains__(self, feature: BaseFeature | str) -> bool:
        """checks if a feature is in the features"""
        if isinstance(feature, BaseFeature):
            return feature.version in self.versions
        return feature in self.versions

    @overload
    def __call__(self, feature_name: str) -> BaseFeature: ...

    @overload
    def __call__(self, feature_name: DataFrame) -> list[str]: ...

    def __call__(self, feature_name: str | DataFrame) -> BaseFeature | list[str]:
        """Returns specified feature by name"""

        # case for dataframes
        if isinstance(feature_name, DataFrame):
            return [feature.version for feature in self if feature.version in feature_name.columns]

        # looking for feature names
        self_dict = self.to_dict()
        if feature_name in self_dict:
            return self_dict[feature_name]

        # looking for version names
        if feature_name in self:
            return next(feature for feature in self if feature.version == feature_name)

        # not found feature
        raise ValueError(f"[{self.__name__}] '{feature_name}' not in features.")

    def __len__(self) -> int:
        """Returns number of features"""
        return len(self.to_list())

    def __iter__(self):
        """Returns an iterator of all features"""
        return iter(self.to_list())

    @overload
    def __getitem__(self, index: int | str) -> BaseFeature | None: ...
    @overload
    def __getitem__(self, index: list[int] | list[str] | slice | DataFrame) -> Sequence[BaseFeature] | None: ...

    def __getitem__(
        self, index: int | str | list[int] | list[str] | slice | DataFrame
    ) -> BaseFeature | Sequence[BaseFeature] | None:
        """Get item by index in list of features, by feature name or with a list of
        indices/feature names
        """
        # list index/slice request
        if isinstance(index, int | slice):
            return self.to_list()[index]

        # feature name request
        if isinstance(index, str):
            return self(index)

        # dataframe request
        if isinstance(index, DataFrame):
            index = list(index.columns)

        # list request and element to search for
        if isinstance(index, list) and len(index) > 0:
            # list index request
            if all(isinstance(idx, int) for idx in index):
                self_list = self.to_list()
                return [self_list[idx] for idx in index if isinstance(idx, int)]

            # feature name request
            if all(isinstance(idx, str) for idx in index):
                return [self(name) for name in index if isinstance(name, str)]

        return None

    @property
    def names(self) -> list[str]:
        """Returns names of all features"""
        return get_names(self.to_list())

    @property
    def versions(self) -> list[str]:
        """Returns versions of all features"""
        return get_versions(self.to_list())

    @property
    def qualitatives(self) -> Sequence[QualitativeFeature]:
        """Returns all qualitative features"""
        return self.categoricals + self.ordinals

    @property
    def categoricals(self) -> list[CategoricalFeature]:
        """Returns all categorical features"""
        return self._categoricals

    @categoricals.setter
    def categoricals(self, values: list[CategoricalFeature]) -> None:
        """sets ordinal features"""

        if not all(isinstance(feature, CategoricalFeature) for feature in values):
            raise AttributeError(f"[{self}] Trying to set categorical feature with wrongly typed feature")
        self._categoricals = values

    @property
    def ordinals(self) -> list[OrdinalFeature]:
        """Returns all ordinal features"""
        return self._ordinals

    @ordinals.setter
    def ordinals(self, values: list[OrdinalFeature]) -> None:
        """sets ordinal features"""

        if not all(isinstance(feature, OrdinalFeature) for feature in values):
            raise AttributeError(f"[{self}] Trying to set ordinal feature with wrongly typed feature")
        self._ordinals = values

    @property
    def quantitatives(self) -> list[QuantitativeFeature]:
        """Returns all quantitative features"""
        return self._quantitatives

    @quantitatives.setter
    def quantitatives(self, values: list[QuantitativeFeature]) -> None:
        """sets quantitative features"""

        if not all(isinstance(feature, QuantitativeFeature) for feature in values):
            raise AttributeError(f"[{self}] Trying to set quantitative feature with wrongly typed feature")
        self._quantitatives = values

    @property
    def dropna(self) -> bool:
        """whether or not to drop missing values"""
        return self._dropna

    @dropna.setter
    def dropna(self, value: bool) -> None:
        """Sets features in dropna mode"""

        if not isinstance(value, bool):
            raise ValueError("Can only set dropna has a bool")

        for feature in self:  # iterating over each feature
            feature.dropna = value

        self._dropna = value

    @property
    def ordinal_encoding(self) -> bool:
        """whether or not to ordinal encode labels"""
        return self._ordinal_encoding

    @ordinal_encoding.setter
    def ordinal_encoding(self, value: bool) -> None:
        """Sets features in ordinal_encoding mode"""

        if not isinstance(value, bool):
            raise ValueError("Can only set ordinal_encoding has a bool")

        for feature in self:  # iterating over each feature
            feature.ordinal_encoding = value

        self._ordinal_encoding = value

    @property
    def content(self) -> dict:
        """Returns per feature content

        Returns
        -------
        dict
            per feature content
        """
        # returning all features' content
        return {feature.version: feature.content for feature in self}

    def remove(self, feature_version: str) -> None:
        """Removes a feature by version"""
        self.categoricals = remove_version(feature_version, self.categoricals)
        self.ordinals = remove_version(feature_version, self.ordinals)
        self.quantitatives = remove_version(feature_version, self.quantitatives)

    def keep(self, kept: list[str]) -> None:
        """list of features' versions to keep (removes the others)"""
        self.categoricals = keep_versions(kept, self.categoricals)
        self.ordinals = keep_versions(kept, self.ordinals)
        self.quantitatives = keep_versions(kept, self.quantitatives)

    def check_values(self, X: DataFrame) -> None:
        """Cheks for unexpected values for each feature in columns of DataFrame X"""
        # iterating over all features
        for feature in self:
            # checking for non-fitted features
            if not feature.is_fitted:
                raise RuntimeError(f"[{self.__name__}] '{feature}' not yet fitted!")

            # checking for unexpected values
            feature.check_values(X)

    def fit(self, X: DataFrame, y: Series | None = None) -> None:
        """fits all features to there respective column in DataFrame X"""
        # iterating over all features
        for feature in self:
            # checking for fitted features
            if feature.is_fitted:
                feature.check_values(X)

            # fitting feature
            else:
                feature.fit(X, y)

    def fillna(self, X: DataFrame, ignore_dropna: bool = False) -> DataFrame:
        """fills nans of a DataFrame"""

        # fills features with nans when dropna is True
        X.fillna(
            {feature.version: feature.nan for feature in self if feature.has_nan and (feature.dropna or ignore_dropna)},
            inplace=True,
        )

        return X

    def unfillna(self, X: DataFrame) -> DataFrame:
        """unfills nans when not supposed to have filled them"""

        # reinstating nans of features for which nans should not have been dropped
        X.replace(
            {
                feature.version: {feature.label_per_value.get(feature.nan, feature.nan): nan}
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
        for feature, values in feature_values.items():  # updating each features
            self(feature).update(values, convert_labels, sorted_values, replace)

    def update_labels(self) -> None:
        """Updates all feature labels"""
        for feature in self:  # updating each features
            feature.update_labels()

    def add_feature_versions(self, y_classes: list[str]) -> None:
        """Builds versions of all features for each y_class"""
        self.categoricals = make_versions(self.categoricals, y_classes)
        self.ordinals = make_versions(self.ordinals, y_classes)
        self.quantitatives = make_versions(self.quantitatives, y_classes)

    def get_version_group(self, y_class: str) -> list[BaseFeature]:
        """Returns all features with specified version_tag"""

        return [feature for feature in self if feature.version_tag == y_class]

    @property
    def summary(self) -> DataFrame:
        """Summary of discretization process for all features"""
        # iterating over each feature
        summaries = []
        for feature in self:
            summaries += feature.summary

        # converting to DataFrame
        summaries = DataFrame(summaries)

        # defining indices to set
        indices = []
        for col in summaries.columns:
            if col not in ["feature", "label", "content", "target_mean", "frequency"]:
                indices += [col]
        indices = ["feature"] + indices + ["label"]

        return summaries.set_index(indices)

    def to_json(self, light_mode: bool = False) -> dict:
        """Serializes :class:`Features` for JSON saving

        Parameters
        ----------
        light_mode : bool, optional
            Whether or not to serialize in light mode (without statistics and history),
            by default ``False``

        """
        features_json = {feature.version: feature.to_json(light_mode) for feature in self}
        features_json.update({"is_fitted": self.is_fitted})
        return features_json

    def to_list(self) -> Sequence[BaseFeature]:
        """Returns a list of all features"""
        return self.categoricals + self.ordinals + self.quantitatives

    def to_dict(self) -> dict[str, BaseFeature]:
        """Returns a dict of all versionned features"""
        return {feature.version: feature for feature in self.to_list()}

    @classmethod
    def load(cls, features_json: dict) -> "Features":
        """Allows one to load a set of :class:`Features`

        Parameters
        ----------
        features_json : dict
            Dictionary of serialized :class:`Features`

        Returns
        -------
        Features
            Loaded :class:`Features`.
        """

        # checking for fitted features
        is_fitted = features_json.pop("is_fitted", None)

        # casting each feature to there corresponding type
        unpacked_features: list[BaseFeature] = []
        for _, feature in features_json.items():
            # categorical feature
            if feature.get("is_categorical"):
                unpacked_features += [CategoricalFeature.load(feature)]

            # ordinal feature
            elif feature.get("is_ordinal"):
                unpacked_features += [OrdinalFeature.load(feature)]

            # ordinal feature
            elif feature.get("is_quantitative"):
                unpacked_features += [QuantitativeFeature.load(feature)]

        # initiating features
        return cls.from_list(unpacked_features, is_fitted=is_fitted)


FeatureIn = TypeVar("FeatureIn", bound=BaseFeature)
FeatureOut = TypeVar("FeatureOut", bound=BaseFeature, covariant=True)


def remove_version(removed_version: str, features: list[FeatureIn]) -> list[FeatureIn]:
    """removes a feature according its version"""
    return [feature for feature in features if feature.version != removed_version]


def keep_versions(kept_versions: list[str], features: list[FeatureIn]) -> list[FeatureIn]:
    """keeps requested feature versions according its version"""
    return [feature for feature in features if feature.version in kept_versions]


def make_versions(features: list[FeatureIn], y_classes: list[str]) -> list[FeatureIn]:
    """Makes a copy of a list of features with specified version"""
    return [make_version(feature, y_class) for y_class in y_classes for feature in features]


@overload
def make_version(feature: CategoricalFeature, y_class: str) -> CategoricalFeature: ...
@overload
def make_version(feature: OrdinalFeature, y_class: str) -> OrdinalFeature: ...
@overload
def make_version(feature: QuantitativeFeature, y_class: str) -> QuantitativeFeature: ...
@overload
def make_version(feature: FeatureIn, y_class: str) -> FeatureIn: ...


def make_version(
    feature: CategoricalFeature | OrdinalFeature | QuantitativeFeature | FeatureIn, y_class: str
) -> CategoricalFeature | OrdinalFeature | QuantitativeFeature | FeatureIn:
    """Makes a copy of a feature with specified version"""

    # converting feature to json
    feature_json = feature.to_json(light_mode=False)

    # categorical feature
    if feature_json.get("is_categorical"):
        cat_feature = CategoricalFeature.load(feature_json)
        cat_feature.version_tag = y_class  # modifying version and tag
        cat_feature.version = make_version_name(cat_feature.name, y_class)
        return cat_feature

    # ordinal feature
    elif feature_json.get("is_ordinal"):
        ord_feature = OrdinalFeature.load(feature_json)
        ord_feature.version_tag = y_class  # modifying version and tag
        ord_feature.version = make_version_name(ord_feature.name, y_class)
        return ord_feature

    # ordinal feature
    elif feature_json.get("is_quantitative"):
        quant_feature = QuantitativeFeature.load(feature_json)
        quant_feature.version_tag = y_class  # modifying version and tag
        quant_feature.version = make_version_name(quant_feature.name, y_class)
        return quant_feature
    # base feature
    raise TypeError(f"feature {feature} is not a known feature type.")


def make_version_name(feature_name: str, y_class: str) -> str:
    """Builds a version name for a feature and target class"""

    return f"{feature_name}__y={y_class}"


def cast_features(
    features: list[str] | None,
    target_class: type[FeatureIn],
    ordinal_values: dict[str, list[str]] | None = None,
    **kwargs,
) -> list[FeatureIn]:
    """converts a list of string feature names to there corresponding Feature class"""

    # inititating features if not provided
    if features is None:
        features = []

    # inititating ordered values for ordinal features if not provided
    if ordinal_values is None:
        ordinal_values = {}

    # initiating list of converted features
    converted_features: list[FeatureIn] = []

    # iterating over each feature
    for feature in features:
        # string case, initiating feature
        if isinstance(feature, str):
            converted_features += [target_class(feature, values=ordinal_values.get(feature), **kwargs)]
        # already a BaseFeature
        elif isinstance(feature, BaseFeature):
            converted_features += [feature]

        else:
            raise TypeError(f"[Features] feature {feature} is neither a str, nor a {target_class.__name__}.")

    # deduplicating features by version name
    return [
        feature
        for n, feature in enumerate(converted_features)
        if feature.version not in get_versions(converted_features[n + 1 :])
    ]


def get_names(features: Sequence[BaseFeature]) -> list[str]:
    """Gives names from Features"""
    return [feature.name for feature in features]


def get_versions(features: Sequence[BaseFeature]) -> list[str]:
    """Gives version names from Features"""
    return [feature.version for feature in features]


def check_duplicate_features(
    ordinals: list[OrdinalFeature],
    categoricals: list[CategoricalFeature],
    quantitatives: list[QuantitativeFeature],
) -> None:
    """Checks that features are distinct"""

    # getting feature names
    ordinal_names = get_versions(ordinals)
    categorcial_names = get_versions(categoricals)
    quantitative_names = get_versions(quantitatives)

    # checking for duplicates
    duplicate = [feature in ordinal_names + quantitative_names for feature in categorcial_names]
    if any(duplicate):
        raise ValueError(f"Provided categoricals found in ordinals/quantitatives: {duplicate}. Please, check inputs!")
    duplicate = [feature in ordinal_names + categorcial_names for feature in quantitative_names]
    if any(duplicate):
        raise ValueError(f"Provided quantitatives found in ordinals/categoricals: {duplicate}. Please, check inputs!")
    duplicate = [feature in quantitative_names + categorcial_names for feature in ordinal_names]
    if any(duplicate):
        raise ValueError(f"Provided ordinals found in categoricals/quantitatives: {duplicate}. Please, check inputs!")
