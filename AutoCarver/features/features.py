""" Defines a set of features"""

from .base_feature import BaseFeature
from .categorical_feature import CategoricalFeature
from .continuous_feature import QuantitativeFeature
from .ordinal_feature import OrdinalFeature
from ..discretizers import GroupedList


class Features:

    def __init__(
        self,
        categoricals: list[str] = None,
        quantitatives: list[str] = None,
        ordinals: list[str] = None,
        ordinal_values: dict[str, list[str]] = None,
    ) -> None:

        # ordered values per ordinal features
        self.ordinal_values = ordinal_values

        # casting features accordingly
        self.categoricals = cast_features(categoricals, CategoricalFeature)
        self.quantitatives = cast_features(quantitatives, QuantitativeFeature)
        self.ordinals = cast_features(ordinals, OrdinalFeature, ordinal_values=self.ordinal_values)

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

        self.names = (
            get_names(self.categoricals) + get_names(self.ordinals) + get_names(self.quantitatives)
        )

        self.list = self.categoricals + self.ordinals + self.quantitatives
        self.dict = {feature.name: feature for feature in self.list}

    def __repr__(self):
        return f"Features({str(list(self.names))})"

    def __call__(self, feature_name: str):
        return self.dict.get(feature_name)

    def __iter__(self):
        return iter(self.list)

    def __getitem__(self, index):
        return self.list[index]

    def update(self, feature_values: dict[str, GroupedList]) -> None:
        for feature, values in feature_values.items():
            self(feature).update(values)


def cast_features(
    features: list[str],
    target_class: type = BaseFeature,
    ordinal_values: dict[str, list[str]] = None,
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
            converted_features += [target_class(feature, values=ordinal_values.get(feature))]
        # already a BaseFeature
        elif isinstance(feature, target_class):
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
