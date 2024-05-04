""" Defines a set of features"""

from .base_feature import BaseFeature
from .categorical_feature import CategoricalFeature
from .continuous_feature import ContinuousFeature
from .ordinal_feature import OrdinalFeature


def cast_features(
    features: list[str | BaseFeature], target_class: type = BaseFeature
) -> list[BaseFeature]:
    """converts a list of string feature names to"""

    # inititating features if not provided
    if features is None:
        features = []

    converted_features = []  # initiating

    # iterating over each feature
    for feature in features:
        # string case, initiating feature
        if isinstance(feature, str):
            converted_features += [target_class(feature)]
        # already a BaseFeature
        elif isinstance(feature, target_class):
            converted_features += [feature]
        # what?
        else:
            raise TypeError(
                f" - [Features] feature {feature} is neither a str, nor a {target_class.__name__}."
            )

    # deduplicating features by name
    return [
        feature
        for n, feature in enumerate(converted_features)
        if feature.name not in [f.name for f in converted_features[n + 1 :]]
    ]


class Features:

    def __init__(
        self,
        categorical_features: list[str | CategoricalFeature] = None,
        continuous_features: list[str | ContinuousFeature] = None,
        ordinal_features: list[str | OrdinalFeature] = None,
    ) -> None:

        # casting features accordingly
        self.categorical_features = cast_features(categorical_features, CategoricalFeature)
        self.continuous_features = cast_features(continuous_features, ContinuousFeature)
        self.ordinal_features = cast_features(ordinal_features, OrdinalFeature)

        # checking that features were passed as input
        if (
            len(self.categorical_features) == 0
            and len(self.continuous_features) == 0
            and len(self.ordinal_features) == 0
        ):
            raise ValueError(
                " - [Features] No feature passed as input. Pleased provided column names"
                "by setting categorical_features, continuous_features or ordinal_features."
            )
