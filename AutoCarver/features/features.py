"""Defines a set of features"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypeVar, overload

import numpy as np
import pandas as pd

from AutoCarver.features.qualitatives import (
    CategoricalFeature,
    NestedFeature,
    OrdinalFeature,
    get_categorical_features,
    get_nested_features,
    get_ordinal_features,
)
from AutoCarver.features.quantitatives import (
    DatetimeFeature,
    NumericalFeature,
    QuantitativeFeature,
    get_datetime_features,
    get_quantitative_features,
)
from AutoCarver.features.utils.base_feature import BaseFeature
from AutoCarver.features.utils.grouped_list import GroupedList

# Generic over BaseFeature subclasses — lets helpers below preserve the
# concrete type of feature lists (e.g. list[CategoricalFeature] in → out).
TFeature = TypeVar("TFeature", bound=BaseFeature)


@dataclass
class FeaturesConfig:
    """Collection-level config applied to each feature in a :class:`Features`.

    Internal feature state (``nan``/``default``/``ordinal_encoding``/…) is not part of
    the public ``BaseFeature`` constructor — pass them via this dataclass to ``Features``
    or ``Features.from_list`` and they are propagated to each constituent feature.
    """

    nan: str | None = None
    default: str | None = None
    ordinal_encoding: bool = False
    is_fitted: bool = False
    has_nan: bool = False
    has_default: bool = False
    dropna: bool = False


# class AutoFeatures(Features):
#     """TODO"""

#     __name__ = "AutoFeatures"

#     def __init__(self):
#         raise EnvironmentError(
#             f"[{self.__name__}] Should be instantiated with AutoFeatures.from_dataframe()"
#         )

#     def from_dataframe(self, X: pd.DataFrame) -> None:
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


def _check_names_only(names: list[str] | None, kind: str) -> None:
    """Ensures a list of column names contains only strings (no feature instances)."""
    if names is None:
        return
    for name in names:
        if not isinstance(name, str):
            raise TypeError(
                f"[Features] {kind} must be a list of column names (str); got {type(name).__name__}."
                f" To wrap feature instances, use Features.from_list."
            )


def _check_ordinal_mapping(ordinals: dict[str, list[str]] | None) -> None:
    """Ensures ``ordinals`` is a ``{name: [values...]}`` mapping."""
    if ordinals is None:
        return
    if not isinstance(ordinals, dict):
        raise TypeError(
            f"[Features] ordinals must be a dict of {{name: [ordered values]}};"
            f" got {type(ordinals).__name__}. To wrap OrdinalFeature instances, use Features.from_list."
        )


def _check_nested_mapping(nested: dict[str, list[str]] | None) -> None:
    """Ensures ``nested`` is a ``{output_column: [parent columns coarser-ward]}`` mapping."""
    if nested is None:
        return
    if not isinstance(nested, dict):
        raise TypeError(
            f"[Features] nested must be a dict of {{output column: [parent columns]}};"
            f" got {type(nested).__name__}. To wrap NestedFeature instances, use Features.from_list."
        )
    for name, parents in nested.items():
        if not isinstance(name, str) or not isinstance(parents, list) or not all(isinstance(p, str) for p in parents):
            raise TypeError(
                "[Features] nested must map an output column name (str) to a list of parent"
                f" column names (str); got {{{name!r}: {parents!r}}}."
            )


def _check_datetime_pairs(datetimes: list[tuple[str, str]] | None) -> None:
    """Ensures ``datetimes`` is a list of ``(name, reference_date)`` string pairs."""
    if datetimes is None:
        return
    for pair in datetimes:
        if not isinstance(pair, tuple) or len(pair) != 2 or not all(isinstance(item, str) for item in pair):
            raise TypeError(
                "[Features] datetimes must be a list of (name, reference_date) string tuples;"
                f" got {pair!r}. To wrap DatetimeFeature instances, use Features.from_list."
            )


def _dedupe_by_version(features: list[BaseFeature]) -> list[BaseFeature]:
    """Deduplicates features by version, keeping the last occurrence (matches legacy behavior)."""
    later_versions = [f.version for f in features]
    return [f for i, f in enumerate(features) if f.version not in later_versions[i + 1 :]]


class Features:
    """A set of typed features"""

    __name__ = "Features"

    def __init__(
        self,
        categoricals: list[str] | None = None,
        numericals: list[str] | None = None,
        ordinals: dict[str, list[str]] | None = None,
        datetimes: list[tuple[str, str]] | None = None,
        nested: dict[str, list[str]] | None = None,
        config: FeaturesConfig | None = None,
    ) -> None:
        """Build a :class:`Features` collection from column names.

        Parameters
        ----------

        categoricals : list[str], optional
            Categorical column names, by default ``None``.

        numericals : list[str], optional
            Numerical column names, by default ``None``.

        ordinals : dict[str, list[str]], optional
            Ordinal column names mapped to their ordered value list, by default ``None``.

        datetimes : list[tuple[str, str]], optional
            Datetime features as ``(column name, reference_date)`` pairs, by default ``None``.
            Values are discretized as the number of seconds elapsed since ``reference_date``.

        nested : dict[str, list[str]], optional
            Nested features as ``{output column: [parent columns coarser-ward]}``, by default
            ``None``. The output column is the finest level; parents are listed from nearest
            to farthest. Rare modalities of the output column are rolled up to their
            data-derived parent until frequent enough (see :class:`NestedDiscretizer`).

        config : FeaturesConfig, optional
            Collection-level config propagated to each feature, by default ``None``.


        .. warning::
            At least one of ``categoricals``, ``numericals``, ``ordinals``, ``datetimes``
            or ``nested`` must be provided. To build a :class:`Features` from
            already-instantiated feature objects, use :meth:`Features.from_list` instead.
        """
        # validate input types (strings only — instances belong on from_list)
        _check_names_only(categoricals, "categoricals")
        _check_names_only(numericals, "numericals")
        _check_ordinal_mapping(ordinals)
        _check_datetime_pairs(datetimes)
        _check_nested_mapping(nested)

        # build feature instances from names
        all_features: list[BaseFeature] = []
        all_features += [CategoricalFeature(name) for name in (categoricals or [])]
        all_features += [NumericalFeature(name) for name in (numericals or [])]
        all_features += [OrdinalFeature(name, values=values) for name, values in (ordinals or {}).items()]
        all_features += [DatetimeFeature(name, reference_date=ref) for name, ref in (datetimes or [])]
        all_features += [NestedFeature(name, parents=parents) for name, parents in (nested or {}).items()]

        self._build(all_features, config)

    @classmethod
    def from_list(
        cls,
        features: "Iterable[BaseFeature] | Features",
        config: FeaturesConfig | None = None,
    ) -> "Features":
        """Build a :class:`Features` from already-instantiated feature objects.

        Parameters
        ----------
        features : Iterable[BaseFeature] | Features
            Feature instances to wrap. Iterating an existing :class:`Features` is supported.

        config : FeaturesConfig, optional
            Collection-level config propagated to each feature, by default ``None``.
        """
        feature_list = list(features)
        for feature in feature_list:
            if not isinstance(feature, BaseFeature):
                raise TypeError(f"[Features.from_list] expected BaseFeature instances, got {type(feature).__name__}")

        instance = cls.__new__(cls)
        instance._build(_dedupe_by_version(feature_list), config)
        return instance

    def _build(self, features: list[BaseFeature], config: FeaturesConfig | None) -> None:
        """Shared construction body: apply config and group by type."""

        if config is not None:
            apply_collection_state(features, config)

        self._categoricals = get_categorical_features(features)
        self._ordinals = get_ordinal_features(features)
        # nested features must be extracted before datetimes/quantitatives so they don't
        # leak into other typed lists (NestedFeature is a QualitativeFeature subclass)
        self._nested = get_nested_features(features)
        self._datetimes = get_datetime_features(features)
        # quantitatives stored without datetimes; the ``quantitatives`` view recombines them
        self._quantitatives = [f for f in get_quantitative_features(features) if not f.is_datetime]

        if not (self.categoricals or self.quantitatives or self.ordinals or self._nested):
            raise ValueError(
                f"[{self}] No feature passed as input. Please provide column names"
                " by setting categoricals, quantitatives, ordinals, datetimes or nested."
            )

        check_duplicate_features(self.ordinals, self.categoricals, self.quantitatives, self._nested)

        # Derive collection-level flags from per-feature state (the source of truth) so they stay
        # consistent after Features.load — apply_collection_state restores per-feature flags but a
        # hardcoded collection False would desync, and the ordinal_encoding/dropna setters cascade
        # the collection value back onto every feature, silently dropping a restored encoding.
        self._dropna = bool(features) and all(feature.dropna for feature in features)
        self._ordinal_encoding = bool(features) and all(feature.ordinal_encoding for feature in features)
        self.is_fitted = config.is_fitted if config is not None else False

    def __repr__(self) -> str:
        """Returns names of all features"""
        return f"{self.__name__}({str(self.versions)})"

    def __contains__(self, feature: str | BaseFeature) -> bool:
        """checks if a feature is in the features"""
        if isinstance(feature, BaseFeature):
            return feature.version in self.versions
        return feature in self.versions

    @overload
    def __call__(self, feature_name: str) -> BaseFeature: ...
    @overload
    def __call__(self, feature_name: pd.DataFrame) -> list[str]: ...
    def __call__(self, feature_name: str | pd.DataFrame) -> BaseFeature | list[str]:
        """Returns specified feature by name"""

        # case for dataframes
        if isinstance(feature_name, pd.DataFrame):
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

    def __getitem__(self, index: int | str | list[int] | list[str] | slice) -> BaseFeature | list[BaseFeature] | None:
        """Get item by index in list of features, by feature name or with a list of
        indices/feature names
        """
        # list index/slice request
        if isinstance(index, (int, slice)):
            return self.to_list()[index]

        # feature name request
        if isinstance(index, str):
            return self(index)

        # dataframe request
        if isinstance(index, pd.DataFrame):
            index = list(index.columns)

        # list request and element to search for
        if isinstance(index, list) and len(index) > 0:
            # list index request
            if all(isinstance(idx, int) for idx in index):
                self_list = self.to_list()
                return [self_list[idx] for idx in index if isinstance(idx, int)]

            # feature name request
            elif all(isinstance(idx, str) for idx in index):
                return [self(name) for name in index if isinstance(name, str)]
            else:
                raise TypeError(f"[{self.__name__}] List indices must be all int or all str; got {index}")

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
    def qualitatives(self) -> list[OrdinalFeature | CategoricalFeature | NestedFeature]:
        """Returns all qualitative features (categoricals, ordinals and nested)"""
        return self.categoricals + self.ordinals + self._nested

    @property
    def nested(self) -> list[NestedFeature]:
        """Returns all nested features"""
        return self._nested

    @nested.setter
    def nested(self, values: list[NestedFeature]) -> None:
        """sets nested features"""

        if not all(isinstance(feature, NestedFeature) for feature in values):
            raise AttributeError(f"[{self}] Trying to set nested feature with wrongly typed feature")
        self._nested = values

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
        """Returns all quantitative features (datetimes included)"""
        return self._quantitatives + self._datetimes

    @quantitatives.setter
    def quantitatives(self, values: list[QuantitativeFeature]) -> None:
        """sets quantitative features (datetimes are routed to their own list)"""

        if not all(isinstance(feature, QuantitativeFeature) for feature in values):
            raise AttributeError(f"[{self}] Trying to set quantitative feature with wrongly typed feature")
        self._quantitatives = [feature for feature in values if not feature.is_datetime]
        self._datetimes = get_datetime_features(values)

    @property
    def datetimes(self) -> list[DatetimeFeature]:
        """Returns all datetime features (also part of :attr:`quantitatives`)"""
        return self._datetimes

    @datetimes.setter
    def datetimes(self, values: list[DatetimeFeature]) -> None:
        """sets datetime features"""

        if not all(isinstance(feature, DatetimeFeature) for feature in values):
            raise AttributeError(f"[{self}] Trying to set datetime feature with wrongly typed feature")
        self._datetimes = values

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
        self.nested = remove_version(feature_version, self.nested)
        self.quantitatives = remove_version(feature_version, self.quantitatives)

    def keep(self, kept: list[str]) -> None:
        """list of features' versions to keep (removes the others)"""
        self.categoricals = keep_versions(kept, self.categoricals)
        self.ordinals = keep_versions(kept, self.ordinals)
        self.nested = keep_versions(kept, self.nested)
        self.quantitatives = keep_versions(kept, self.quantitatives)

    def replace_feature(self, updated: BaseFeature) -> None:
        """Replaces the stored feature matching ``updated.version`` with ``updated`` (in place).

        Datetimes are checked before quantitatives since :class:`DatetimeFeature` subclasses
        :class:`QuantitativeFeature`.
        """
        # DatetimeFeature must be tested before QuantitativeFeature (it is a subclass)
        if isinstance(updated, CategoricalFeature) and _replace_version(self._categoricals, updated):
            return
        if isinstance(updated, OrdinalFeature) and _replace_version(self._ordinals, updated):
            return
        if isinstance(updated, NestedFeature) and _replace_version(self._nested, updated):
            return
        if isinstance(updated, DatetimeFeature) and _replace_version(self._datetimes, updated):
            return
        if isinstance(updated, QuantitativeFeature) and _replace_version(self._quantitatives, updated):
            return
        raise KeyError(f"[{self.__name__}] feature {updated.version!r} not in Features")

    def check_values(self, X: pd.DataFrame) -> None:
        """Cheks for unexpected values for each feature in columns of DataFrame X"""
        # iterating over all features
        for feature in self:
            # checking for non-fitted features
            if not feature.is_fitted:
                raise RuntimeError(f"[{self.__name__}] '{feature}' not yet fitted!")

            # checking for unexpected values
            feature.check_values(X)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        """fits all features to there respective column in DataFrame X"""
        # iterating over all features
        for feature in self:
            # checking for fitted features
            if feature.is_fitted:
                feature.check_values(X)

            # fitting feature
            else:
                feature.fit(X, y)

    def fillna(self, X: pd.DataFrame, ignore_dropna: bool = False) -> pd.DataFrame:
        """fills nans of a DataFrame"""

        # fills features with nans when dropna is True
        X.fillna(
            {feature.version: feature.nan for feature in self if feature.has_nan and (feature.dropna or ignore_dropna)},
            inplace=True,
        )

        return X

    def unfillna(self, X: pd.DataFrame) -> pd.DataFrame:
        """unfills nans when not supposed to have filled them"""

        # reinstating nans of features for which nans should not have been dropped
        X.replace(
            {
                feature.version: {feature.label_per_value.get(feature.nan, feature.nan): np.nan}
                for feature in self
                if feature.has_nan and not feature.dropna
            },
            inplace=True,
        )

        return X

    def update(
        self,
        feature_values: "dict[str, GroupedList | list]",
        convert_labels: bool = False,
        sorted_values: bool = False,
        replace: bool = False,
    ) -> None:
        """Updates all features using provided feature_values.

        Values may be a plain ``list`` when ``sorted_values=True`` (the list is
        forwarded to ``GroupedList.sort_by``); otherwise a ``GroupedList`` is
        required by ``BaseFeature.update``.
        """
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
        self.nested = make_versions(self.nested, y_classes)
        self.quantitatives = make_versions(self.quantitatives, y_classes)

    def get_version_group(self, y_class: str) -> list[BaseFeature]:
        """Returns all features with specified version_tag"""

        return [feature for feature in self if feature.version_tag == y_class]

    @property
    def history(self) -> pd.DataFrame:
        """Combined history of all features (concatenated, with a ``feature`` column)."""
        frames = []
        for feature in self:
            df = feature.history
            if len(df) > 0:
                df = df.assign(feature=str(feature))
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    @property
    def summary(self) -> pd.DataFrame:
        """Summary of discretization process for all features"""
        # iterating over each feature
        summaries = []
        for feature in self:
            summaries += feature.summary

        # converting to DataFrame
        summaries = pd.DataFrame(summaries)
        if summaries.empty:
            return summaries

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

    def to_list(self) -> list[BaseFeature]:
        """Returns a list of all features"""
        features: list[BaseFeature] = []
        features.extend(self.categoricals)
        features.extend(self.ordinals)
        features.extend(self._nested)
        features.extend(self.quantitatives)
        return features

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
        is_fitted = features_json.pop("is_fitted", False)

        # casting each feature to there corresponding type
        unpacked_features: list[BaseFeature] = []
        for _, feature in features_json.items():
            if feature.get("is_nested"):
                unpacked_features += [NestedFeature.load(feature)]

            elif feature.get("is_categorical"):
                unpacked_features += [CategoricalFeature.load(feature)]

            elif feature.get("is_ordinal"):
                unpacked_features += [OrdinalFeature.load(feature)]

            elif feature.get("is_datetime"):
                unpacked_features += [DatetimeFeature.load(feature)]

            elif feature.get("is_quantitative"):
                unpacked_features += [NumericalFeature.load(feature)]

        # initiating features
        return cls.from_list(unpacked_features, config=FeaturesConfig(is_fitted=bool(is_fitted)))


def _replace_version(features: list[TFeature], updated: TFeature) -> bool:
    """Replaces a feature matching ``updated.version`` in place; returns whether it was found."""
    for i, existing in enumerate(features):
        if existing.version == updated.version:
            features[i] = updated
            return True
    return False


def remove_version(removed_version: str, features: list[TFeature]) -> list[TFeature]:
    """removes a feature according its version"""
    return [feature for feature in features if feature.version != removed_version]


def keep_versions(kept_versions: list[str], features: list[TFeature]) -> list[TFeature]:
    """keeps requested feature versions according its version"""
    return [feature for feature in features if feature.version in kept_versions]


def make_versions(features: list[TFeature], y_classes: list[str]) -> list[TFeature]:
    """Makes a copy of a list of features with specified version"""
    return [make_version(feature, y_class) for y_class in y_classes for feature in features]


def make_version(feature: TFeature, y_class: str) -> TFeature:
    """Makes a copy of a feature with specified version."""

    # round-trip through JSON to deep-copy; dispatch on the runtime class so the
    # returned feature has the same concrete type as the input.
    new_feature = type(feature).load(feature.to_json(light_mode=False))

    new_feature.version_tag = y_class
    new_feature.version = make_version_name(new_feature.name, y_class)

    return new_feature


def make_version_name(feature_name: str, y_class: str) -> str:
    """Builds a version name for a feature and target class"""

    return f"{feature_name}__y={y_class}"


def apply_collection_state(features: list[BaseFeature], config: FeaturesConfig) -> None:
    """Apply collection-level :class:`FeaturesConfig` to each constituent feature.

    Internal state (nan/default/ordinal_encoding/dropna/has_nan/has_default/is_fitted)
    is not part of the public BaseFeature constructor — Features sets it here.
    """
    # nan/default are string config — propagate when explicitly provided (non-None).
    # The booleans use *truthy-only* propagation: a collection-level False shouldn't
    # override per-feature state (matters on Features.load, where per-feature is_fitted
    # is restored True from JSON while FeaturesConfig.is_fitted is False).
    # The bool sets bypass property setters because features have no values yet at
    # construction time (the dropna/ordinal_encoding setters would otherwise raise).
    for feature in features:
        if config.nan is not None:
            feature.nan = config.nan

        if config.default is not None:
            feature.default = config.default

        if config.ordinal_encoding:
            feature._ordinal_encoding = True

        if config.dropna:
            feature._dropna = True

        if config.has_nan:
            feature.has_nan = True

        if config.has_default:
            feature._has_default = True

        if config.is_fitted:
            feature.is_fitted = True


def get_names(features: list[BaseFeature]) -> list[str]:
    """Gives names from Features"""
    return [feature.name for feature in features]


def get_versions(features: list[TFeature]) -> list[str]:
    """Gives version names from Features"""
    return [feature.version for feature in features]


def check_duplicate_features(
    ordinals: list[OrdinalFeature],
    categoricals: list[CategoricalFeature],
    quantitatives: list[QuantitativeFeature],
    nested: list[NestedFeature] | None = None,
) -> None:
    """Checks that features are distinct"""

    # getting feature names
    ordinal_names = get_versions(ordinals)
    categorcial_names = get_versions(categoricals)
    quantitative_names = get_versions(quantitatives)
    nested_names = get_versions(nested or [])

    # checking for duplicates
    duplicate = [feature in ordinal_names + quantitative_names + nested_names for feature in categorcial_names]
    if any(duplicate):
        raise ValueError(f"Provided categoricals found in ordinals/quantitatives: {duplicate}. Please, check inputs!")
    duplicate = [feature in ordinal_names + categorcial_names + nested_names for feature in quantitative_names]
    if any(duplicate):
        raise ValueError(f"Provided quantitatives found in ordinals/categoricals: {duplicate}. Please, check inputs!")
    duplicate = [feature in quantitative_names + categorcial_names + nested_names for feature in ordinal_names]
    if any(duplicate):
        raise ValueError(f"Provided ordinals found in categoricals/quantitatives: {duplicate}. Please, check inputs!")
    duplicate = [feature in quantitative_names + categorcial_names + ordinal_names for feature in nested_names]
    if any(duplicate):
        raise ValueError(f"Provided nested found in categoricals/ordinals/quantitatives: {duplicate}. Check inputs!")
