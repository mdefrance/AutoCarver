""" TODO: initiate features from dataset
"""

import json
from typing import Any

from pandas import DataFrame, Series

from ...config import DEFAULT, NAN
from .grouped_list import GroupedList
from .serialization import json_deserialize_content, json_serialize_feature

from abc import ABC, abstractmethod


class BaseFeature(ABC):
    __name__ = "Feature"
    is_quantitative = False
    is_qualitative = False
    is_categorical = False
    is_ordinal = False

    def __init__(self, name: str, **kwargs: dict) -> None:
        self.name = name

        # whether or not feature has some NaNs
        self.has_nan = kwargs.get("has_nan", False)
        self.nan = kwargs.get("nan", NAN)

        # whether or not feature has some default values
        self.has_default = kwargs.get("has_default", False)
        self.default = kwargs.get("default", DEFAULT)

        # whether or not nans must be removed
        self.dropna = kwargs.get("dropna", False)

        # whether or not feature has been fitted
        self.is_fitted = kwargs.get("is_fitted", False)

        # feature values, type and labels
        self.values = None  # current values
        self.labels = None  # current labels
        self.label_per_value: dict[str, str] = {}  # current label for all existing values
        self.value_per_label: dict[str, str] = {}  # a value for each current label

        # initating feature dtypes
        self.is_ordinal = kwargs.get("is_ordinal", self.is_ordinal)
        self.is_categorical = kwargs.get("is_categorical", self.is_categorical)
        self.is_qualitative = kwargs.get("is_qualitative", self.is_qualitative)
        self.is_quantitative = kwargs.get("is_quantitative", self.is_quantitative)

        # max number of characters per label
        self.max_n_chars = kwargs.get("max_n_chars", 50)

        # initiating feature's trained statistics
        self.statistics: dict[str:Any] = kwargs.get("statistics", {})

        # initiating feature's combination history
        self.history: list[dict, Any] = kwargs.get("history", [])

        # initiating base ordering (quantitative features)
        self.raw_order: list[str] = kwargs.get("raw_order", [])

        # initiating feature version
        self.version: str = kwargs.get("version", self.name)
        self.version_tag: str = kwargs.get("version_tag", self.name)

    def __repr__(self):
        return f"{self.__name__}('{self.version}')"

    def fit(self, X: DataFrame, y: Series = None) -> None:
        """Fits the feature to a DataFrame"""
        _, _ = X, y  # unused attributes

        # cehcking for previous fit
        if self.is_fitted:
            raise RuntimeError(f"[{self}] Already been fitted!")

        # looking for NANS
        if any(X[self.name].isna()):
            self.has_nan = True

        self.is_fitted = True  # feature is fitted

    def check_values(self, X: DataFrame) -> None:
        """checks for unexpected values from unique values in DataFrame"""

        # checking for nans whereas at training none were witnessed
        if (any(X[self.version].isna()) or any(X[self.version] == self.nan)) and not self.has_nan:
            raise ValueError(f" - [{self}] Unexpected NaNs.")

    def update_labels(self, ordinal_encoding: bool = False) -> None:
        """updates label for each value of the feature"""

        # initiating labels for qualitative features
        labels = self.get_labels()

        # requested float output (AutoCarver) -> converting to integers
        if ordinal_encoding:
            labels = [n for n, _ in enumerate(labels)]

        # saving updated labels
        self.labels = labels[:]

        # updating label_per_value and value_per_label accordingly
        self.value_per_label = {}
        for value, label in zip(self.values, labels):

            # updating label_per_value
            for grouped_value in self.values.get(value):
                self.label_per_value.update({grouped_value: label})

            # udpating value_per_label
            self.value_per_label.update({label: value})

    @abstractmethod
    def _specific_update(self, values: GroupedList, convert_labels: bool = False) -> None:
        """update content of values specifically per feature type"""
        pass

    def update(
        self,
        values: GroupedList,
        convert_labels: bool = False,
        sorted_values: bool = False,
        replace: bool = False,
        ordinal_encoding: bool = False,
    ) -> None:
        """updates content of values of the feature"""

        # values are the same but sorted
        if sorted_values:
            self.values = self.values.sort_by(values)

        # checking for GroupedList
        elif not isinstance(values, GroupedList):
            raise ValueError(f" - [{self.__name__}] Wrong input, expected GroupedList object.")

        # replacing existing values
        elif replace:
            self.values = values

        # using type method specific
        else:
            self._specific_update(values, convert_labels=convert_labels)

        # updating labels accordingly
        self.update_labels(ordinal_encoding=ordinal_encoding)

    def group_list(
        self, to_discard: list[Any], to_keep: Any, ordinal_encoding: bool = False
    ) -> None:
        """wrapper of GroupedList: groups a list of values into a kept value"""

        values = GroupedList(self.values)
        values.group_list(to_discard, to_keep)
        self.update(values, replace=True, ordinal_encoding=ordinal_encoding)

    def get_labels(self) -> GroupedList:
        """gives labels per values"""
        # default labels are values
        return self.values

    def set_has_default(self, has_default: bool = True, ordinal_encoding: bool = False) -> None:
        """adds default to the feature"""
        # copying values
        values = GroupedList(self.values)

        # setting default
        if has_default:
            # adding to the values
            values.append(self.default)
            self.has_default = True

            # updating labels
            self.update(values, replace=True, ordinal_encoding=ordinal_encoding)

    def set_dropna(self, dropna: bool = True, ordinal_encoding: bool = False) -> None:
        """Activates or deactivates feature's dropna mode"""
        # activating dropna mode
        if dropna:
            # setting dropna
            self.dropna = dropna

            # adding nan to the values only if they were found
            if self.has_nan and not self.values.contains(self.nan):
                values = GroupedList(self.values)
                values.append(self.nan)

                # updating values
                self.update(values, replace=True, ordinal_encoding=ordinal_encoding)

        # deactivating dropna mode
        else:
            # setting dropna
            self.dropna = dropna

            # checking for values merged with nans
            if len(self.values.get(self.nan)) > 1:
                raise RuntimeError(
                    "Can not set feature dropna=False has values were grouped with nans."
                )

            # dropping nans from values
            values = GroupedList(self.values)
            if self.nan in self.values:
                values.remove(self.nan)

            # updating values
            self.update(values, replace=True, ordinal_encoding=ordinal_encoding)

    def get_content(self) -> dict:
        """returns feature values' content"""
        if isinstance(self.values, GroupedList):
            return self.values.content
        return self.values

    def to_json(self, light_mode: bool = False) -> dict:
        """Converts to JSON format.

        To be used with ``json.dump``.

        Parameters
        ----------
        light_mode: bool, optional
            Whether or not to save feature's history and statistics, by default False

        Returns
        -------
        str
            JSON serialized object
        """
        # minimal output json
        feature = {
            "name": self.name,
            "version": self.version,
            "version_tag": self.version_tag,
            "has_nan": self.has_nan,
            "nan": self.nan,
            "has_default": self.has_default,
            "default": self.default,
            "dropna": self.dropna,
            "is_fitted": self.is_fitted,
            "values": self.values,
            "content": self.get_content(),
            "is_ordinal": self.is_ordinal,
            "is_categorical": self.is_categorical,
            "is_qualitative": self.is_qualitative,
            "is_quantitative": self.is_quantitative,
            "raw_order": self.raw_order,
        }

        # light output
        if light_mode:
            # serializing feature dict
            return json_serialize_feature(feature)

        # enriched mode (larger json)
        feature.update({"statistics": self.statistics, "history": self.history})
        return json_serialize_feature(feature)

    @classmethod
    def load(cls, feature_json: dict, ordinal_encoding: bool) -> "BaseFeature":
        """Loads a feature"""

        # deserializing content into grouped list of values
        values = json_deserialize_content(feature_json)

        # loading history
        history = []
        if "history" in feature_json:
            history = json.loads(feature_json.pop("history"))

        # initiating feature without content
        feature = cls(**dict(feature_json, history=history, load_mode=True))

        # updating feature with deserialized content
        if values is not None:
            feature.update(values, replace=True, ordinal_encoding=ordinal_encoding)

        return feature
