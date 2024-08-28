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
        self._has_default = kwargs.get("has_default", False)
        self.default = kwargs.get("default", DEFAULT)

        # whether or not nans must be removed
        self._dropna = kwargs.get("dropna", False)

        # whether or not feature has been fitted
        self.is_fitted = kwargs.get("is_fitted", False)

        # whether or not to ordinally encode labels
        self._ordinal_encoding = kwargs.get("ordinal_encoding", False)

        # feature values, type and labels
        self.values = None  # current values
        self._labels = None  # current labels
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

    @property
    def has_default(self) -> bool:
        """whether or not to the feature has default values"""
        return self._has_default

    @has_default.setter
    def has_default(self, value: bool) -> None:
        """sets ordinal_encoding and labels accordingly"""

        # check input value
        if not isinstance(value, bool):
            raise ValueError(f"Trying to set has_default with type {type(value)}")

        # copying values
        values = GroupedList(self.values)

        # setting has_default
        if value and not self.has_default:

            # adding to the values
            values.append(self.default)

            # updating labels
            self.update(values, replace=True)

        # checking that it was not already set to True
        elif not value and self.has_default:
            raise RuntimeError(f"[{self}] has_default has been set to True, can't go back")

        # updating attribute
        self._has_default = value

    @property
    def ordinal_encoding(self) -> bool:
        """whether or not to ordinally encode feature's labels"""
        return self._ordinal_encoding

    @ordinal_encoding.setter
    def ordinal_encoding(self, value: bool) -> None:
        """sets ordinal_encoding and labels accordingly"""

        # check input value
        if not isinstance(value, bool):
            raise ValueError(f"Trying to set ordinal_encoding with type {type(value)}")

        # updating attribute
        self._ordinal_encoding = value

        # updating labels
        if self.values is not None:
            self.update_labels()

    @property
    def dropna(self) -> bool:
        """whether or not to drop missing values (nan)"""
        return self._dropna

    @dropna.setter
    def dropna(self, value: bool) -> None:
        """Activates or deactivates feature's dropna mode"""

        # check input value
        if not isinstance(value, bool):
            raise ValueError(f"Trying to set dropna with type {type(value)}")

        # check input value
        if self.values is None:
            raise ValueError("Trying to set dropna before there where values observed")

        # activating dropna mode
        if value and not self.dropna:

            # adding nan to the values only if they were found
            if self.has_nan and not self.values.contains(self.nan):
                values = GroupedList(self.values)
                values.append(self.nan)

                # updating values
                self.update(values, replace=True)

        # deactivating dropna mode
        elif not value and self.dropna:

            # checking for values merged with nans
            if self.values is not None and len(self.values.get(self.nan)) > 1:
                raise RuntimeError(
                    "Can not set feature dropna=False has values were grouped with nans."
                )

            # dropping nans from values
            values = GroupedList(self.values)
            if self.nan in self.values:
                values.remove(self.nan)

            # updating values
            self.update(values, replace=True)

        # setting dropna
        self._dropna = value

    @property
    def content(self) -> dict:
        """returns feature values' content"""
        if isinstance(self.values, GroupedList):
            return self.values.content
        return self.values

    @property
    def labels(self) -> GroupedList:
        """gives labels associated to feature's values"""
        # default labels are values
        return self._labels

    @labels.setter
    def labels(self, values: GroupedList) -> None:
        """updates labels per values and values per label associated to feature's labels"""

        # updating label_per_value and value_per_label accordingly
        self.value_per_label = {}
        for value, label in zip(self.values, values):

            # updating label_per_value
            for grouped_value in self.values.get(value):
                self.label_per_value.update({grouped_value: label})

            # udpating value_per_label
            self.value_per_label.update({label: value})

        self._labels = values[:]

    @abstractmethod
    def make_labels(self) -> GroupedList:
        """builds labels according to feature's values"""
        # default labels are values
        return self.values

    def update_labels(self) -> None:
        """updates label for each value of the feature"""

        # initiating labels for qualitative features
        labels = self.make_labels()

        # requested float output (AutoCarver) -> converting to integers
        if self.ordinal_encoding:
            labels = [n for n, _ in enumerate(labels)]

        # saving updated labels
        self.labels = labels

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
    ) -> None:
        """updates content of values of the feature"""

        # values are the same but sorted
        if sorted_values:
            self.values = self.values.sort_by(values)

        # checking for GroupedList
        elif not isinstance(values, GroupedList):
            raise ValueError(f"[{self}] Wrong input, expected GroupedList object.")

        # replacing existing values
        elif replace:
            self.values = values

        # using type method specific
        else:
            self._specific_update(values, convert_labels=convert_labels)

        # updating labels accordingly
        self.update_labels()

    def fit(self, X: DataFrame, y: Series = None) -> None:
        """Fits the feature to a DataFrame"""
        _, _ = X, y  # unused attributes

        # checking for previous fit
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
            raise ValueError(f"[{self}] Unexpected NaNs.")

    def group(self, to_discard: list[str], to_keep: str) -> None:
        """wrapper of GroupedList: groups a list of values into a kept value"""

        # using GroupedList's group
        values = GroupedList(self.values)
        values.group(to_discard, to_keep)

        # updating feature's values
        self.update(values, replace=True)

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
            "content": self.content,
            "is_ordinal": self.is_ordinal,
            "is_categorical": self.is_categorical,
            "is_qualitative": self.is_qualitative,
            "is_quantitative": self.is_quantitative,
            "raw_order": self.raw_order,
            "ordinal_encoding": self.ordinal_encoding,
        }

        # enriched mode (larger json)
        if not light_mode:
            feature.update({"statistics": self.statistics, "history": self.history})

        return json_serialize_feature(feature)

    @classmethod
    def load(cls, feature_json: dict) -> "BaseFeature":
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
            feature.update(values, replace=True)

        return feature
