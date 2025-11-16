""" TODO: initiate features from dataset
"""

from abc import ABC, abstractmethod
from typing import Any, Union

from pandas import DataFrame, Series, option_context

from ...config import Constants
from .grouped_list import GroupedList
from .serialization import json_deserialize_content, json_serialize_feature


class BaseFeature(ABC):
    """Base class for all features

    Parameters
    ----------
    name : str

    Attributes
    ----------
    name : str
        Name of the feature.
    has_nan : bool
        Whether or not feature has some NaNs.
    nan : any
        Value used to represent NaNs.
    has_default : bool
        Whether or not feature has some default values.
    default : any
        Default value.
    dropna : bool
        Whether or not nans must be removed.
    is_fitted : bool
        Whether or not feature has been fitted.
    ordinal_encoding : bool
        Whether or not to ordinally encode labels.
    values : GroupedList
        Feature values.
    labels : GroupedList
        Labels associated to feature's values.
    label_per_value : dict[str, str]
        Label for each value.
    value_per_label : dict[str, str]
        Value for each label.
    is_ordinal : bool
        Whether or not feature is ordinal.
    is_categorical : bool
        Whether or not feature is categorical.
    is_qualitative : bool
        Whether or not feature is qualitative.
    is_quantitative : bool
        Whether or not feature is quantitative.
    max_n_chars : int
        Max number of characters per label.
    statistics : dict[str, Any]
        Feature's trained statistics.
    history : list[dict, Any]
        Feature's combination history.
    raw_order : list[str]
        Base ordering for quantitative features.
    version : str
        Feature version.
    version_tag : str
        Feature version tag.
    """

    __name__ = "Feature"

    is_quantitative = False
    """Whether or not feature is quantitative"""

    is_qualitative = False
    """Whether or not feature is qualitative"""

    is_categorical = False
    """Whether or not feature is categorical"""

    is_ordinal = False
    """Whether or not feature is ordinal"""

    def __init__(self, name: str, **kwargs) -> None:
        """
        Parameters
        ----------
        name : str
            Name of the feature

        Keyword Arguments
        -----------------
        ordinal_encoding : bool, optional
            Whether or not to ordinal encode labels, by default ``False``
        nan : str, optional
            Label for missing values, by default ``"__NAN__"``
        default : str, optional
            Label for default values, by default ``"__OTHER__"``
        """
        self.name = name

        # whether or not feature has some NaNs
        self._has_nan = kwargs.get("has_nan", False)
        self.nan = kwargs.get("nan", Constants.NAN)

        # whether or not feature has some default values
        self._has_default = kwargs.get("has_default", False)
        self.default = kwargs.get("default", Constants.DEFAULT)

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
        self.value_per_label: dict[str, str] = {}  # label for all existing values

        # initating feature dtypes
        self.is_ordinal = kwargs.get("is_ordinal", self.is_ordinal)
        self.is_categorical = kwargs.get("is_categorical", self.is_categorical)
        self.is_qualitative = kwargs.get("is_qualitative", self.is_qualitative)
        self.is_quantitative = kwargs.get("is_quantitative", self.is_quantitative)

        # max number of characters per label
        self.max_n_chars = kwargs.get("max_n_chars", 50)

        # initiating feature's trained statistics
        self._statistics: dict[str:Any] = kwargs.get("statistics", None)

        # measures and filters used by selectors
        self.measures = kwargs.get("measures", {})
        self.filters = kwargs.get("filters", {})

        # initiating feature's combination history
        self._history: list[dict] = kwargs.get("history", None)

        # initiating base ordering (quantitative features)
        self.raw_order: list[str] = kwargs.get("raw_order", [])

        # initiating feature version
        self.version: str = kwargs.get("version", self.name)
        self.version_tag: str = kwargs.get("version_tag", self.name)

    def __repr__(self) -> str:
        """Feature representation"""
        return f"{self.__name__}('{self.version}')"

    @property
    def has_nan(self) -> bool:
        """Wether or not feature has nans"""
        return self._has_nan

    @has_nan.setter
    def has_nan(self, value: bool) -> None:
        """sets has_nan attribute"""
        if not isinstance(value, bool):
            raise ValueError(f"Trying to set has_nan with type {type(value)}")
        self._has_nan = value

    @property
    def statistics(self) -> DataFrame:
        """Feature's trained statistics"""
        # conversion to dataframe
        stats = self._statistics
        if stats is not None:
            stats = DataFrame(stats)

        # convertion to ordinal encoding if requested
        if self.ordinal_encoding and stats is not None:
            stats_copy = stats.copy()
            rev_value_per_label = {v: k for k, v in self.value_per_label.items()}

            # adding nan to the dictionary
            rev_value_per_label[self.nan] = self.label_per_value.get(self.nan)
            stats_copy.index = list(map(rev_value_per_label.get, stats_copy.index))
            return stats_copy
        return stats

    @statistics.setter
    def statistics(self, value: DataFrame) -> None:
        """Feature's trained statistics"""

        # case for binary targets
        if isinstance(value, DataFrame):
            self._statistics = value.to_dict()

        # case for continuous targets
        elif isinstance(value, Series):
            self._statistics = value.to_frame().to_dict()

        # case for selectors
        elif isinstance(value, dict):
            if self._statistics is None:
                self._statistics = {}
            self._statistics.update(value)
        else:
            raise ValueError(f"Trying to set statistics with type {type(value)}")

    @property
    def history(self) -> DataFrame:
        """Feature's combination history"""
        if self._history is not None:
            return DataFrame(self._history)
        return []

    @history.setter
    def history(self, value: list[dict]) -> None:
        """Feature's combination history"""
        # initiating history
        if self._history is None:
            self._history: list[dict] = []
        self._history += value

    @property
    def has_default(self) -> bool:
        """Whether or not the feature has default values"""
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
            raise ValueError("Trying to set dropna before there were values observed")

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
    def labels(self, raw_labels: GroupedList) -> None:
        """updates labels per values and values per label associated to feature's labels"""

        # requested float output (AutoCarver) -> converting to integers
        labels = raw_labels[:]
        if self.ordinal_encoding:
            labels = [n for n, _ in enumerate(labels)]

        # updating list of labels
        self._labels = list(labels)

        # updating label_per_value accordingly
        self._update_value_per_label(raw_labels=raw_labels)

    def _update_value_per_label(self, raw_labels: list[str]) -> None:
        """updates value per label and label per value"""

        # iterating over values and labels
        self.value_per_label = {}
        for value, label, raw_label in zip(self.values, self._labels, raw_labels):
            # updating label_per_value
            for grouped_value in self.values.get(value):
                self.label_per_value.update({grouped_value: label})

            # getting value
            value_to_keep = value
            if self.ordinal_encoding:
                value_to_keep = raw_label

            # updating value_per_label
            self.value_per_label.update({label: value_to_keep})

    @abstractmethod
    def make_labels(self) -> GroupedList:
        """builds labels according to feature's values"""
        # default labels are values
        return self.values

    @abstractmethod
    def _make_summary(self) -> dict:
        """returns a summary of the feature"""

    @property
    def summary(self) -> dict:
        """Summary of feature's discretization process"""
        return self._make_summary()

    def _add_statistics_to_summary(self, summary: list[dict]) -> list[dict]:
        """adds statistics to summary"""
        # adding statistics
        if self.statistics is not None:
            # iterating over each modality
            for label_content in summary:
                label = label_content["label"]
                statistics = self.statistics.loc[label].to_dict()
                label_content.update(statistics)

        # adding hisory
        history = self.history
        if len(history) > 0:
            selected = {}

            # checking for viable combination without dropna
            with option_context("future.no_silent_downcasting", True):
                viable = history["viable"].fillna(False).astype(bool)
            if viable.any():
                # checking for requested dropna
                dropna = history["dropna"].fillna(False)
                if dropna.any():
                    # checking for viable combination with dropna
                    if history[dropna].viable.any():
                        selected = history[viable & dropna].iloc[0].to_dict()

                # best_combination for without dropna
                else:
                    selected = history[viable].iloc[0].to_dict()

            # TODO checking for match between selected combination and summary
            # removing unwanted keys
            selected.pop("viable", None)
            selected.pop("dropna", None)
            selected.pop("combination", None)
            selected.pop("info", None)
            selected.pop("train", None)
            selected.pop("dev", None)

            # adding selected combination to summary
            for label_content in summary:
                label_content.update(selected)

        return summary

    def update_labels(self) -> None:
        """updates label for each value of the feature"""

        # initiating labels for qualitative features
        raw_labels = self.make_labels()

        # saving updated labels
        self.labels = raw_labels

    def _update_statistics_value(
        self, kept_label: Union[str, float], kept_value: Union[str, float]
    ) -> None:
        """updates feature's statistics index with values"""

        # updating feature's statistics
        if self.statistics is not None and isinstance(self.statistics, DataFrame):
            self.statistics.rename(index={kept_label: kept_value}, inplace=True)

    @abstractmethod
    def _specific_update(self, values: GroupedList, convert_labels: bool = False) -> None:
        """update content of values specifically per feature type"""

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

    def _historize(self, combination: dict) -> None:
        """historizes a combination"""
        self.history.append(combination)

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
            "statistics": self._statistics,
            "measures": self.measures,
            "filters": self.filters,
        }

        # enriched mode (larger json)
        if not light_mode:
            feature.update({"history": self._history})

        return json_serialize_feature(feature)

    @classmethod
    def load(cls, feature_json: dict) -> "BaseFeature":
        """Loads a feature"""

        # deserializing content into grouped list of values
        values = json_deserialize_content(feature_json)

        # initiating feature without content
        feature = cls(**dict(feature_json, load_mode=True))

        # updating feature with deserialized content
        if values is not None:
            feature.update(values, replace=True)

        return feature
