"""Base class for all features."""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from AutoCarver.config import Constants
from AutoCarver.features.utils.grouped_list import GroupedList
from AutoCarver.features.utils.serialization import json_deserialize_content, json_serialize_feature


class BaseFeature(ABC):
    """Base class for all features.

    Parameters
    ----------
    name : str
        Name of the feature.
    """

    __name__ = "Feature"

    # class-level type traits (set by subclasses, never per-instance)
    is_quantitative: bool = False
    is_qualitative: bool = False
    is_categorical: bool = False
    is_ordinal: bool = False

    def __init__(self, name: str) -> None:
        self.name = name

        # version metadata — set by Features / make_version, not by user input
        self.version: str = name
        self.version_tag: str = name

        # configurable labels — set by Features when nan/default kwargs are passed
        self.nan: str = Constants.NAN
        self.default: str = Constants.DEFAULT

        # state flags — set by fit(), Features, or load()
        self.has_nan: bool = False
        self._has_default: bool = False
        self._dropna: bool = False
        self._ordinal_encoding: bool = False
        self.is_fitted: bool = False

        # values and labels (populated by fit/update)
        self.values: GroupedList | None = None
        # _labels stays None until update_labels() runs; callers like xtab.reindex
        # rely on the None-vs-list distinction ("no labels yet" vs "empty labels").
        self._labels: list | None = None
        self.label_per_value: dict[Any, Any] = {}
        self.value_per_label: dict[Any, Any] = {}

        # statistics and history (populated by carver)
        self._statistics: dict[str, Any] | None = None
        self._history: list[dict[str, Any]] = []

        # selector metrics (populated by selectors at runtime)
        self.measures: dict[str, dict[str, Any]] = {}
        self.filters: dict[str, dict[str, Any]] = {}

    def __repr__(self) -> str:
        return f"{self.__name__}('{self.version}')"

    # ------------------------------------------------------------------
    # state flag properties
    # ------------------------------------------------------------------

    @property
    def has_default(self) -> bool:
        """Whether the feature has default values."""
        return self._has_default

    @has_default.setter
    def has_default(self, value: bool) -> None:
        # adding default to values when toggled on
        if value and not self._has_default:
            values = GroupedList(self.values)
            values.append(self.default)
            self.update(values, replace=True)
        elif not value and self._has_default:
            raise RuntimeError(f"[{self}] has_default has been set to True, can't go back")

        self._has_default = value

    @property
    def ordinal_encoding(self) -> bool:
        """Whether to ordinally encode feature labels."""
        return self._ordinal_encoding

    @ordinal_encoding.setter
    def ordinal_encoding(self, value: bool) -> None:
        self._ordinal_encoding = value
        if self.values is not None:
            self.update_labels()

    @property
    def dropna(self) -> bool:
        """Whether NaNs should be dropped."""
        return self._dropna

    @dropna.setter
    def dropna(self, value: bool) -> None:
        if self.values is None:
            raise ValueError("Trying to set dropna before there were values observed")

        # activating dropna mode
        if value and not self._dropna:
            if self.has_nan and not self.values.contains(self.nan):
                values = GroupedList(self.values)
                values.append(self.nan)
                self.update(values, replace=True)

        # deactivating dropna mode
        elif not value and self._dropna:
            if self.values is not None and len(self.values.get(self.nan)) > 1:
                raise RuntimeError("Can not set feature dropna=False has values were grouped with nans.")

            values = GroupedList(self.values)
            if self.nan in self.values:
                values.remove(self.nan)
            self.update(values, replace=True)

        self._dropna = value

    # ------------------------------------------------------------------
    # values / labels / content
    # ------------------------------------------------------------------

    @property
    def content(self) -> dict:
        """Feature values' content as a leader-keyed dict."""
        if isinstance(self.values, GroupedList):
            return self.values.content
        return {}

    @property
    def labels(self) -> list | None:
        """Labels associated to feature's values, or ``None`` before ``update_labels()`` runs."""
        return self._labels

    @labels.setter
    def labels(self, raw_labels: GroupedList) -> None:
        labels = raw_labels[:]
        if self.ordinal_encoding:
            labels = [n for n, _ in enumerate(labels)]

        self._labels = list(labels)
        self._update_value_per_label(raw_labels=raw_labels)

    def _update_value_per_label(self, raw_labels: list) -> None:
        """Updates value_per_label and label_per_value dicts."""

        self.value_per_label = {}
        self.label_per_value = {}
        for value, label, raw_label in zip(self.values, self._labels, raw_labels):
            for grouped_value in self.values.get(value):
                self.label_per_value[grouped_value] = label

            value_to_keep = value
            if self.ordinal_encoding:
                value_to_keep = raw_label
            self.value_per_label[label] = value_to_keep

    # ------------------------------------------------------------------
    # statistics
    # ------------------------------------------------------------------

    @property
    def statistics(self) -> dict[str, Any] | None:
        """Raw trained statistics as a column-major dict (or ``None``)."""
        return self._statistics

    @statistics.setter
    def statistics(self, value: pd.DataFrame | pd.Series | dict) -> None:
        # binary targets: DataFrame
        if isinstance(value, pd.DataFrame):
            self._statistics = value.to_dict()
        # continuous targets: Series
        elif isinstance(value, pd.Series):
            self._statistics = value.to_frame().to_dict()
        # selectors: dict (merged into existing statistics)
        elif isinstance(value, dict):
            if self._statistics is None:
                self._statistics = {}
            self._statistics.update(value)
        else:
            raise ValueError(f"Trying to set statistics with type {type(value)}")

    @property
    def statistics_dataframe(self) -> pd.DataFrame | None:
        """DataFrame view of statistics. Reindexed when ordinal_encoding is active."""
        if self._statistics is None:
            return None

        stats = pd.DataFrame(self._statistics)
        if self.ordinal_encoding:
            rev_value_per_label = {v: k for k, v in self.value_per_label.items()}
            rev_value_per_label[self.nan] = self.label_per_value.get(self.nan)
            stats = stats.copy()
            stats.index = list(map(rev_value_per_label.get, stats.index))
        return stats

    # ------------------------------------------------------------------
    # history
    # ------------------------------------------------------------------

    @property
    def history(self) -> list[dict[str, Any]]:
        """Raw combination history as a list of dicts."""
        return self._history

    @property
    def history_dataframe(self) -> pd.DataFrame:
        """DataFrame view of history (empty DataFrame when no history)."""
        return pd.DataFrame(self._history)

    def historize(self, combination: dict[str, Any]) -> None:
        """Appends a single combination to history."""
        self._history.append(combination)

    # ------------------------------------------------------------------
    # abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    def make_labels(self) -> GroupedList | None:
        """Builds labels according to feature's values."""
        # default: labels are the leader values themselves
        return self.values

    @abstractmethod
    def _make_summary(self) -> list[dict]:
        """Returns a summary of the feature."""

    @abstractmethod
    def _specific_update(self, values: GroupedList, convert_labels: bool = False) -> None:
        """Update content of values specifically per feature type."""

    # ------------------------------------------------------------------
    # summary
    # ------------------------------------------------------------------

    @property
    def summary(self) -> list[dict]:
        """Summary of feature's discretization process."""
        return self._make_summary()

    def _add_statistics_to_summary(self, summary: list[dict]) -> list[dict]:
        """Adds statistics and selected history combination to summary entries."""

        stats_df = self.statistics_dataframe
        if stats_df is not None:
            for label_content in summary:
                label = label_content["label"]
                label_content.update(stats_df.loc[label].to_dict())

        history_df = self.history_dataframe
        if len(history_df) > 0:
            selected: dict = {}

            with pd.option_context("future.no_silent_downcasting", True):
                viable = history_df["viable"].fillna(False).astype(bool)

            if viable.any():
                dropna = history_df["dropna"].fillna(False)
                if dropna.any():
                    if history_df[dropna].viable.any():
                        selected = history_df[viable & dropna].iloc[0].to_dict()
                else:
                    selected = history_df[viable].iloc[0].to_dict()

            # removing unwanted keys
            for key in ("viable", "dropna", "combination", "info", "train", "dev"):
                selected.pop(key, None)

            for label_content in summary:
                label_content.update(selected)

        return summary

    # ------------------------------------------------------------------
    # update / fit / check
    # ------------------------------------------------------------------

    def update_labels(self) -> None:
        """Updates label for each value of the feature."""
        self.labels = self.make_labels()

    def _update_statistics_value(self, kept_label: str | float, kept_value: str | float) -> None:
        """Renames a statistics index entry to track a value-keep operation."""

        if self._statistics is None:
            return

        # statistics is stored as a dict of {col: {index: value}} — rename keys in each inner dict
        for index_map in self._statistics.values():
            if isinstance(index_map, dict) and kept_label in index_map:
                index_map[kept_value] = index_map.pop(kept_label)

    def update(
        self,
        values: GroupedList,
        convert_labels: bool = False,
        sorted_values: bool = False,
        replace: bool = False,
    ) -> None:
        """Updates content of values of the feature."""

        # values are the same but sorted
        if sorted_values:
            self.values = self.values.sort_by(values)

        elif not isinstance(values, GroupedList):
            raise ValueError(f"[{self}] Wrong input, expected GroupedList object.")

        elif replace:
            self.values = values

        else:
            self._specific_update(values, convert_labels=convert_labels)

        self.update_labels()

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        """Fits the feature to a DataFrame."""
        _ = y

        if self.is_fitted:
            raise RuntimeError(f"[{self}] Already been fitted!")

        if any(X[self.name].isna()):
            self.has_nan = True

        self.is_fitted = True

    def check_values(self, X: pd.DataFrame) -> None:
        """Checks for unexpected values in DataFrame."""

        if (any(X[self.version].isna()) or any(X[self.version] == self.nan)) and not self.has_nan:
            raise ValueError(f"[{self}] Unexpected NaNs.")

    def group(self, to_discard: list[str], to_keep: str) -> None:
        """Groups a list of values into a kept value."""

        values = GroupedList(self.values)
        values.group(to_discard, to_keep)
        self.update(values, replace=True)

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------

    def to_json(self, light_mode: bool = False) -> dict[str, Any]:
        """Converts to a JSON-serializable dict.

        Parameters
        ----------
        light_mode : bool, optional
            Skip ``history`` when True, by default ``False``.
        """

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
            # class-level type traits (discriminator for Features.load)
            "is_qualitative": self.is_qualitative,
            "is_quantitative": self.is_quantitative,
            "is_categorical": self.is_categorical,
            "is_ordinal": self.is_ordinal,
            "ordinal_encoding": self.ordinal_encoding,
            "statistics": self._statistics,
        }

        if not light_mode:
            feature["history"] = self._history

        return json_serialize_feature(feature)

    @classmethod
    def load(cls, feature_json: dict) -> "BaseFeature":
        """Loads a feature from a JSON dict (bypasses subclass init validations)."""

        instance = cls.__new__(cls)
        BaseFeature.__init__(instance, name=feature_json["name"])
        instance._restore_from_json(feature_json)
        return instance

    def _restore_from_json(self, feature_json: dict) -> None:
        """Restores feature state from a JSON dict (post-init)."""

        self.version = feature_json.get("version", self.name)
        self.version_tag = feature_json.get("version_tag", self.name)
        self.nan = feature_json.get("nan", Constants.NAN)
        self.default = feature_json.get("default", Constants.DEFAULT)
        self._ordinal_encoding = feature_json.get("ordinal_encoding", False)
        self.is_fitted = feature_json.get("is_fitted", False)
        self.has_nan = feature_json.get("has_nan", False)
        self._has_default = feature_json.get("has_default", False)
        self._dropna = feature_json.get("dropna", False)
        self._statistics = feature_json.get("statistics")
        self._history = list(feature_json.get("history") or [])

        # restore values / labels
        values = json_deserialize_content(feature_json)
        if values is not None:
            self.update(values, replace=True)
