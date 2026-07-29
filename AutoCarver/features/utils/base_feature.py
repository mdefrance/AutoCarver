"""Base class for all features."""

from abc import ABC, abstractmethod
from typing import Any, Self

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
    is_nested: bool = False
    is_datetime: bool = False

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
        # values starts empty; use ``self.values.is_empty()`` to check the
        # "no values observed yet" state (was previously ``self.values is None``).
        self.values: GroupedList = GroupedList()
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

    def __eq__(self, other: object) -> bool:
        # keyed on version so a bare list of features can subset a DataFrame's
        # columns (``X[[feat_a, feat_b]]`` matches each ``feature.version`` string)
        if isinstance(other, BaseFeature):
            return self.version == other.version
        if isinstance(other, str):
            return self.version == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.version)

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
        if not self.values.is_empty():
            self.update_labels()

    @property
    def dropna(self) -> bool:
        """Whether NaNs should be dropped."""
        return self._dropna

    @dropna.setter
    def dropna(self, value: bool) -> None:
        if self.values.is_empty():
            raise ValueError("Trying to set dropna before there were values observed")

        # activating dropna mode
        if value and not self._dropna:
            if self.has_nan and not self.values.contains(self.nan):
                values = GroupedList(self.values)
                values.append(self.nan)
                self.update(values, replace=True)

        # deactivating dropna mode
        elif not value and self._dropna:
            if len(self.values.get(self.nan)) > 1:
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
        self._update_value_per_label(labels=labels, raw_labels=raw_labels)

    def _update_value_per_label(self, labels: list, raw_labels: GroupedList) -> None:
        """Updates value_per_label and label_per_value dicts.

        - value_per_label maps labels to values to keep in the dataset;
        - label_per_value maps all values to their label for grouping during transform.

        Parameters
        ----------
        labels : list
            Labels to associate to values, in the same order as self.values leaders.
        raw_labels : GroupedList
            Original labels before ordinal encoding, used to keep track of the right value when encoding is active.
        """

        raw_leaders = list(raw_labels)

        self.value_per_label = {}
        self.label_per_value = {}

        # Iterate self.values (authoritative source of leaders) by position so
        # every leader gets a label_per_value entry — even when labels happens
        # to be shorter because raw_labels (a GroupedList) deduplicated
        # collision-prone entries (see format_quantiles + min_decimals_to_differentiate).
        # Without this invariant, transform_quantitative_feature raises
        # ``KeyError: inf`` on the trailing leader.
        for i, value in enumerate(self.values):
            if i < len(labels):
                label = labels[i]
                raw_label = raw_leaders[i] if i < len(raw_leaders) else labels[i]
            else:
                label = labels[-1] if labels else value
                raw_label = raw_leaders[-1] if raw_leaders else label

            # mapping for grouping during transform — all values in the same group get the same label
            for grouped_value in self.values.get(value):
                self.label_per_value[grouped_value] = label

            # mapping for transform output — only the kept value gets mapped to the label, even when multiple values
            # share the same label due to deduplication or ordinal encoding
            value_to_keep = value
            if self.ordinal_encoding:
                value_to_keep = raw_label
            self.value_per_label[label] = value_to_keep

    # ------------------------------------------------------------------
    # statistics
    # ------------------------------------------------------------------

    @property
    def statistics(self) -> pd.DataFrame | None:
        """Trained statistics as a DataFrame (``None`` when not yet computed).

        Stored internally as a dict in ``_statistics`` for JSON serialization; the
        DataFrame is rebuilt on access and reindexed when ``ordinal_encoding`` is active.
        """
        if self._statistics is None:
            return None

        stats = pd.DataFrame(self._statistics)
        if self.ordinal_encoding:
            rev_value_per_label = {v: k for k, v in self.value_per_label.items()}
            rev_value_per_label[self.nan] = self.label_per_value.get(self.nan)
            stats = stats.copy()
            stats.index = list(map(rev_value_per_label.get, stats.index))
        return stats

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

    def _raw_label_snapshot(self) -> dict:
        """Maps every raw member value to its current raw (string) label.

        Raw labels are the pre-ordinal-encoding labels — the space the stored
        ``_statistics`` index lives in (in both encoding modes).
        """
        raw_labels = list(self.make_labels())
        snapshot: dict = {}
        for i, leader in enumerate(self.values):
            if i < len(raw_labels):
                label = raw_labels[i]
            else:
                label = raw_labels[-1] if raw_labels else leader
            for member in self.values.get(leader):
                snapshot[member] = label
        return snapshot

    def _rebuild_statistics(self, old_snapshot: dict, force_nan: list | None = None) -> None:
        """Re-indexes stored statistics onto the current labels after a manual edit.

        - a bin that absorbed exactly one former bin carries its row over;
        - a bin that absorbed several whole former bins gets an exact aggregate
          (counts/frequencies summed, other columns count-weighted);
        - any bin touched by a *partial* former bin (a split) gets NaN — the
          true per-bin statistics are unknowable without a refit.
        """
        if self._statistics is None:
            return

        old_stats = pd.DataFrame(self._statistics)  # index = old raw labels
        new_snapshot = self._raw_label_snapshot()

        old_labels_per_new = _old_labels_per_new_label(new_snapshot, old_snapshot)
        split_old_labels = _find_split_old_labels(old_labels_per_new)
        new_rows = _build_new_rows(old_labels_per_new, old_stats, split_old_labels)

        # every current label gets a row (bins holding only post-fit synthetic values get NaN)
        nan_row = {col: float("nan") for col in old_stats.columns}
        for label in set(new_snapshot.values()):
            if label not in new_rows:
                new_rows[label] = dict(nan_row)

        # bins the caller knows were split: their true statistics are unknowable
        for label in force_nan or []:
            new_rows[label] = dict(nan_row)

        self._statistics = pd.DataFrame.from_dict(new_rows, orient="index").to_dict()

    # ------------------------------------------------------------------
    # history
    # ------------------------------------------------------------------

    @property
    def history(self) -> pd.DataFrame:
        """Combination history as a DataFrame (empty when no history yet).

        Stored internally as a list of dicts in ``_history`` for JSON serialization;
        the DataFrame is rebuilt on access. Append entries with :meth:`historize`.
        """
        return pd.DataFrame(self._history)

    def historize(self, combination: dict[str, Any]) -> None:
        """Appends a single combination to history."""
        self._history.append(combination)

    # ------------------------------------------------------------------
    # abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    def make_labels(self) -> GroupedList:
        """Builds labels according to feature's values."""
        # default: labels are the leader values themselves
        return self.values

    @abstractmethod
    def _make_summary(self) -> list[dict]:
        """Returns a summary of the feature."""

    def _specific_update(self, values: GroupedList, convert_labels: bool = False) -> None:
        """Update content of values, converting labels back to values if needed."""

        # no values have been set
        if not convert_labels and self.values.is_empty():
            self._check_empty_values(values)
            self.values = values
            return

        # values are not labels
        if not convert_labels:
            # updating: iterating over each grouped values
            for kept_value, grouped_values in values.content.items():
                self.values.group(grouped_values, kept_value)
            return

        # values are labels -> converting them back to values
        # Snapshot the encoded-label -> leader-value map once. self.values.group()
        # below mutates leader positions, so rebuilding this inside the loop would
        # desync self.values[k] (IndexError on the 2nd+ grouped label).
        r_value_per_label = self._reverse_value_per_label()

        # iterating over each grouped values
        for kept_label, grouped_labels in values.content.items():
            self._update_grouped_label(kept_label, grouped_labels, r_value_per_label)

    def _reverse_value_per_label(self) -> dict:
        """Maps each encoded label back to its current leader value (ordinal only)."""
        if self.ordinal_encoding:
            leaders = list(self.values)
            return {v: leaders[k] for k, v in self.value_per_label.items()}
        return {}

    def _update_grouped_label(self, kept_label: str | float, grouped_labels: list, r_value_per_label: dict) -> None:
        """Converts one labelled group back to values and groups them."""

        # checking that kept value exists
        if kept_label not in self.value_per_label:
            raise AttributeError(f"{self} no {kept_label}, in value_per_label: {self.value_per_label}")

        # converting labels to values
        grouped_values = [self.value_per_label.get(label) for label in grouped_labels]

        # checking that grouped values exists
        for grouped_value, grouped_label in zip(grouped_values, grouped_labels):
            if grouped_value is None:
                print(f"{self} no {grouped_label}, in value_per_label: {self.value_per_label}")

        # feature-specific: choosing kept value and finalizing grouped values
        grouped_values, kept_value = self._resolve_grouping(kept_label, grouped_values, r_value_per_label)

        # updating values if any to group
        if len(grouped_values) > 0:
            self.values.group(grouped_values, kept_value)

    def _check_empty_values(self, values: GroupedList) -> None:
        """Optional hook: validates values before the first assignment (no-op by default)."""
        return

    @abstractmethod
    def _resolve_grouping(
        self, kept_label: str | float, grouped_values: list, r_value_per_label: dict
    ) -> tuple[list, str | float]:
        """Selects the kept value and finalizes grouped values specifically per feature type."""

    # ------------------------------------------------------------------
    # summary
    # ------------------------------------------------------------------

    @property
    def summary(self) -> list[dict]:
        """Summary of feature's discretization process."""
        return self._make_summary()

    def _add_statistics_to_summary(self, summary: list[dict]) -> list[dict]:
        """Adds statistics and selected history combination to summary entries."""

        stats = self.statistics
        if stats is not None:
            for label_content in summary:
                label = label_content["label"]
                label_content.update(stats.loc[label].to_dict())

        history = self.history
        if len(history) > 0:
            selected: dict = {}

            viable = history["viable"].fillna(False).astype(bool)

            if viable.any():
                dropna = history["dropna"].fillna(False)
                if dropna.any():
                    if history[dropna].viable.any():
                        selected = history[viable & dropna].iloc[0].to_dict()
                else:
                    selected = history[viable].iloc[0].to_dict()

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

    def update(
        self,
        values: "GroupedList | list",
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

        # features that already account for nans can't have "unexpected" ones — short-circuit before
        # touching the column so the full-length scan is skipped entirely for them.
        if self.has_nan:
            return

        # vectorized ``.any()`` (C-level, short-circuits) instead of the builtin ``any()`` which
        # iterates the 300k-element Series in python.
        column = X[self.version]
        if column.isna().any() or (column == self.nan).any():
            raise ValueError(f"[{self}] Unexpected NaNs.")

    def group(self, to_discard: list[str], to_keep: str, convert_labels: bool = True) -> None:
        """Groups a list of labels (or raw values, when ``convert_labels=False``) into a kept one."""

        old_snapshot = self._raw_label_snapshot() if self._statistics is not None else {}
        grouped = GroupedList({to_keep: [*to_discard, to_keep]})
        self.update(grouped, convert_labels=convert_labels)
        self._rebuild_statistics(old_snapshot)

    def _leader_of_label(self, label: str | int) -> Any:
        """Resolves a display label (str, or int code when ordinal_encoding) to its leader value."""
        if self.labels is None or label not in self.labels:
            raise ValueError(f"[{self}] Unknown label {label}. Available labels: {self.labels}")
        if len(self.labels) != len(self.values):
            raise ValueError(
                f"[{self}] Some bins share the same display label (e.g. truncated at max_n_chars);"
                " labels are ambiguous, manual editing is not supported in this state."
            )
        return list(self.values)[self.labels.index(label)]

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

        feature: dict[str, Any] = {
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
            "is_nested": self.is_nested,
            "is_datetime": self.is_datetime,
            "ordinal_encoding": self.ordinal_encoding,
            "statistics": self._statistics,
        }

        if not light_mode:
            feature["history"] = self._history

        return json_serialize_feature(feature)

    @classmethod
    def load(cls, feature_json: dict) -> Self:
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


def _old_labels_per_new_label(new_snapshot: dict, old_snapshot: dict) -> dict:
    """Old labels absorbed by each new label (members unseen at fit have no old label)."""
    old_labels_per_new: dict = {}
    for member, new_label in new_snapshot.items():
        if member in old_snapshot:
            old_labels_per_new.setdefault(new_label, set()).add(old_snapshot[member])
    return old_labels_per_new


def _find_split_old_labels(old_labels_per_new: dict) -> set:
    """An old label spread across several new labels means a split: stats unknowable."""
    owner_per_old: dict = {}
    split_old_labels: set = set()
    for new_label, old_labels in old_labels_per_new.items():
        for old_label in old_labels:
            if old_label in owner_per_old and owner_per_old[old_label] != new_label:
                split_old_labels.add(old_label)
            owner_per_old[old_label] = new_label
    return split_old_labels


def _build_new_rows(old_labels_per_new: dict, old_stats: pd.DataFrame, split_old_labels: set) -> dict:
    """Aggregates old stats rows onto each new label (NaN when unknowable or unseen)."""
    new_rows: dict = {}
    for new_label, old_labels in old_labels_per_new.items():
        known = [label for label in old_labels if label in old_stats.index]
        if any(label in split_old_labels for label in old_labels) or len(known) == 0:
            new_rows[new_label] = {col: float("nan") for col in old_stats.columns}
        elif len(known) == 1:
            new_rows[new_label] = old_stats.loc[known[0]].to_dict()
        else:
            new_rows[new_label] = _aggregate_stats_rows(old_stats.loc[known])
    return new_rows


def _aggregate_stats_rows(rows: pd.DataFrame) -> dict:
    """Aggregates several bins' statistics rows into one merged bin's row.

    ``count`` and ``frequency`` sum exactly; every other column is pooled by a
    count-weighted mean (falling back to frequency weights, then a plain mean),
    which is exact for per-bin means such as target rates.
    """
    if "count" in rows.columns and rows["count"].notna().all():
        weights = rows["count"].astype(float)
    elif "frequency" in rows.columns and rows["frequency"].notna().all():
        weights = rows["frequency"].astype(float)
    else:
        weights = pd.Series(1.0, index=rows.index)
    if weights.sum() == 0:
        weights = pd.Series(1.0, index=rows.index)

    aggregated: dict = {}
    for col in rows.columns:
        values = rows[col]
        if not pd.api.types.is_numeric_dtype(values) or values.isna().any():
            # a NaN input (already-unknowable bin, e.g. from a prior split) must
            # propagate — pandas' default skipna sum/mean would otherwise silently
            # treat it as absent and understate the aggregate.
            aggregated[col] = float("nan")
        elif col in ("count", "frequency"):
            aggregated[col] = values.sum()
        else:
            aggregated[col] = float((values * weights).sum() / weights.sum())
    return aggregated
