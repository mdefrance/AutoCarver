"""Defines a categorical feature"""

from abc import abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from AutoCarver.features.utils.base_feature import BaseFeature, features_of_type
from AutoCarver.features.utils.grouped_list import GroupedList, is_equal


class QualitativeFeature(BaseFeature):
    """Defines a qualitative feature"""

    __name__ = "Qualitative"
    is_qualitative = True

    def __init__(self, name: str) -> None:
        super().__init__(name)
        # base ordering used to format labels
        self.raw_order: list = []

    def to_json(self, light_mode: bool = False) -> dict[str, Any]:
        feature = super().to_json(light_mode=light_mode)
        feature["raw_order"] = self.raw_order
        return feature

    def _restore_from_json(self, feature_json: dict) -> None:
        # raw_order must be restored before super() since super() triggers
        # update_labels(), which reads self.raw_order via _format_modalities
        self.raw_order = list(feature_json.get("raw_order") or [])
        super()._restore_from_json(feature_json)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        # checking for feature's unique non-nan values
        sorted_unique_values = nan_unique(X[self.version], sort=True)

        # checking that feature is not ordinal (already set values)
        if self.values.is_empty():
            # initiating feature with its unique non-nan values
            self.update(GroupedList(sorted_unique_values))

        # checking that raw order has not been set (also useful when loading from json)
        if len(self.raw_order) == 0:
            # saving up number ordering for labeling
            self.raw_order = [self.values.get_group(value) for value in sorted_unique_values]

        # fitting BaseFeature
        super().fit(X, y)

        # class-specific checking for unexpected values
        self.check_values(X)

    def check_values(self, X: pd.DataFrame) -> None:
        """checks for unexpected values from unique values in DataFrame"""

        # computing unique labels in dataframe
        unique_labels = pd.unique(X[self.version])

        # converting to labels
        unique_values = unique_labels[:]
        if not self.ordinal_encoding:
            unique_values = [self.value_per_label.get(label, label) for label in unique_labels]

        # unexpected values for this feature
        unexpected = [
            value
            for value in unique_values
            if not self.values.contains(value) and pd.notna(value) and value != self.nan
        ]
        if len(unexpected) > 0:
            # feature does not have a default value
            if not self.has_default:
                raise ValueError(f"[{self}] Unexpected values: {str(list(unexpected))}")

            # feature has default value:
            # adding unexpected value to list of known values
            for unexpected_value in unexpected:
                self.values.append(unexpected_value)

            # adding unexpected to default
            default_group = self.values.get_group(self.default)
            self.group(unexpected, default_group, convert_labels=False)

        super().check_values(X)

    def move(self, value: str, to_label: str | int) -> None:
        """Moves a single raw modality out of its current bin into the bin labelled ``to_label``.

        ``value`` is a raw (pre-carving) modality; ``to_label`` is a current label
        as shown by ``labels``/``summary`` (an int code when ordinal_encoding is on).
        """
        if not self.values.contains(value):
            raise ValueError(f"[{self}] Unknown value {value}")
        target_leader = self._leader_of_label(to_label)
        source_leader = self.values.get_group(value)
        if is_equal(source_leader, target_leader):
            return

        if self.is_ordinal:
            source_after = [v for v in self.values.get(source_leader) if not is_equal(v, value)]
            target_after = self.values.get(target_leader) + [value]
            self._check_contiguity(source_after, f"bin of {value} after removal")
            self._check_contiguity(target_after, f"bin {to_label} after adding {value}")

        old_snapshot = self._raw_label_snapshot() if self._statistics is not None else {}
        values = GroupedList(self.values)
        values.move_member(value, target_leader)
        self.update(values, replace=True)
        self._rebuild_statistics(old_snapshot)

    def ungroup(self, value: str) -> None:
        """Extracts a single raw modality into its own bin."""
        if not self.values.contains(value):
            raise ValueError(f"[{self}] Unknown value {value}")
        source_leader = self.values.get_group(value)
        if len(self.values.get(source_leader)) == 1:
            return  # already its own bin

        if self.is_ordinal:
            source_after = [v for v in self.values.get(source_leader) if not is_equal(v, value)]
            self._check_contiguity(source_after, f"bin of {value} after removal")

        old_snapshot = self._raw_label_snapshot() if self._statistics is not None else {}
        values = GroupedList(self.values)
        values.extract_member(value)
        self.update(values, replace=True)
        self._rebuild_statistics(old_snapshot)

    def _check_contiguity(self, members: list, context: str) -> None:
        """Ordinal bins must stay contiguous runs of ``raw_order`` (labels render '{first} to {last}')."""
        positions = sorted(self.raw_order.index(v) for v in members if v in self.raw_order)
        if positions and positions != list(range(positions[0], positions[-1] + 1)):
            raise ValueError(
                f"[{self}] {context} would not be contiguous in the declared ordinal order; "
                f"move the in-between modalities too, or use group()."
            )

    def make_labels(self) -> GroupedList:
        """gives labels per values"""

        # iterating over each value and there content
        labels = []
        for group, content in self.content.items():
            # formatting label
            labels += [self._format_modalities(group, content)]

        return GroupedList(labels)

    def _make_summary(self):
        """returns summary of feature's values' content"""
        # iterating over each value
        summary = []
        for group, values in self.content.items():
            # getting group label
            group_label = self.label_per_value.get(group)

            # Qualtiative features: filtering out numbers
            values = [value for value in values if isinstance(value, str)]

            # if there is only one value converting to str
            if len(values) == 1:
                values = values[0]

            # adding group summary
            summary += [{"feature": str(self), "label": group_label, "content": values}]

        # adding statistics and history
        return self._add_statistics_to_summary(summary)

    def _resolve_grouping(
        self, kept_label: str | float, grouped_values: list, r_value_per_label: dict
    ) -> tuple[list, str | float]:
        """selects the kept value and finalizes grouped values"""

        # choosing which value to keep: getting group of kept_value
        kept_value = self.values.get_group(self.value_per_label.get(kept_label))
        # TODO force kept_value to != self.nan like in quantitative feature?

        # keeping only values not already grouped with kept_value
        grouped_values = [
            self.values.get_group(value) for value in grouped_values if self.values.get_group(value) != kept_value
        ]

        # deduplicating
        grouped_values = [value for num, value in enumerate(grouped_values) if value not in grouped_values[num + 1 :]]

        # if ordinal_encoding, converting values to unique values
        if len(grouped_values) > 0 and self.ordinal_encoding:
            grouped_values = [r_value_per_label[value] for value in grouped_values]

        return grouped_values, kept_value

    @abstractmethod
    def _specific_formatting(self, ordered_content: list[str]) -> str:
        """specific label formatting"""

    def _format_modalities(self, group: str, content: list[str]) -> str:
        """Formats a list of float quantiles into a list of boundaries.

        Rounds quantiles to the closest power of 1000.

        Parameters
        ----------
        a_list : list[float]
            Sorted list of quantiles to convert into string

        Returns
        -------
        list[str]
            List of boundaries per quantile
        """

        # ordering content as per original ordering (removes DEFAULT and NAN)
        ordered_content = [
            value
            for value in self.raw_order
            if value in content
            # removing nan
            and value != self.nan
            # removing floats
            and not isinstance(value, np.floating)
            and not isinstance(value, float)
            # removing ints
            and not isinstance(value, np.integer)
            and not isinstance(value, int)
        ]
        # removing duplicates
        ordered_content = list(dict.fromkeys(ordered_content))

        # building label from ordered content
        if len(ordered_content) == 0:
            label = group
        elif len(ordered_content) == 1:
            label = ordered_content[0]
        else:
            label = self._specific_formatting(ordered_content)

        # adding nans
        if self.nan in content and label != self.nan:  # and self.nan not in label:
            label += f", {self.nan}"

        return label


def nan_unique(x: pd.Series, sort: bool = False) -> list[str]:
    """Unique non-NaN values.

    Parameters
    ----------
    x : pd.Series
        Values to be deduplicated.
    sorted : boolean, optionnal
        Whether or not to sort unique by appearance.

    Returns
    -------
    list[str]
        List of unique non-nan values
    """

    # unique values not sorted
    if sort:
        uniques = pd.unique(x)

    # sorting unique values
    else:
        uniques = list(x.value_counts(sort=True, ascending=False).index)

    # filtering out nans
    uniques = [value for value in uniques if pd.notna(value)]

    return uniques


def get_qualitative_features(features: list[BaseFeature]) -> list[QualitativeFeature]:
    """returns qualitative features amongst provided features"""
    return features_of_type(features, QualitativeFeature)
