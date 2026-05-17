"""Base tools to build simple buckets out of Quantitative and Qualitative features
for a binary classification model.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class GroupedList:
    """Ordered groups of values backed by a single dict.

    - Keys of ``content`` are the *group leaders* and define the ordering.
    - ``content[leader]`` is the list of values that belong to that group
      (the leader itself is always included in its values).

    Invariants enforced by :meth:`sanity_check`:
      1. Self-membership: each leader appears in its own values list.
      2. Disjointness: no value appears in more than one group.

    NaN handling: equality between values uses :func:`is_equal` so that
    ``NaN == NaN`` is treated as ``True`` (pandas/numpy NaNs included).
    """

    # Single source of truth: keys are the ordered group leaders,
    # values are the list of raw values currently grouped under that leader.
    content: dict[Any, list[Any]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, iterable: "np.ndarray | dict | list | tuple | GroupedList" = ()) -> None:
        """Builds a :class:`GroupedList` from one of the supported inputs.

        Parameters
        ----------
        iterable : Union[ndarray, dict, list, tuple, GroupedList], optional
            Initializer, by default ``()`` (yields an empty GroupedList).

            * ``dict`` → main constructor (``{leader: [values...]}``)
            * ``GroupedList`` → deep-copy constructor (independent content)
            * ``list`` / ``tuple`` / ``np.ndarray`` → each element becomes its
              own singleton group, in the iteration order of the input
            * ``()`` (default) → empty GroupedList
        """
        # Start with an empty content dict regardless of input.
        # (dataclass would normally do this, but our custom __init__ overrides it)
        self.content = {}

        # case -1: ndarray → list (handled identically to list below)
        if isinstance(iterable, np.ndarray):
            iterable = list(iterable)

        # case 0: dict input — the canonical constructor
        if isinstance(iterable, dict):
            self._init_from_dict(iterable)

        # case 1: copy constructor
        elif isinstance(iterable, GroupedList):
            # Deep-copy each group's values list so the new instance is independent.
            self.content = {k: list(v) for k, v in iterable.content.items()}

        # case 2: list / tuple — each element is a singleton group, in order
        elif isinstance(iterable, (list, tuple)):
            self.content = {v: [v] for v in iterable}

        else:
            raise TypeError(f"Unsupported initializer for GroupedList: {type(iterable)!r}")

        # final invariant check on the constructed object
        self.sanity_check()

    def _init_from_dict(self, iterable: dict[Any, list[Any]]) -> None:
        """Initializes ``content`` from a ``{leader: [values...]}`` dict.

        Handles two edge cases:

        * A leader that is *missing from its own values list* is auto-appended
          (so users can write ``{"a": ["b", "c"]}`` and get ``["b", "c", "a"]``).
        * A leader whose key is already a *grouped value* under another leader
          (e.g. ``{"a": ["b", "c"], "b": []}``) is dropped: it has already been
          subsumed by the other group.
        """
        # Copy each group's values list to avoid mutating the caller's dict.
        content = {k: list(v) for k, v in iterable.items()}

        # Disjointness check: no value may appear in more than one group's values.
        all_values = [value for values in content.values() for value in values]
        if len(set(all_values)) != len(all_values):
            raise ValueError(f"[{self}] A value is present in several keys (groups)")

        # Iterate over a snapshot of keys because we may pop entries below.
        for key in list(content):
            # Values present in groups other than `key` itself.
            other_values = [val for other_key, values in content.items() if other_key != key for val in values]
            if key not in other_values:
                # `key` is a real (un-grouped) leader: ensure self-membership.
                if key not in content[key]:
                    content[key].append(key)
            else:
                # `key` already belongs to another group → drop its (empty) leader.
                content.pop(key)

        self.content = content

    # ------------------------------------------------------------------
    # List-like read behavior
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"GroupedList({list(self)!r})"

    def __iter__(self) -> Iterator[Any]:
        # Iterate over leaders, in insertion order.
        return iter(self.content)

    def __len__(self) -> int:
        return len(self.content)

    def __contains__(self, key: Any) -> bool:
        # Membership refers to group leaders only (not values inside groups).
        # Use :meth:`contains` for "is this value tracked anywhere?".
        return key in self.content

    def __getitem__(self, key):
        """Returns the leader(s) at the given position.

        * ``gl[i]`` → the i-th leader (supports negative indices)
        * ``gl[i:j]`` → ``list`` of leaders in the slice
        * ``gl[:]`` → full ``list`` copy of leaders

        Note: returns plain ``list`` for slices (matches the legacy
        ``list`` subclass behavior, where slicing already returned a list).
        """
        # Materializing once handles ints (including negative), slices, and
        # raises TypeError for unsupported key types — same as ``list.__getitem__``.
        return list(self.content)[key]

    def __eq__(self, other: object) -> bool:
        # Backward-compat with the legacy ``list`` subclass: equality is
        # leader-order based (i.e. ``list(gl) == ...``), NOT content-dict
        # equality. Callers that need a full content match should compare
        # ``gl.content`` directly. This is what downstream tests and several
        # production paths relied on.
        if isinstance(other, GroupedList):
            return list(self.content) == list(other.content)
        if isinstance(other, list):
            return list(self.content) == other
        return NotImplemented

    def __hash__(self) -> int:  # noqa: D401
        # GroupedList is mutable → unhashable, matching ``list`` semantics.
        raise TypeError("unhashable type: 'GroupedList'")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def values(self) -> list[Any]:
        """Flattened list of *all* values across every group.

        Order is: group-by-group, in each group's own value-list order.
        """
        return [value for values in self.content.values() for value in values]

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def get(self, key: Any, default: Any | None = None) -> list[Any]:
        """Returns the values of group ``key``.

        Mirrors ``dict.get`` but with a twist on the ``default`` argument:

        * If ``default`` is ``None`` (the unset case) → returns ``[]``.
        * If ``default`` is a ``list`` → returns it as-is when ``key`` is missing.
        * Otherwise → returns ``[default]`` (the default is wrapped in a list
          so callers always get a list back).

        Parameters
        ----------
        key : Any
            Group leader to look up.
        default : Any, optional
            Fallback to return if ``key`` is not a leader, by default ``None``.

        Returns
        -------
        list[Any]
            Values content in key, or the (possibly wrapped) default.
        """
        # Computing the effective default once keeps the return-type stable (always list).
        default_value: list[Any] = []
        if default is not None:
            if isinstance(default, list):
                default_value = default
            else:
                default_value = [default]
        return self.content.get(key, default_value)

    def contains(self, value: Any) -> bool:
        """Checks whether ``value`` is tracked in any group (NaN-aware).

        Unlike ``value in gl`` (which only checks leaders), this walks every
        group's values and uses NaN-safe equality.
        """
        return any(is_equal(value, known) for known in self.values)

    def get_group(self, value: Any) -> Any:
        """Returns the leader of the group that contains ``value``.

        NaN-aware via :func:`is_equal`. If ``value`` is not found anywhere,
        falls back to returning ``value`` itself (legacy behavior — callers
        rely on this to treat unknown values as their own group).
        """
        for key, values in self.content.items():
            if any(is_equal(value, elt) for elt in values):
                return key
        return value

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def append(self, new_value: Any) -> None:
        """Appends ``new_value`` as a new singleton group at the end.

        Raises ``ValueError`` if ``new_value`` is already tracked anywhere
        (as a leader or as a member of an existing group).
        """
        # Disallow duplicates across the whole structure (not just leaders).
        if new_value in self.values:
            raise ValueError(f"- [{self}] Value {new_value} already in list!")

        # New singleton group at the end (dict insertion order = list order).
        self.content[new_value] = [new_value]

        # sanity check after modification
        self.sanity_check()

    def update(self, new_value: dict[Any, list[Any]]) -> None:
        """Merges/overrides groups from a ``{leader: [values...]}`` dict.

        * Each entry whose leader is missing from its own values list gets
          the leader appended (same convention as :meth:`__init__`).
        * Existing leaders are *replaced* (not merged) — their previous
          values list is overwritten.
        * New leaders are appended at the end, preserving overall order.

        Raises ``ValueError`` if ``new_value`` is not a ``dict[Any, list]`` or
        if the resulting structure would have duplicate values.
        """
        # Input shape validation: must be dict and every value must be a list.
        if not isinstance(new_value, dict) or not all(isinstance(value, list) for value in new_value.values()):
            raise ValueError(f"[{self}] Provide a dictionnary of lists (values)")

        # Ensure self-membership before splicing into self.content.
        for key, value in new_value.items():
            if key not in value:
                new_value[key] = value + [key]

        # Splice in (existing leaders are overwritten; new leaders are appended).
        self.content.update(new_value)

        # sanity check after modification (also catches disjointness violations)
        self.sanity_check()

    def _group_single_value(self, discarded: Any, kept: Any) -> None:
        """Groups the discarded leader into the kept leader.

        ``discarded``'s values are *prepended* to ``kept``'s values, then the
        ``discarded`` leader is removed. No-op when ``discarded == kept``
        (NaN-aware via :func:`is_equal`).
        """
        # No self-merge (NaN-safe).
        if is_equal(discarded, kept):
            return

        # Both must be existing leaders.
        if discarded not in self.content:
            raise ValueError(f"[{self}] {discarded} not in list")
        if kept not in self.content:
            raise ValueError(f"[{self}] {kept} not in list")

        # Merge values (discarded's values first, keeping their original order) and
        # drop the discarded leader from the dict (which also removes its ordering).
        self.content[kept] = self.content[discarded] + self.content[kept]
        self.content.pop(discarded)

    def group(self, to_discard: list[Any] | Any, to_keep: Any) -> None:
        """Groups one or several discarded leaders under ``to_keep``.

        Accepts either a single value or a list of values for ``to_discard``.
        """
        # case of single value to discard
        if not isinstance(to_discard, list):
            self._group_single_value(to_discard, to_keep)

        # list of values to group
        else:
            for discarded in to_discard:
                self._group_single_value(discarded, to_keep)

        # sanity check after modification
        self.sanity_check()

    def remove(self, value: Any) -> None:
        """Removes leader ``value`` (and its entire group) from the structure.

        Raises ``ValueError`` if ``value`` is not a leader. To check whether
        a value is grouped under some leader, use :meth:`contains`.
        """
        if value not in self.content:
            raise ValueError(f"[{self}] {value} not in list")
        self.content.pop(value)

        # sanity check after modification
        self.sanity_check()

    def pop(self, idx: int) -> None:
        """Removes the leader at position ``idx`` (and its group).

        Raises ``IndexError`` if ``idx`` is out of range and ``TypeError``
        if ``idx`` is not an integer — matching ``list.pop`` semantics.
        """
        value = self[idx]
        self.remove(value)

    def replace_group_leader(self, group_leader: Any, group_member: Any) -> None:
        """Promotes a member of ``group_leader``'s group to be the new leader.

        The new leader keeps the same ordinal position and the same group
        contents; only the dict key changes.
        """
        # The promoted member must actually be in the group.
        if group_member not in self.content[group_leader]:
            raise ValueError(f"[{self}] {group_member} is not in {group_leader}")

        # Rebuild the dict to preserve insertion order while renaming the key:
        # dict.update() on an existing key keeps its position, but we're swapping
        # keys, not values — so we re-create the dict with the substitution applied.
        new_content: dict[Any, list[Any]] = {}
        for key, values in self.content.items():
            if key == group_leader:
                new_content[group_member] = list(values)
            else:
                new_content[key] = values
        self.content = new_content

    # ------------------------------------------------------------------
    # Ordering
    # ------------------------------------------------------------------

    def sort(self) -> "GroupedList":
        """Returns a new :class:`GroupedList` with leaders sorted.

        Strings come first (sorted alphabetically), then non-strings sorted
        numerically — NaNs end up last via numpy's sort rules.
        """
        # str values
        keys_str = [key for key in self if isinstance(key, str)]

        # non-str values
        keys_float = [key for key in self if not isinstance(key, str)]

        # sorting and merging keys
        keys = list(np.sort(keys_str)) + list(np.sort(keys_float))

        # recreating an ordered GroupedList
        return GroupedList({k: self.get(k) for k in keys})

    def sort_by(self, ordering: list[Any]) -> "GroupedList":
        """Returns a new :class:`GroupedList` with leaders ordered by ``ordering``.

        Raises ``ValueError`` if ``ordering`` contains unknown leaders or
        if any current leader is missing from ``ordering`` (ordering must
        be a permutation of the current leaders).
        """
        # checking that all values are given an order
        if not all(o in self for o in ordering):
            raise ValueError(f"[{self}] Unknown values in ordering: {str([v for v in ordering if v not in self])}")
        if not all(s in ordering for s in self):
            raise ValueError(f"[{self}] Missing value from ordering: {str([v for v in self if v not in ordering])}")

        # ordering the content
        return GroupedList({k: self.get(k) for k in ordering})

    # ------------------------------------------------------------------
    # Invariants
    # ------------------------------------------------------------------

    def sanity_check(self) -> None:
        """Raises ``ValueError`` if any structural invariant is violated.

        Checks:
        1. Self-membership: every leader is present in its own value list.
        2. Disjointness: no value appears in more than one group.
        """
        # 1. Self-membership.
        for key, values in self.content.items():
            if key not in values:
                raise ValueError(f"[{self}] Missing group leader {key} in its values: {values}")

        # 2. Disjointness of values across groups.
        all_values = self.values
        if len(set(all_values)) != len(all_values):
            raise ValueError(f"[{self}] Some values are in several groups")


def is_equal(a: Any, b: Any) -> bool:
    """Checks whether ``a`` and ``b`` are equal (NaN-insensitive).

    ``pd.isna`` is used to also catch ``np.nan``, ``pd.NA``, ``None`` and
    NaT — anything pandas considers a missing value.
    """
    # default equality
    equal = a == b

    # Case where a and b are NaNs (treat NaN == NaN as True).
    if pd.isna(a) and pd.isna(b):
        equal = True

    return equal
