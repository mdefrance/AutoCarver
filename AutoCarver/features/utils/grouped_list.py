"""Base tools to build simple buckets out of Quantitative and Qualitative features
for a binary classification model.
"""

from typing import Any, Union

from numpy import ndarray, sort
from pandas import isna


class GroupedList(list):
    """An ordered list that's extended with a per-value content dict."""

    def __repr__(self) -> str:
        return f"GroupedList({super().__repr__()})"

    def __init__(self, iterable: Union[ndarray, dict, list, tuple] = ()) -> None:
        """
        Parameters
        ----------
        iterable : Union[ndarray, dict, list, tuple], optional
            List-like or :class:`GroupedList`, by default ``()``
        """
        # case -1: iterable is an array
        if isinstance(iterable, ndarray):
            iterable = list(iterable)

        # case 0: iterable is the content dict
        if isinstance(iterable, dict):
            # storing ordered keys of the dict
            keys = list(iterable)

            # storing the values content per key
            self.content = dict(iterable.items())

            # checking that all values are only present once
            all_values = [value for values in iterable.values() for value in values]
            if not len(list(set(all_values))) == len(all_values):
                raise ValueError(f"[{self}] A value is present in several keys (groups)")

            # adding key to itself if it's not present in an other key
            keys_copy = keys[:]  # copying initial keys
            for key in keys_copy:
                # checking that the value is not comprised in an other key
                all_values = [
                    val
                    for iter_key, values in iterable.items()
                    for val in values
                    if key != iter_key
                ]
                if key not in all_values:
                    # checking that key is missing from its values
                    if key not in iterable[key]:
                        self.content.update({key: self.content[key] + [key]})
                # the key already is in another key (and its values are empty)
                # the key as already been grouped
                else:
                    self.content.pop(key)
                    keys.remove(key)

            # initiating the list with those keys
            super().__init__(keys)

        # case 1: copying a GroupedList
        elif hasattr(iterable, "content"):
            # initiating the list with the provided list of keys
            super().__init__(iterable)

            # copying values associated to keys
            self.content = dict(iterable.content.items())

        # case 2: initiating GroupedList from a list
        elif isinstance(iterable, list):
            # initiating the list with the provided list of keys
            super().__init__(iterable)

            # initiating the values with the provided list of keys
            self.content = {v: [v] for v in iterable}

    # def __getitem__(self, key):
    #     """Returns the value content in the key

    #     Parameters
    #     ----------
    #     key : Any
    #         Key to search for

    #     Returns
    #     -------
    #     list[Any]
    #         Values content in key

    #     Raises
    #     ------
    #     KeyError
    #         If key is not found in Group
    #     """

    #     if key in self.content:
    #         return self.content[key]

    #     raise KeyError(f"Key {key} not found in GroupedList")

    @property
    def values(self) -> list[str]:
        """All values content in all groups

        Returns
        -------
        list[str]
            List of all values in the GroupedList
        """
        return [value for values in self.content.values() for value in values]

    def get(self, key: Any, default: Any = None) -> list[Any]:
        """List of values content in key

        Parameters
        ----------
        key : Any
            Group.
        default : Any, optional
            Value to return if key was not found, by default None

        Returns
        -------
        list[Any]
            Values content in key
        """
        # default to fing an element
        default_value = []
        if default is not None:
            if isinstance(default, list):
                default_value = default
            else:
                default_value = [default]
        found = self.content.get(key, default_value)

        return found

    def sanity_check(self) -> None:
        """raises ValueError if there is an issue with the any element of the GroupedList"""
        # each element of the list should be in its elements
        for key, values in self.content.items():
            if key not in values:
                raise ValueError(f"[{self}] Missing group leader {key} in its values: {values}")

        # an element can not be in several groups
        all_values = self.values
        if len(set(all_values)) != len(all_values):
            raise ValueError(f"[{self}] Some values are in several groups")

        # checking that element of the list are keys of the content dict
        if not len(self) == len(self.content):
            raise ValueError(
                f"[{self}] Keys missing from content dict or element missing from list"
            )
        if any(
            list_element != dict_key
            for list_element, dict_key in zip(self, list(self.content.keys()))
        ):
            raise ValueError(f"[{self}] Not the same ordering between list and content dict")

    def _group_single_value(self, discarded: Any, kept: Any) -> None:
        """Groups the discarded value with the kept value

        Parameters
        ----------
        discarded : Any
            Value to be grouped into the key ``kept``.
        kept : Any
            Key value in which to group ``discarded``.
        """

        # checking that those values are distinct
        if not is_equal(discarded, kept):
            # checking that those values exist in the list
            if discarded not in self:
                raise ValueError(f"[{self}] {discarded} not in list")
            if kept not in self:
                raise ValueError(f"[{self}] {kept} not in list")

            # accessing values content in each value
            content_discarded = self.content.get(discarded)
            content_kept = self.content.get(kept)

            # updating content dict
            self.content.update({kept: content_discarded + content_kept, discarded: []})

            # removing discarded from the list
            self.remove(discarded)

    def group(self, to_discard: Union[list[str], str], to_keep: str) -> None:
        """Groups the discarded value with the kept value

        Parameters
        ----------
        to_discard : list[Any] | Any
            Values to be grouped into the key ``to_keep``.
        to_keep : Any
            Key value in which to group ``to_discard`` values.
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

    def append(self, new_value: Any) -> None:
        """Appends a new_value to the GroupedList

        Parameters
        ----------
        new_value : Any
            New key to be added.
        """

        # checking for already existing values
        if new_value in self.values:
            raise ValueError(f"- [{self}] Value {new_value} already in list!")

        # adding value to list and dict
        self += [new_value]
        self.content.update({new_value: [new_value]})

        # sanity check after modification
        self.sanity_check()

    def update(self, new_value: dict[Any, list[Any]]) -> None:
        """Updates the GroupedList via a dict

        Parameters
        ----------
        new_value : dict[Any, list[Any]]
            Dict of key, values to updated ``content`` dict
        """

        # should provide a dict of lists
        if not isinstance(new_value, dict) or not all(
            isinstance(value, list) for value in new_value.values()
        ):
            raise ValueError(f"[{self}] Provide a dictionnary of lists (values)")

        # adding missing keys to there list of values
        for key, value in new_value.items():
            if key not in value:
                new_value.update({key: value + [key]})

        # adding keys to the order if they are new values
        self += [key for key, _ in new_value.items() if key not in self]

        # updating content according to new_value
        self.content.update(new_value)

        # sanity check after modification
        self.sanity_check()

    def sort(self) -> "GroupedList":
        """Sorts the values of the list and dict (if any, NaNs are last).

        Returns
        -------
        GroupedList
            Sorted GroupedList
        """
        # str values
        keys_str = [key for key in self if isinstance(key, str)]

        # non-str values
        keys_float = [key for key in self if not isinstance(key, str)]

        # sorting and merging keys
        keys = list(sort(keys_str)) + list(sort(keys_float))

        # recreating an ordered GroupedList
        sorted_list = GroupedList({k: self.get(k) for k in keys})

        return sorted_list

    def sort_by(self, ordering: list[Any]) -> "GroupedList":
        """Sorts the values of the list and dict according to ``ordering``, if any,
        NaNs are the last.

        Parameters
        ----------
        ordering : list[Any]
            Order used for ordering of the list of keys.

        Returns
        -------
        GroupedList
            Sorted GroupedList
        """

        # checking that all values are given an order
        if not all(o in self for o in ordering):
            raise ValueError(
                f"[{self}] Unknown values in ordering: "
                f"{str([v for v in ordering if v not in self])}"
            )
        if not all(s in ordering for s in self):
            raise ValueError(
                f"[{self}] Missing value from ordering:"
                f" {str([v for v in self if v not in ordering])}"
            )

        # ordering the content
        sorted_list = GroupedList({k: self.get(k) for k in ordering})

        return sorted_list

    def remove(self, value: Any) -> None:
        """Removes a value from the GroupedList

        Parameters
        ----------
        value : Any
            value to be removed
        """
        super().remove(value)
        self.content.pop(value)

        # sanity check after modification
        self.sanity_check()

    def pop(self, idx: int) -> None:
        """Pop a value from the GroupedList by index

        Parameters
        ----------
        idx : int
            Index of the value to be popped out
        """
        value = self[idx]
        self.remove(value)

    def get_group(self, value: Any) -> Any:
        """Returns the key (group) containing the specified value

        Parameters
        ----------
        value : Any
            Value for which to find the group.

        Returns
        -------
        Any
            Corresponding key (group)
        """

        found = [
            key
            for key, values in self.content.items()
            if any(is_equal(value, elt) for elt in values)
        ]

        if any(found):
            return found[0]

        # TODO expected behavior?
        # if not found, should return itself by default
        return value

    def contains(self, value: Any) -> bool:
        """Checks if a value is content in any group, also matches NaNs.

        Parameters
        ----------
        value : Any
            Value to search for

        Returns
        -------
        bool
            Whether the value is in the GroupedList
        """
        return any(is_equal(value, known) for known in self.values)

    def replace_group_leader(self, group_leader: Any, group_member: Any) -> None:
        """Replaces a group_leader by one of its group_members

        Parameters
        ----------
        group_leader : Any
            One of the list's values (``GroupedList.content.keys()``)
        group_member : Any
            One of the dict's values for specified group_leader
            (``GroupedList.content[group_leader]``)
        """
        # checking that group_member is in group_leader
        if group_member not in self.content[group_leader]:
            raise ValueError(f"[{self}] {group_member} is not in {group_leader}")

        # replacing in the list
        group_idx = self.index(group_leader)
        self[group_idx] = group_member

        # replacing in the dict
        self.content.update({group_member: self.content[group_leader][:]})
        self.content.pop(group_leader)

        # TODO check if this has a purpose
        # sorting things up
        # self.sort_by(self)


def is_equal(a: Any, b: Any) -> bool:
    """Checks if a and b are equal (NaN insensitive)"""

    # default equality
    equal = a == b

    # Case where a and b are NaNs
    if isna(a) and isna(b):
        equal = True

    return equal
