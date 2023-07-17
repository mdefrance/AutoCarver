"""Base tools to build simple buckets out of Quantitative and Qualitative features
for a binary classification model.
"""

from typing import Any, Union

from numpy import ndarray, sort
from pandas import isna


class GroupedList(list):
    """An ordered list that's extended with a per-value content dict."""

    def __init__(self, iterable: Union[ndarray, dict, list, tuple] = ()) -> None:
        """An ordered list that historizes its elements' content.

        Parameters
        ----------
        iterable : Union[ndarray, dict, list, tuple], optional
            List-like or GroupedList, by default ()
        """
        # TODO: move list to an attribute `order`?
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
            all_values = [val for _, values in iterable.items() for val in values]
            assert len(list(set(all_values))) == len(
                all_values
            ), "A value is present in several keys (groups)"

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
        if default is None:
            default_value = []
        found = self.content.get(key, default_value)

        return found

    def group(self, discarded: Any, kept: Any) -> None:
        """Groups the discarded value with the kept value

        Parameters
        ----------
        discarded : Any
            Value to be grouped into the key `to_keep`.
        kept : Any
            Key value in which to group `discarded`.
        """

        # checking that those values are distinct
        if not is_equal(discarded, kept):
            # checking that those values exist in the list
            assert discarded in self, f"{discarded} not in list"
            assert kept in self, f"{kept} not in list"

            # accessing values content in each value
            content_discarded = self.content.get(discarded)
            content_kept = self.content.get(kept)

            # updating content dict
            self.content.update({kept: content_discarded + content_kept, discarded: []})

            # removing discarded from the list
            self.remove(discarded)

    def group_list(self, to_discard: list[Any], to_keep: Any) -> None:
        """Groups elements to_discard into values to_keep

        Parameters
        ----------
        to_discard : list[Any]
            Values to be grouped into the key `to_keep`.
        to_keep : Any
            Key value in which to group `to_discard` values.
        """

        for discarded, kept in zip(to_discard, [to_keep] * len(to_discard)):
            self.group(discarded, kept)

    def append(self, new_value: Any) -> None:
        """Appends a new_value to the GroupedList

        Parameters
        ----------
        new_value : Any
            New key to be added.
        """

        self += [new_value]
        self.content.update({new_value: [new_value]})

    def update(self, new_value: dict[Any, list[Any]]) -> None:  # TODO: not working as expected
        """Updates the GroupedList via a dict

        Parameters
        ----------
        new_value : dict[Any, list[Any]]
            Dict of key, values to updated `content` dict
        """

        # adding keys to the order if they are new values
        self += [key for key, _ in new_value.items() if key not in self]

        # updating content according to new_value
        self.content.update(new_value)

    def sort(self):
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
        sorted = GroupedList({k: self.get(k) for k in keys})

        return sorted

    def sort_by(self, ordering: list[Any]) -> None:
        """Sorts the values of the list and dict according to `ordering`, if any, NaNs are the last.

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
        assert all(
            o in self for o in ordering
        ), f"Unknown values in ordering: {str([v for v in ordering if v not in self])}"
        assert all(
            s in ordering for s in self
        ), f"Missing value from ordering: {str([v for v in self if v not in ordering])}"

        # ordering the content
        sorted = GroupedList({k: self.get(k) for k in ordering})

        return sorted

    def remove(self, value: Any) -> None:
        """Removes a value from the GroupedList

        Parameters
        ----------
        value : Any
            value to be removed
        """
        super().remove(value)
        self.content.pop(value)

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

        return value

    def values(self) -> list[Any]:
        """All values content in all groups

        Returns
        -------
        list[Any]
            List of all values in the GroupedList
        """

        known = [value for values in self.content.values() for value in values]

        return known

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

        known_values = self.values()

        return any(is_equal(value, known) for known in known_values)

    def get_repr(self, char_limit: int = 6) -> list[str]:
        """Returns a representative list of strings of values of groups.

        Parameters
        ----------
        char_limit : int, optional
            Maximum number of character per string, by default 6

        Returns
        -------
        list[str]
            List of short str representation of the keys' values
        """

        # initiating list of group representation
        repr: list[str] = []

        # iterating over each group
        for group in self:
            # accessing group's values
            values = self.get(group)

            if len(values) == 0:  # case 0: there are no value in this group
                pass

            elif len(values) == 1:  # case 1: there is only one value in this group
                repr += [values[0]]

            elif len(values) == 2:  # case 2: two values in this group
                repr += [f"{values[1]}"[:char_limit] + " and " + f"{values[0]}"[:char_limit]]

            elif len(values) > 2:  # case 3: more than two values in this group
                repr += [f"{values[-1]}"[:char_limit] + " to " + f"{values[0]}"[:char_limit]]

        return repr


def is_equal(a: Any, b: Any) -> bool:
    """Checks if a and b are equal (NaN insensitive)

    Parameters
    ----------
    a : Any
        _description_
    b : Any
        _description_

    Returns
    -------
    bool
        _description_
    """

    # default equality
    equal = a == b

    # Case where a and b are NaNs
    if isna(a) and isna(b):
        equal = True

    return equal
