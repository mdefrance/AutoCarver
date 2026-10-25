"""Runtime key check for the TypedDict configurations (a TypedDict does none itself)."""

from collections.abc import Mapping


def check_config_keys(config: Mapping, typed_dict: type, name: str) -> None:
    """Raises ``TypeError`` when ``config`` holds a key that ``typed_dict`` does not define."""
    known = set(typed_dict.__annotations__)
    unknown = set(config) - known
    if unknown:
        raise TypeError(f"Unknown {name} key(s) {sorted(unknown)}; expected some of {sorted(known)}")
