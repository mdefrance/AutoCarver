""" Constants used throughout the modules. """

from dataclasses import dataclass
from typing import Final


@dataclass
class Constants:
    """Constants used throughout the modules."""

    DEFAULT: Final[str] = "__OTHER__"
    NAN: Final[str] = "__NAN__"
    INF: Final[str] = "__INF__"
