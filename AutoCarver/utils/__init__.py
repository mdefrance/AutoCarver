""" Loads utils."""

from .attributes import get_attribute, get_bool_attribute
from .dependencies import has_idisplay
from .extend_docstring import extend_docstring

__all__ = ["extend_docstring", "get_bool_attribute", "get_attribute", "has_idisplay"]
