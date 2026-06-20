"""Tests for ordinal task resolution in the MCP CarverSession."""

import pandas as pd

from AutoCarver.carvers import OrdinalCarver
from AutoCarver.mcp.session import _CARVERS, CarverSession


def test_ordinal_registered():
    """`ordinal` maps to OrdinalCarver in the carver registry."""
    assert _CARVERS["ordinal"] is OrdinalCarver


def test_explicit_ordinal_task_resolves():
    """An explicit task='ordinal' resolves to ordinal regardless of the values."""
    session = CarverSession()
    y = pd.Series(range(1, 21))  # 20 ordered integer levels
    assert session._resolve_task("ordinal", y) == "ordinal"


def test_auto_does_not_pick_ordinal():
    """`auto` cannot distinguish ordinal from multiclass; many-class ints stay multiclass."""
    session = CarverSession()
    y = pd.Series(list(range(1, 21)) * 3)  # integer-coded, >10 levels
    assert session._resolve_task("auto", y) == "multiclass"
