"""Keeps docs/source/examples/quick_start_colab.ipynb in sync with quick_start.py.

The notebook's code cells (minus the leading ``%pip install`` cell and the
trailing ``carver.summary`` display cell) must match the sentinel-extracted
quick-start snippet — the same mechanism that keeps the README honest
(``docs/sync_readme.py``) keeps the Colab notebook honest.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYNC_SCRIPT = PROJECT_ROOT / "docs" / "sync_readme.py"
NOTEBOOK_PATH = PROJECT_ROOT / "docs" / "source" / "examples" / "quick_start_colab.ipynb"


def _extract_snippet() -> str:
    spec = importlib.util.spec_from_file_location("autocarver_sync_readme", SYNC_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.extract_snippet()


def _normalize(source: str) -> list[str]:
    """Non-empty, stripped lines — comparison is modulo whitespace."""
    return [line.strip() for line in source.splitlines() if line.strip()]


def test_colab_notebook_matches_quick_start_snippet() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_cells = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]

    assert code_cells[0].startswith("%pip install autocarver"), "first cell must install the package"
    assert _normalize(code_cells[-1]) == ["carver.summary"], (
        "last cell must display carver.summary as its final expression"
    )

    notebook_body = "\n".join(code_cells[1:-1])
    assert _normalize(notebook_body) == _normalize(_extract_snippet()), (
        "notebook code cells drifted from docs/source/examples/quick_start.py; "
        "update the notebook to match the sentinel-extracted snippet"
    )
