"""Smoke test for docs/source/examples/Comparison/comparison_notebook.ipynb.

Executes the notebook end-to-end with nbclient and fails on any cell error.
Marked ``network`` because the notebook downloads two sklearn-bundled
datasets (German Credit via ``fetch_openml`` and California Housing via
``fetch_california_housing``) the first time it runs on a given machine.
"""

from __future__ import annotations

import os
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = PROJECT_ROOT / "docs" / "source" / "examples" / "Comparison" / "comparison_notebook.ipynb"


@pytest.mark.network
@pytest.mark.filterwarnings("ignore:.*Proactor event loop.*:RuntimeWarning")  # Windows zmq/jupyter
@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="OpenML API is unreliable; fetch_openml('credit-g') intermittently fails in CI",
)
def test_comparison_notebook_executes() -> None:
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}},
    )
    client.execute()
