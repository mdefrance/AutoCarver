"""Runs docs/source/examples/quick_start.py end-to-end.

The example is the source of truth for the README quick-start block (see
``docs/sync_readme.py``); running it here keeps the snippet honest.
"""

import importlib.util
import sys
from pathlib import Path
from urllib.error import URLError

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATH = PROJECT_ROOT / "docs" / "source" / "examples" / "quick_start.py"


@pytest.mark.network
def test_quick_start_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = importlib.util.spec_from_file_location("autocarver_quick_start", EXAMPLE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    monkeypatch.chdir(tmp_path)
    # a failed download (e.g. the host's TLS chain) says nothing about AutoCarver: skip, don't fail
    try:
        module.main()
    except URLError as error:
        pytest.skip(f"Titanic download failed: {error}")

    assert (tmp_path / "titanic_carver.json").exists()
