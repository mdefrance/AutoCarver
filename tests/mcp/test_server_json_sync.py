"""AutoCarver/mcp/server.json's version must track pyproject.toml's version.

Mirrors the sentinel-import pattern used by ``tests/examples/test_colab_sync.py``
to exercise the root-level sync script without adding it to the package.
"""

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYNC_SCRIPT = PROJECT_ROOT / "AutoCarver" / "mcp" / "sync_server_json.py"


def _load_sync_module():
    spec = importlib.util.spec_from_file_location("autocarver_sync_server_json", SYNC_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_server_json_version_matches_pyproject() -> None:
    sync = _load_sync_module()
    assert sync.sync(check=True) == 0, (
        "AutoCarver/mcp/server.json's version drifted from pyproject.toml; run `python sync_server_json.py` to update"
    )
