"""Sync AutoCarver/mcp/server.json's version fields from pyproject.toml.

Keeps the MCP registry manifest's top-level ``version`` and every entry in
``packages[].version`` equal to ``[project].version`` in pyproject.toml, so a
version bump can't silently leave server.json pointing at an unpublished
(or stale) release.

Usage::

    python AutoCarver/mcp/sync_server_json.py            # rewrite server.json in place
    python AutoCarver/mcp/sync_server_json.py --check    # exit non-zero if out of sync
"""

import argparse
import json
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"
SERVER_JSON = ROOT / "AutoCarver" / "mcp" / "server.json"

# MCP registry rejects a longer top-level description at publish time (422).
MAX_DESCRIPTION_LEN = 100


def read_version() -> str:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return data["project"]["version"]


def synced_server_json(version: str) -> dict:
    data = json.loads(SERVER_JSON.read_text(encoding="utf-8"))
    data["version"] = version
    for package in data["packages"]:
        package["version"] = version
    return data


def sync(*, check: bool) -> int:
    version = read_version()
    current = json.loads(SERVER_JSON.read_text(encoding="utf-8"))
    updated = synced_server_json(version)

    description = updated["description"]
    if len(description) > MAX_DESCRIPTION_LEN:
        raise SystemExit(
            f"{SERVER_JSON}'s description is {len(description)} chars, "
            f"exceeds the registry's {MAX_DESCRIPTION_LEN}-char limit: {description!r}"
        )

    if updated == current:
        return 0
    if check:
        print(
            f"{SERVER_JSON} is out of sync with {PYPROJECT} (version {version}); "
            "run `python AutoCarver/mcp/sync_server_json.py` to update."
        )
        return 1
    SERVER_JSON.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
    print(f"updated {SERVER_JSON}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit non-zero if server.json is out of sync")
    args = parser.parse_args()
    sys.exit(sync(check=args.check))


if __name__ == "__main__":
    main()
