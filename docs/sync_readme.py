"""Sync the README quick-start block from docs/source/examples/quick_start.py.

The script extracts the body between ``# --8<-- [start:quick_start]`` and
``# --8<-- [end:quick_start]`` in the example file (dedented), and rewrites the
content between ``<!-- quick-start:start -->`` and ``<!-- quick-start:end -->``
in ``README.md`` to a fenced ``python`` code block of that body.

Usage::

    python docs/sync_readme.py            # rewrite README in place
    python docs/sync_readme.py --check    # exit non-zero if README is out of sync
"""

from __future__ import annotations

import argparse
import re
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "docs" / "source" / "examples" / "quick_start.py"
README = ROOT / "README.md"

START_MARK = "# --8<-- [start:quick_start]"
END_MARK = "# --8<-- [end:quick_start]"
README_START = "<!-- quick-start:start -->"
README_END = "<!-- quick-start:end -->"


def extract_snippet() -> str:
    """Returns the dedented body between the source sentinels."""
    text = SOURCE.read_text(encoding="utf-8")
    lines = text.splitlines()
    start = end = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == START_MARK and start is None:
            start = i
        elif stripped == END_MARK and start is not None and end is None:
            end = i
            break
    if start is None or end is None:
        raise SystemExit(f"sentinel markers not found in {SOURCE}")
    body = "\n".join(lines[start + 1 : end])
    return textwrap.dedent(body).strip("\n")


def render_block(snippet: str) -> str:
    return f"{README_START}\n```python\n{snippet}\n```\n{README_END}"


def sync(*, check: bool) -> int:
    snippet = extract_snippet()
    new_block = render_block(snippet)
    pattern = re.compile(
        rf"{re.escape(README_START)}.*?{re.escape(README_END)}",
        flags=re.DOTALL,
    )
    current = README.read_text(encoding="utf-8")
    if not pattern.search(current):
        raise SystemExit(f"sentinels {README_START!r}/{README_END!r} not found in {README}")
    updated = pattern.sub(new_block, current)
    if updated == current:
        return 0
    if check:
        print(f"{README} is out of sync with {SOURCE}; run `python docs/sync_readme.py` to update.")
        return 1
    README.write_text(updated, encoding="utf-8")
    print(f"updated {README}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit non-zero if README is out of sync")
    args = parser.parse_args()
    sys.exit(sync(check=args.check))


if __name__ == "__main__":
    main()
