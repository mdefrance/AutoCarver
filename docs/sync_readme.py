"""Sync the README quick-start block and quick-start notebook from
docs/source/examples/quick_start.py.

The script extracts the body between ``# --8<-- [start:quick_start]`` and
``# --8<-- [end:quick_start]`` in the example file (dedented). It rewrites the
content between ``<!-- quick-start:start -->`` and ``<!-- quick-start:end -->``
in ``README.md`` to a fenced ``python`` code block of that body, and it
rewrites the code cells (after the leading markdown/pip-install cells) of
``quick_start_colab.ipynb`` by splitting the snippet on its ``# N. ...`` step
comments, one cell per step.

Usage::

    python docs/sync_readme.py            # rewrite README/notebook in place
    python docs/sync_readme.py --check    # exit non-zero if either is out of sync
"""

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "docs" / "source" / "examples" / "quick_start.py"
README = ROOT / "README.md"
NOTEBOOK = ROOT / "docs" / "source" / "examples" / "quick_start_colab.ipynb"

START_MARK = "# --8<-- [start:quick_start]"
END_MARK = "# --8<-- [end:quick_start]"
README_START = "<!-- quick-start:start -->"
README_END = "<!-- quick-start:end -->"
STEP_MARK = re.compile(r"^# \d+\.")
LEADING_CODE_CELLS = 2  # markdown intro + `%pip install` cell


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


def split_into_steps(snippet: str) -> list[str]:
    """Splits the snippet into one block per ``# N. ...`` step comment.

    Any lines before the first step comment (e.g. imports) are prepended to
    the first step's block.
    """
    lines = snippet.splitlines()
    starts = [i for i, line in enumerate(lines) if STEP_MARK.match(line)]
    if not starts:
        raise SystemExit(f"no '# N. ...' step comments found in {SOURCE}")
    bounds = [0] + starts[1:] + [len(lines)]
    return ["\n".join(lines[bounds[i] : bounds[i + 1]]).strip("\n") for i in range(len(bounds) - 1)]


def render_notebook_cells(steps: list[str]) -> list[dict]:
    cells = []
    for step in steps:
        # nbformat convention: every line but the last keeps its trailing newline.
        source = [line + "\n" for line in step.splitlines()[:-1]] + [step.splitlines()[-1]]
        cells.append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source,
            }
        )
    return cells


def sync_readme(*, check: bool) -> int:
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


def sync_notebook(*, check: bool) -> int:
    snippet = extract_snippet()
    steps = split_into_steps(snippet)
    new_cells = render_notebook_cells(steps)
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    current_cells = nb["cells"][LEADING_CODE_CELLS:]
    if current_cells == new_cells:
        return 0
    if check:
        print(f"{NOTEBOOK} is out of sync with {SOURCE}; run `python docs/sync_readme.py` to update.")
        return 1
    nb["cells"] = nb["cells"][:LEADING_CODE_CELLS] + new_cells
    NOTEBOOK.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"updated {NOTEBOOK}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit non-zero if README/notebook are out of sync")
    args = parser.parse_args()
    readme_status = sync_readme(check=args.check)
    notebook_status = sync_notebook(check=args.check)
    sys.exit(readme_status or notebook_status)


if __name__ == "__main__":
    main()
