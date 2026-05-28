"""Generate animated SVGs for the docs from animation example specs.

Walks `docs/source/examples/animations/{discretizers,carvers}/*.py`, reads each
example's `(NAME, FEATURE, TARGET, STOP_AFTER_STAGE)` config, builds the frame
sequence, and writes `<name>.svg` to `docs/source/_static/animations/`.

Usage::

    python docs/build_animations.py            # rebuild all SVGs
    python docs/build_animations.py --check    # exit non-zero if any drifts
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent
ANIM_SRC = ROOT / "docs" / "source" / "examples" / "animations"
OUT_DIR = ROOT / "docs" / "source" / "_static" / "animations"

# Make the animations package importable without adding __init__.py up the chain
sys.path.insert(0, str(ANIM_SRC.parent))  # docs/source/examples


def _load_engine_and_renderer() -> tuple[ModuleType, ModuleType]:
    """Import _engine and _render as a real package so relative imports work."""
    pkg_name = "_anim_pkg"
    pkg_spec = importlib.util.spec_from_file_location(
        pkg_name,
        ANIM_SRC / "__init__.py",
        submodule_search_locations=[str(ANIM_SRC)],
    )
    pkg = importlib.util.module_from_spec(pkg_spec)
    sys.modules[pkg_name] = pkg
    pkg_spec.loader.exec_module(pkg)

    engine_spec = importlib.util.spec_from_file_location(f"{pkg_name}._engine", ANIM_SRC / "_engine.py")
    engine = importlib.util.module_from_spec(engine_spec)
    sys.modules[f"{pkg_name}._engine"] = engine
    engine_spec.loader.exec_module(engine)

    data_spec = importlib.util.spec_from_file_location(f"{pkg_name}._data", ANIM_SRC / "_data.py")
    data = importlib.util.module_from_spec(data_spec)
    sys.modules[f"{pkg_name}._data"] = data
    data_spec.loader.exec_module(data)

    render_spec = importlib.util.spec_from_file_location(f"{pkg_name}._render", ANIM_SRC / "_render.py")
    render = importlib.util.module_from_spec(render_spec)
    sys.modules[f"{pkg_name}._render"] = render
    render_spec.loader.exec_module(render)

    return engine, render


def _load_example(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"_anim_example_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def discover_examples() -> list[Path]:
    paths: list[Path] = []
    for sub in ("discretizers", "carvers"):
        sub_dir = ANIM_SRC / sub
        if sub_dir.exists():
            paths.extend(sorted(p for p in sub_dir.glob("*.py") if not p.name.startswith("_")))
    top = ANIM_SRC / "readme_full_pipeline.py"
    if top.exists():
        paths.append(top)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit non-zero if any committed SVG drifts")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    engine, render = _load_engine_and_renderer()

    paths = discover_examples()
    if not paths:
        print("no animation examples found", file=sys.stderr)
        return 1

    DualFrame = getattr(engine, "DualFrame", None)

    drift = 0
    for path in paths:
        ex = _load_example(path)
        frames = engine.build_animation(ex.FEATURE, ex.TARGET, ex.STOP_AFTER_STAGE)
        is_dual = DualFrame is not None and frames and isinstance(frames[0], DualFrame)
        if is_dual:
            svg = render.render_dual_svg(frames, ex.STOP_AFTER_STAGE)
        else:
            svg = render.render_svg(frames, ex.STOP_AFTER_STAGE)
        out = OUT_DIR / f"{ex.NAME}.svg"
        if args.check:
            current = out.read_text(encoding="utf-8") if out.exists() else ""
            if current != svg:
                print(f"DRIFT: {out.relative_to(ROOT)}")
                drift += 1
        else:
            out.write_text(svg, encoding="utf-8")
            print(f"wrote {out.relative_to(ROOT)}")
    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main())
