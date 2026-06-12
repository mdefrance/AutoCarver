"""Generate the AutoCarver logo assets (2026 redesign: "curve → bars").

The mark: a navy density-curve silhouette sliced by a vertical dashed gold cut
(the hero animation's "winning cut" idiom), with three palette bars coming off
it. Gold is reserved as the cut/decision accent across logo + animations.

Writes to ``docs/source/artwork/``:

  auto_carver_symbol{,_light,_dark}.svg    glyph only
  auto_carver_logo{,_light,_dark}.svg      glyph + "AutoCarver" wordmark
  repository-logo.png                      1280x640 GitHub social preview

The no-suffix SVGs are **self-adapting**: a ``prefers-color-scheme: dark``
media query inside the SVG flips the ink colour, which works even when the
SVG is loaded through ``<img>``. The wordmark is converted to outline paths
(matplotlib TextPath + the local Segoe UI files) so it renders identically
everywhere — SVG ``<text>`` would fall back to the viewer's fonts.

Usage::

    uv run python docs/build_logo.py

Requires matplotlib (wordmark outlines, ships with the ``jupyter`` extra) and
a local Chrome/Edge install (social-preview PNG rasterization).
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ARTWORK = Path(__file__).resolve().parent / "source" / "artwork"

NAVY = "#2E3A59"
NAVY_DARKMODE = "#E2E8F0"  # ink flips to this on dark backgrounds
BLUE = "#5B8FF9"
GREEN = "#5AD8A6"
CYAN = "#6DC8EC"
GOLD = "#F6BD16"

VIEW_W, VIEW_H = 260, 200
BASELINE = 172.0

# Self-adapting single file: ink colour follows the viewer's colour scheme.
ADAPTIVE_STYLE = (
    "  <style>\n"
    f"    :root {{ --ink: {NAVY}; }}\n"
    f"    @media (prefers-color-scheme: dark) {{ :root {{ --ink: {NAVY_DARKMODE}; }} }}\n"
    "  </style>\n"
)

WORDMARK_SIZE = 64
WORDMARK_GAP = 30  # gap between glyph and wordmark
FONT_REGULAR = "C:/Windows/Fonts/segoeui.ttf"
FONT_BOLD = "C:/Windows/Fonts/segoeuib.ttf"

SOCIAL_W, SOCIAL_H = 1280, 640  # GitHub social-preview size
SOCIAL_BG = "#0f172a"
SOCIAL_TAGLINE = "Optimal supervised feature discretization"


# --- Glyph ---------------------------------------------------------------------


def _curve_points(x0: float, x1: float, n: int = 120) -> list[tuple[float, float]]:
    """Asymmetric-gaussian density silhouette: sharp rise, heavy right tail —
    keeps real height under the right-side bars (peak 128 at x = 78)."""
    xs = np.linspace(x0, x1, n)
    peak_x, w_left, w_right = 78.0, 40.0, 92.0
    w = np.where(xs < peak_x, w_left, w_right)
    pdf = 128.0 * np.exp(-((xs - peak_x) ** 2) / (2 * w**2))
    return [(float(x), float(h)) for x, h in zip(xs, pdf)]


CURVE = _curve_points(16, 244)


def _curve_h(x: float) -> float:
    xs = np.array([p[0] for p in CURVE])
    hs = np.array([p[1] for p in CURVE])
    return float(np.interp(x, xs, hs))


def _glyph_parts(ink: str) -> list[str]:
    """Curve silhouette + palette bars + baseline + dashed gold cut."""
    x_cut = 118.0
    pts = [(x, h) for x, h in CURVE if x <= x_cut]
    d = [f"M {pts[0][0]:.1f},{BASELINE:.1f}"]
    d += [f"L {x:.1f},{BASELINE - h:.1f}" for x, h in pts]
    d.append(f"L {x_cut:.1f},{BASELINE - _curve_h(x_cut):.1f}")
    d.append(f"L {x_cut:.1f},{BASELINE:.1f} Z")
    left = f'<path d="{" ".join(d)}" fill="{ink}"/>'

    bars = []
    for (bx0, bx1), color in (((128, 164), BLUE), ((172, 208), GREEN), ((216, 244), CYAN)):
        h = _curve_h((bx0 + bx1) / 2)
        bars.append(
            f'<rect x="{bx0}" y="{BASELINE - h:.1f}" width="{bx1 - bx0}" height="{h:.1f}" rx="6" fill="{color}"/>'
        )
    base = f'<rect x="12" y="{BASELINE:.0f}" width="236" height="6" rx="3" fill="{ink}"/>'
    cut = (
        f'<line x1="123" y1="22" x2="123" y2="{BASELINE - 2:.0f}" '
        f'stroke="{GOLD}" stroke-width="4" stroke-dasharray="8 6" stroke-linecap="round"/>'
    )
    return [left, *bars, base, cut]


def symbol(ink: str, style: str = "") -> str:
    body = "  " + "\n  ".join(_glyph_parts(ink))
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {VIEW_W} {VIEW_H}" '
        f'width="{VIEW_W}" height="{VIEW_H}">\n{style}{body}\n</svg>\n'
    )


# --- Wordmark lockup -------------------------------------------------------------


def _word_path(text: str, fname: str) -> tuple[str, float]:
    """SVG path `d` for `text` (y-up coords, baseline at 0) + its ink width."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.path import Path as MplPath
    from matplotlib.textpath import TextPath

    tp = TextPath((0, 0), text, size=WORDMARK_SIZE, prop=FontProperties(fname=fname))
    parts: list[str] = []
    for verts, code in tp.iter_segments():
        if code == MplPath.MOVETO:
            parts.append(f"M {verts[0]:.1f},{verts[1]:.1f}")
        elif code == MplPath.LINETO:
            parts.append(f"L {verts[0]:.1f},{verts[1]:.1f}")
        elif code == MplPath.CURVE3:
            parts.append(f"Q {verts[0]:.1f},{verts[1]:.1f} {verts[2]:.1f},{verts[3]:.1f}")
        elif code == MplPath.CURVE4:
            parts.append(f"C {verts[0]:.1f},{verts[1]:.1f} {verts[2]:.1f},{verts[3]:.1f} {verts[4]:.1f},{verts[5]:.1f}")
        elif code == MplPath.CLOSEPOLY:
            parts.append("Z")
    return " ".join(parts), float(tp.get_extents().xmax)


def lockup(ink: str, style: str = "") -> str:
    """Glyph + 'AutoCarver' wordmark (Auto regular, Carver bold) to the right."""
    auto_d, auto_w = _word_path("Auto", FONT_REGULAR)
    carver_d, carver_w = _word_path("Carver", FONT_BOLD)

    text_x0 = VIEW_W + WORDMARK_GAP
    baseline_y = 128.0  # optically centred on the glyph (spans ~14..178)
    carver_x0 = text_x0 + auto_w + 4
    view_w = carver_x0 + carver_w + 16

    body = "\n  ".join(
        [
            *_glyph_parts(ink),
            f'<g transform="translate({text_x0:.0f},{baseline_y:.0f}) scale(1,-1)">'
            f'<path d="{auto_d}" fill="{ink}"/></g>',
            f'<g transform="translate({carver_x0:.1f},{baseline_y:.0f}) scale(1,-1)">'
            f'<path d="{carver_d}" fill="{ink}"/></g>',
        ]
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {view_w:.0f} {VIEW_H}" '
        f'width="{view_w:.0f}" height="{VIEW_H}">\n{style}  {body}\n</svg>\n'
    )


# --- Social preview PNG ----------------------------------------------------------


def _find_browser() -> str | None:
    """Locate Chrome or Edge via the Windows App Paths registry + known paths."""
    try:
        import winreg

        for exe in ("chrome.exe", "msedge.exe"):
            try:
                key = winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    rf"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{exe}",
                )
                path, _ = winreg.QueryValueEx(key, None)
                if path and Path(path).exists():
                    return path
            except OSError:
                continue
    except ImportError:
        pass
    for path in (
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    ):
        if Path(path).exists():
            return path
    return None


def social_preview_png(out: Path) -> bool:
    """Rasterize the dark lockup on a slate card via a headless browser."""
    browser = _find_browser()
    if browser is None:
        print("WARNING: no Chrome/Edge found - skipping social preview PNG", file=sys.stderr)
        return False

    html = f"""<!doctype html><html><head><style>
  body {{ margin: 0; width: {SOCIAL_W}px; height: {SOCIAL_H}px; background: {SOCIAL_BG};
         display: flex; flex-direction: column; align-items: center; justify-content: center;
         gap: 38px; font-family: "Segoe UI", sans-serif; }}
  .tagline {{ color: #94a3b8; font-size: 32px; }}
</style></head><body>
<img src="auto_carver_logo_dark.svg" width="880">
<div class="tagline">{SOCIAL_TAGLINE}</div>
</body></html>
"""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        (tmp_dir / "auto_carver_logo_dark.svg").write_text(
            (ARTWORK / "auto_carver_logo_dark.svg").read_text(encoding="utf-8"), encoding="utf-8"
        )
        page = tmp_dir / "social.html"
        page.write_text(html, encoding="utf-8")
        subprocess.run(
            [
                browser,
                "--headless",
                "--disable-gpu",
                f"--window-size={SOCIAL_W},{SOCIAL_H}",
                f"--screenshot={out}",
                str(page),
            ],
            check=True,
            capture_output=True,
        )
    return True


# --- Entry point -----------------------------------------------------------------


def main() -> int:
    ARTWORK.mkdir(parents=True, exist_ok=True)
    for name, build in (("auto_carver_symbol", symbol), ("auto_carver_logo", lockup)):
        (ARTWORK / f"{name}_light.svg").write_text(build(NAVY), encoding="utf-8")
        (ARTWORK / f"{name}_dark.svg").write_text(build(NAVY_DARKMODE), encoding="utf-8")
        (ARTWORK / f"{name}.svg").write_text(build("var(--ink)", style=ADAPTIVE_STYLE), encoding="utf-8")
        print(f"wrote {name}{{,_light,_dark}}.svg")
    if social_preview_png(ARTWORK / "repository-logo.png"):
        print(f"wrote repository-logo.png ({SOCIAL_W}x{SOCIAL_H})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
