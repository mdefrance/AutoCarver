"""Render a Frame sequence to an animated SVG (CSS keyframes, passive loop).

Per-stage layout (top → bottom):
  1. Stage caption + metric chip
  2. One-line callout
  3. Main display — either a filled density curve (Stage 0/1) or value-positioned
     frequency bars (Stage 2). One display per stage, no duplicate strips.
  4. X-axis tick labels (Stage 2 cut-point values)

CSS keyframes cross-fade between stage <g> groups on a passive infinite loop.
"""

from __future__ import annotations

from textwrap import dedent

from ._engine import Frame

# --- Layout ------------------------------------------------------------------
VIEW_W = 800
VIEW_H = 340
PAD_X = 32
PAD_Y_TOP = 12

TITLE_H = 26
GAP_AFTER_TITLE = 6
CALLOUT_H = 30
GAP_AFTER_CALLOUT = 14
MAIN_H = 160
GAP_AFTER_MAIN = 6
TICK_H = 80  # room for rotated -45° labels
# Top portion of MAIN_H reserved for target-rate dots. Sized generously (≈40 %
# of MAIN_H) so even narrow rate spreads stay readable after zooming, since
# the strip is the only place small target-rate differences can be seen.
TARGET_STRIP_H = 66

_TITLE_Y0 = PAD_Y_TOP
_CALLOUT_Y0 = _TITLE_Y0 + TITLE_H + GAP_AFTER_TITLE
_MAIN_Y0 = _CALLOUT_Y0 + CALLOUT_H + GAP_AFTER_CALLOUT
_MAIN_BASELINE = _MAIN_Y0 + MAIN_H
_TICK_Y0 = _MAIN_BASELINE + GAP_AFTER_MAIN

STRIP_W = VIEW_W - 2 * PAD_X
NAN_W = 56
BIN_W = STRIP_W - NAN_W - 10  # gutter before NaN strip

PALETTE = {
    0: "#cfd2d6",
    1: "#5B8FF9",
    2: "#5AD8A6",
    3: "#F6BD16",
    4: "#E8684A",
    5: "#6DC8EC",
    6: "#9270CA",
    7: "#5D7092",
}
HIGHLIGHT_COLOR = "#f97316"
CURVE_FILL = "#cfd2d6"
CURVE_STROKE = "#6b7280"
MIN_BAR_W = 8  # singleton (over-rep) bars get this width

N_STAGES = 5  # max stage count across animations (QuantitativeDiscretizer = 5)
STAGE_MS = 5000
FADE_MS = 450


def render_svg(frames: list[Frame], stop_after_stage: int) -> str:
    total_stages = stop_after_stage
    stage_groups = "\n".join(_render_stage(f, total_stages) for f in frames)
    style = _render_style(stop_after_stage)
    return _DOC.format(view_w=VIEW_W, view_h=VIEW_H, style=style, stage_groups=stage_groups)


# --- Stage rendering ----------------------------------------------------------


def _render_stage(frame: Frame, total_stages: int) -> str:
    parts = [
        _stage_caption(frame, total_stages),
        _metric_chip(frame),
        _callout(frame),
        _main_display(frame),
        _target_dots(frame),
        _merge_arrows(frame),
        _nan_strip(frame),
        _baseline_line(),
        _min_freq_line(frame),
    ]
    inner = "\n".join(p for p in parts if p)
    return f'  <g class="stage stage-{frame.stage}">\n{inner}\n  </g>'


def _bar_zone_h(frame: Frame) -> float:
    """When a target-rate strip is shown, the bar zone is the lower portion
    of MAIN_H; otherwise bars use the full MAIN_H."""
    if frame.target_strip_max is not None:
        return MAIN_H - TARGET_STRIP_H
    return MAIN_H


def _min_freq_line(frame: Frame) -> str:
    if frame.min_freq_y_norm is None:
        return ""
    y_norm = max(0.0, min(1.0, frame.min_freq_y_norm))
    y = _MAIN_BASELINE - y_norm * _bar_zone_h(frame)
    line = (
        f'<line x1="{PAD_X}" y1="{y:.2f}" '
        f'x2="{VIEW_W - PAD_X}" y2="{y:.2f}" '
        f'stroke="#ef4444" stroke-width="1" stroke-dasharray="5 3"/>'
    )
    label = (
        f'<text class="threshold-label" x="{VIEW_W - PAD_X - 4:.2f}" '
        f'y="{y - 4:.2f}" text-anchor="end">{_escape(frame.min_freq_label)}</text>'
    )
    return "    " + line + "\n    " + label


def _stage_caption(frame: Frame, total_stages: int) -> str:
    return (
        f'    <text class="title" x="{VIEW_W / 2:.2f}" y="{_TITLE_Y0 + 17:.2f}" '
        f'text-anchor="middle">Stage {frame.stage} / {total_stages} '
        f"&#8212; {_escape(frame.title)}</text>"
    )


def _metric_chip(frame: Frame) -> str:
    chip_w, chip_h = 110, 22
    chip_x = VIEW_W - PAD_X - chip_w
    chip_y = _TITLE_Y0
    return (
        f'    <rect x="{chip_x}" y="{chip_y}" width="{chip_w}" height="{chip_h}" '
        f'rx="11" fill="#111827"/>\n'
        f'    <text class="chip" x="{chip_x + chip_w / 2:.2f}" '
        f'y="{chip_y + chip_h / 2 + 4:.2f}" text-anchor="middle">'
        f"{_escape(frame.metric)}</text>"
    )


def _callout(frame: Frame) -> str:
    if not frame.callout:
        return ""
    line1, line2 = _split_callout(frame.callout, max_chars=95)
    parts = [
        f'<text class="callout" x="{VIEW_W / 2:.2f}" y="{_CALLOUT_Y0 + 12:.2f}" '
        f'text-anchor="middle">{_escape(line1)}</text>'
    ]
    if line2:
        parts.append(
            f'<text class="callout" x="{VIEW_W / 2:.2f}" y="{_CALLOUT_Y0 + 26:.2f}" '
            f'text-anchor="middle">{_escape(line2)}</text>'
        )
    return "    " + "\n    ".join(parts)


def _main_display(frame: Frame) -> str:
    """Density curve + numeric x-axis ticks (Stages 0/1), or value-axis bars
    with rotated modality labels (Stage 2)."""
    parts: list[str] = []
    if frame.density_curve:
        parts.append(_density_path(frame.density_curve))
        for x_norm, label in frame.overrep_markers:
            parts.append(_overrep_marker(x_norm, label))
        for x_norm, text in frame.tick_values:
            parts.append(_xaxis_tick(x_norm, text))
        return "    " + "\n    ".join(parts)
    if frame.bins:
        return _freq_bars(frame)
    return ""


def _rotated_label(cx: float, label: str) -> str:
    cy = _MAIN_BASELINE + 8
    return (
        f'<text class="binlabel-rot" x="{cx:.2f}" y="{cy:.2f}" '
        f'text-anchor="end" transform="rotate(-45 {cx:.2f} {cy:.2f})">'
        f"{_escape(label)}</text>"
    )


def _xaxis_tick(x_norm: float, text: str) -> str:
    """Small tick mark + horizontal numeric label below the baseline."""
    x = PAD_X + x_norm * BIN_W
    tick = (
        f'<line x1="{x:.2f}" y1="{_MAIN_BASELINE:.2f}" x2="{x:.2f}" '
        f'y2="{_MAIN_BASELINE + 4:.2f}" stroke="#6b7280" stroke-width="1"/>'
    )
    label = (
        f'<text class="tick-label" x="{x:.2f}" y="{_MAIN_BASELINE + 16:.2f}" '
        f'text-anchor="middle">{_escape(text)}</text>'
    )
    return tick + "\n    " + label


def _density_path(curve: tuple[tuple[float, float], ...]) -> str:
    """Filled grey path under the density curve."""
    if not curve:
        return ""
    pts = [(PAD_X + x * BIN_W, _MAIN_BASELINE - y * MAIN_H) for x, y in curve]
    d = [f"M {pts[0][0]:.2f},{_MAIN_BASELINE:.2f}"]
    d += [f"L {x:.2f},{y:.2f}" for x, y in pts]
    d.append(f"L {pts[-1][0]:.2f},{_MAIN_BASELINE:.2f} Z")
    return (
        f'<path d="{" ".join(d)}" fill="{CURVE_FILL}" stroke="{CURVE_STROKE}" '
        f'stroke-width="1.2" stroke-linejoin="round"/>'
    )


def _overrep_marker(x_norm: float, label: str) -> str:
    """Vertical dashed orange line + label at a specific x along the main strip."""
    x = PAD_X + x_norm * BIN_W
    line = (
        f'<line x1="{x:.2f}" y1="{_MAIN_Y0 - 4:.2f}" x2="{x:.2f}" '
        f'y2="{_MAIN_BASELINE:.2f}" stroke="{HIGHLIGHT_COLOR}" stroke-width="2" '
        f'stroke-dasharray="5 3"/>'
    )
    lbl = (
        f'<text class="overrep-label" x="{x + 6:.2f}" y="{_MAIN_Y0 + 12:.2f}" '
        f'text-anchor="start">{_escape(label)}</text>'
    )
    return line + "\n    " + lbl


def _freq_bars(frame: Frame) -> str:
    """Bars at their value-axis or equal-slot x positions. Widths reflect each
    modality's range; heights ∝ real frequency. Labels rotated below. Bars in
    `frame.highlight_bins` get an orange outline.
    """
    if not frame.bins:
        return ""
    max_freq = frame.bar_max_freq if frame.bar_max_freq > 0 else max(b.freq for b in frame.bins)
    bar_zone_h = _bar_zone_h(frame)
    parts: list[str] = []
    bar_pad = 1.0  # px gutter between bars
    for bin_idx, b in enumerate(frame.bins):
        x = PAD_X + b.x_start * BIN_W + bar_pad / 2
        w = max((b.x_end - b.x_start) * BIN_W - bar_pad, 1.0)
        h = (b.freq / max_freq) * bar_zone_h if max_freq > 0 else 0.0
        y = _MAIN_BASELINE - h
        is_highlight = bin_idx in frame.highlight_bins
        stroke = HIGHLIGHT_COLOR if is_highlight else "#374151"
        stroke_w = 2.0 if is_highlight else 0.6
        parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
            f'rx="2" fill="{PALETTE[b.color_id]}" stroke="{stroke}" '
            f'stroke-width="{stroke_w}"/>'
        )
        # Label below the bar centre, rotated -45° (supports two lines).
        cx = PAD_X + ((b.x_start + b.x_end) / 2) * BIN_W
        cy = _MAIN_BASELINE + 8
        lines = b.label.split("\n")
        if len(lines) == 1:
            parts.append(
                f'<text class="binlabel-rot" x="{cx:.2f}" y="{cy:.2f}" '
                f'text-anchor="end" transform="rotate(-45 {cx:.2f} {cy:.2f})">'
                f"{_escape(lines[0])}</text>"
            )
        else:
            tspans = "".join(
                f'<tspan x="{cx:.2f}" dy="{0 if li == 0 else 11}">{_escape(s)}</tspan>' for li, s in enumerate(lines)
            )
            parts.append(
                f'<text class="binlabel-rot" x="{cx:.2f}" y="{cy:.2f}" '
                f'text-anchor="end" transform="rotate(-45 {cx:.2f} {cy:.2f})">'
                f"{tspans}</text>"
            )
    return "    " + "\n    ".join(parts)


def _merge_arrows(frame: Frame) -> str:
    """Curved dashed orange arrows arcing over the bar zone from each absorbed
    modality's bar centre to its merge target's bar centre. Drawn high enough
    in the bar zone to avoid clipping tall bars; arrowhead at the target end.
    """
    if not frame.merge_arrows:
        return ""
    bar_zone_h = _bar_zone_h(frame)
    bar_zone_top = _MAIN_BASELINE - bar_zone_h
    y_anchor = bar_zone_top + bar_zone_h * 0.18  # near the top of the bar zone
    y_arc = bar_zone_top + bar_zone_h * 0.02
    parts: list[str] = []
    for arr in frame.merge_arrows:
        x1 = PAD_X + arr.from_x * BIN_W
        x2 = PAD_X + arr.to_x * BIN_W
        mx = (x1 + x2) / 2
        # Quadratic Bezier; stop the path 6px short of the bar centre so the
        # arrowhead doesn't overlap the bar outline.
        dx = x2 - x1
        if dx == 0:
            continue
        shrink = 6.0
        x2_end = x2 - shrink * (1 if dx > 0 else -1)
        d = f"M {x1:.2f},{y_anchor:.2f} Q {mx:.2f},{y_arc:.2f} {x2_end:.2f},{y_anchor:.2f}"
        parts.append(
            f'<path d="{d}" fill="none" stroke="{HIGHLIGHT_COLOR}" '
            f'stroke-width="1.5" stroke-dasharray="4 3" '
            f'marker-end="url(#arrowhead)"/>'
        )
        parts.append(
            f'<text class="merge-label" x="{mx:.2f}" y="{y_arc - 2:.2f}" '
            f'text-anchor="middle">{_escape(arr.label)}</text>'
        )
    return "    " + "\n    ".join(parts)


def _target_dots(frame: Frame) -> str:
    """Top strip of MAIN_H: one filled circle per bin, connected by a light
    line, scaled to [`target_strip_min`, `target_strip_max`]. The range is
    zoomed to the actual rates (not anchored at 0) so small differences read
    clearly; the caption shows the displayed range."""
    if frame.target_strip_max is None or not frame.bins:
        return ""
    strip_top = _MAIN_Y0 + 6
    strip_bot = _MAIN_Y0 + TARGET_STRIP_H - 8
    tmin = frame.target_strip_min
    tmax = frame.target_strip_max
    span = tmax - tmin if tmax > tmin else 1.0
    pts: list[tuple[float, float, int]] = []
    for _i, b in enumerate(frame.bins):
        if b.freq <= 0:
            # Empty bins have no observed target rate; painting a dot at 0
            # would be misleading (looks like "0 % target") and drags the
            # dashed connector to the strip floor.
            continue
        cx = PAD_X + ((b.x_start + b.x_end) / 2) * BIN_W
        cy = strip_bot - ((b.target - tmin) / span) * (strip_bot - strip_top)
        pts.append((cx, cy, b.color_id))
    parts: list[str] = []
    if len(pts) >= 2:
        d = " ".join(("M" if i == 0 else "L") + f" {x:.2f},{y:.2f}" for i, (x, y, _c) in enumerate(pts))
        parts.append(f'<path d="{d}" fill="none" stroke="#9ca3af" stroke-width="1" stroke-dasharray="2 3"/>')
    for x, y, c in pts:
        parts.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{PALETTE[c]}" stroke="#374151" stroke-width="0.6"/>'
        )
    # Strip caption (left edge) shows the zoomed range so readers can tell
    # that the strip's bottom edge isn't necessarily 0.
    parts.append(
        f'<text class="strip-label" x="{PAD_X:.2f}" y="{_MAIN_Y0 + 8:.2f}" '
        f'text-anchor="start">target rate &#8712; [{tmin:.2f}, {tmax:.2f}]'
        f"</text>"
    )
    return "    " + "\n    ".join(parts)


def _baseline_line() -> str:
    return (
        f'    <line x1="{PAD_X}" y1="{_MAIN_BASELINE:.2f}" '
        f'x2="{PAD_X + BIN_W}" y2="{_MAIN_BASELINE:.2f}" '
        f'stroke="#9ca3af" stroke-width="1"/>'
    )


def _nan_strip(frame: Frame) -> str:
    """NaN visualisation on the far right of every stage. Height ∝ freq via the
    same `bar_max_freq` scale as the Stage 2 bars (so the morph is consistent).
    """
    if frame.nan_bin is None:
        return ""
    x_nan = PAD_X + BIN_W + 10
    max_freq = frame.bar_max_freq if frame.bar_max_freq > 0 else 1.0
    h = (frame.nan_bin.freq / max_freq) * _bar_zone_h(frame) if max_freq > 0 else 0.0
    h = max(h, 6.0)  # always at least visible
    y = _MAIN_BASELINE - h
    rect = (
        f'<rect x="{x_nan:.2f}" y="{y:.2f}" width="{NAN_W:.2f}" '
        f'height="{h:.2f}" rx="3" fill="url(#nan-hatch)" '
        f'stroke="{PALETTE[7]}" stroke-width="1" opacity="0.85"/>'
    )
    # NaN label below, rotated like the bin labels for visual consistency
    anchor_x = x_nan + NAN_W / 2
    anchor_y = _MAIN_BASELINE + 8
    lbl_text = f"NaN ({frame.nan_bin.freq * 100:.1f}%)"
    lbl = (
        f'<text class="binlabel-rot" x="{anchor_x:.2f}" y="{anchor_y:.2f}" '
        f'text-anchor="end" transform="rotate(-45 {anchor_x:.2f} '
        f'{anchor_y:.2f})">{_escape(lbl_text)}</text>'
    )
    return "    " + rect + "\n    " + lbl


# --- CSS keyframes ------------------------------------------------------------


def _render_style(stop_after_stage: int) -> str:
    n_visible = stop_after_stage + 1
    total = n_visible * STAGE_MS + STAGE_MS  # +end-pause
    fade_pct = (FADE_MS / total) * 100
    stage_pct = (STAGE_MS / total) * 100

    rules = []
    for k in range(N_STAGES):
        if k > stop_after_stage:
            rules.append(f".stage-{k} {{ display: none; }}")
            continue
        start = k * stage_pct
        end = (k + 1) * stage_pct
        is_last = k == stop_after_stage
        kf_name = f"show-{k}"
        if is_last:
            kf_body = _kf_last(start, fade_pct)
        elif k == 0:
            kf_body = _kf_first(end, fade_pct)
        else:
            kf_body = _kf_middle(start, end, fade_pct)
        rules.append(
            f"@keyframes {kf_name} {{\n{kf_body}\n}}\n"
            f".stage-{k} {{ opacity: 0; animation: {kf_name} {total}ms infinite; }}"
        )

    base = dedent(
        """\
        text { font-family: -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif; }
        .title { font-size: 14px; font-weight: 600; fill: #111827; }
        .callout { font-size: 11px; fill: #4b5563; }
        .chip { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 12px; fill: #f9fafb; }
        .binlabel { font-size: 10px; fill: #374151; }
        .binlabel-rot { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 10px; fill: #374151; }
        .tick-label { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 9px; fill: #6b7280; }
        .overrep-label { font-size: 11px; fill: #c2410c; font-weight: 600; }
        .merge-label { font-size: 9px; fill: #c2410c; font-weight: 600; }
        .strip-label { font-size: 9px; fill: #6b7280; font-style: italic; }
        .threshold-label {
            font-family: ui-monospace, "SF Mono", Menlo, monospace;
            font-size: 10px; fill: #ef4444; font-weight: 600;
        }
        """
    )
    return base + "\n" + "\n".join(rules)


def _kf_first(end: float, fade: float) -> str:
    out_start = max(end - fade, 0.0)
    return (
        f"  0% {{ opacity: 1; }}\n"
        f"  {out_start:.2f}% {{ opacity: 1; }}\n"
        f"  {end:.2f}% {{ opacity: 0; }}\n"
        f"  100% {{ opacity: 0; }}"
    )


def _kf_middle(start: float, end: float, fade: float) -> str:
    in_done = min(start + fade, end)
    out_start = max(end - fade, in_done)
    return (
        f"  0% {{ opacity: 0; }}\n"
        f"  {start:.2f}% {{ opacity: 0; }}\n"
        f"  {in_done:.2f}% {{ opacity: 1; }}\n"
        f"  {out_start:.2f}% {{ opacity: 1; }}\n"
        f"  {end:.2f}% {{ opacity: 0; }}\n"
        f"  100% {{ opacity: 0; }}"
    )


def _kf_last(start: float, fade: float) -> str:
    in_done = min(start + fade, 100.0)
    return (
        f"  0% {{ opacity: 0; }}\n"
        f"  {start:.2f}% {{ opacity: 0; }}\n"
        f"  {in_done:.2f}% {{ opacity: 1; }}\n"
        f"  100% {{ opacity: 1; }}"
    )


# --- Utilities ---------------------------------------------------------------


def _split_callout(text: str, max_chars: int) -> tuple[str, str]:
    if len(text) <= max_chars:
        return text, ""
    cut = text.rfind(" ", 0, max_chars)
    if cut == -1:
        cut = max_chars
    return text[:cut].rstrip(), text[cut:].lstrip()


def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_DOC = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {view_w} {view_h}"
     width="{view_w}" height="{view_h}" role="img"
     aria-label="AutoCarver discretizer animation">
  <defs>
    <pattern id="nan-hatch" patternUnits="userSpaceOnUse" width="6" height="6"
             patternTransform="rotate(45)">
      <rect width="6" height="6" fill="#e5e7eb"/>
      <line x1="0" y1="0" x2="0" y2="6" stroke="#5D7092" stroke-width="1.2"/>
    </pattern>
    <marker id="arrowhead" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 Z" fill="#f97316"/>
    </marker>
  </defs>
  <style>
{style}
  </style>
{stage_groups}
</svg>
"""
