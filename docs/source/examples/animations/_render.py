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

# Hero-only colour themes: the README hero animation (readme_full_pipeline)
# ships light + dark variants for GitHub's colour-scheme toggle. Every other
# animation is embedded in always-light RTD pages and never passes `theme`,
# so its output is unaffected (defaults reproduce the previous hardcoded
# colours exactly).
_LIGHT_THEME = {
    "title": "#111827",
    "callout": "#4b5563",
    "chip_bg": "#111827",
    "binlabel": "#374151",
    "tick_label": "#6b7280",
    "strip_label": "#6b7280",
    "threshold": "#ef4444",
    "overrep": "#c2410c",
    "merge": "#c2410c",
    "curve_fill": CURVE_FILL,
    "curve_stroke": CURVE_STROKE,
    "axis_stroke": "#9ca3af",
    "tick_stroke": "#6b7280",
    "bar_stroke": "#374151",
    "sep_stroke": "#d1d5db",
    "gold_bg": "#fef3c7",  # == GOLD_BG
    "gold_stroke": "#d97706",  # == GOLD_STROKE
    "nan_hatch_bg": "#e5e7eb",
    "nan_hatch_stroke": PALETTE[7],
}

_DARK_THEME = {
    "title": "#e6edf3",
    "callout": "#9ca3af",
    "chip_bg": "#30363d",
    "binlabel": "#c9d1d9",
    "tick_label": "#8b949e",
    "strip_label": "#8b949e",
    "threshold": "#f87171",
    "overrep": "#fb923c",
    "merge": "#fb923c",
    "curve_fill": "#768491",
    "curve_stroke": "#c9d1d9",
    "axis_stroke": "#484f58",
    "tick_stroke": "#6e7681",
    "bar_stroke": "#6e7681",
    "sep_stroke": "#21262d",
    "gold_bg": "#3b2f10",
    "gold_stroke": "#f6bd16",
    "nan_hatch_bg": "#30363d",
    "nan_hatch_stroke": "#8b949e",
}

# Hero light variant: like _LIGHT_THEME but with a single neutral mid-grey for
# every body-text element (title/callout/labels) instead of near-black, so the
# light SVG stays legible even when a viewer that ignores <picture> (or hasn't
# loaded the dark file) shows it on a dark background. Semantic marker colours
# (threshold red, over-rep/merge orange) keep their meaning.
_HERO_LIGHT_THEME = {
    **_LIGHT_THEME,
    "title": "#6b7280",
    "callout": "#6b7280",
    "binlabel": "#6b7280",
    "tick_label": "#6b7280",
    "strip_label": "#6b7280",
}


def render_svg(frames: list[Frame], stop_after_stage: int) -> str:
    total_stages = stop_after_stage
    stage_groups = "\n".join(_render_stage(f, total_stages) for f in frames)
    style = _render_style(stop_after_stage)
    return _doc(VIEW_W, VIEW_H, style, stage_groups)


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


def _min_freq_line(frame: Frame, theme: dict = _LIGHT_THEME) -> str:
    if frame.min_freq_y_norm is None:
        return ""
    y_norm = max(0.0, min(1.0, frame.min_freq_y_norm))
    y = _MAIN_BASELINE - y_norm * _bar_zone_h(frame)
    line = (
        f'<line x1="{PAD_X}" y1="{y:.2f}" '
        f'x2="{VIEW_W - PAD_X}" y2="{y:.2f}" '
        f'stroke="{theme["threshold"]}" stroke-width="1" stroke-dasharray="5 3"/>'
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


def _metric_chip(frame: Frame, theme: dict = _LIGHT_THEME) -> str:
    chip_w, chip_h = 110, 22
    chip_x = VIEW_W - PAD_X - chip_w
    chip_y = _TITLE_Y0
    return (
        f'    <rect x="{chip_x}" y="{chip_y}" width="{chip_w}" height="{chip_h}" '
        f'rx="11" fill="{theme["chip_bg"]}"/>\n'
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


def _main_display(frame: Frame, theme: dict = _LIGHT_THEME) -> str:
    """Density curve + numeric x-axis ticks (Stages 0/1), or value-axis bars
    with rotated modality labels (Stage 2)."""
    parts: list[str] = []
    if frame.density_curve:
        parts.append(_density_path(frame.density_curve, theme))
        for x_norm, label in frame.overrep_markers:
            parts.append(_overrep_marker(x_norm, label))
        for x_norm, text in frame.tick_values:
            parts.append(_xaxis_tick(x_norm, text, theme))
        return "    " + "\n    ".join(parts)
    if frame.bins:
        return _freq_bars(frame, theme)
    return ""


def _rotated_label(cx: float, label: str) -> str:
    cy = _MAIN_BASELINE + 8
    return (
        f'<text class="binlabel-rot" x="{cx:.2f}" y="{cy:.2f}" '
        f'text-anchor="end" transform="rotate(-45 {cx:.2f} {cy:.2f})">'
        f"{_escape(label)}</text>"
    )


def _xaxis_tick(x_norm: float, text: str, theme: dict = _LIGHT_THEME) -> str:
    """Small tick mark + horizontal numeric label below the baseline."""
    x = PAD_X + x_norm * BIN_W
    tick = (
        f'<line x1="{x:.2f}" y1="{_MAIN_BASELINE:.2f}" x2="{x:.2f}" '
        f'y2="{_MAIN_BASELINE + 4:.2f}" stroke="{theme["tick_stroke"]}" stroke-width="1"/>'
    )
    label = (
        f'<text class="tick-label" x="{x:.2f}" y="{_MAIN_BASELINE + 16:.2f}" '
        f'text-anchor="middle">{_escape(text)}</text>'
    )
    return tick + "\n    " + label


def _density_path(curve: tuple[tuple[float, float], ...], theme: dict = _LIGHT_THEME) -> str:
    """Filled grey path under the density curve."""
    if not curve:
        return ""
    pts = [(PAD_X + x * BIN_W, _MAIN_BASELINE - y * MAIN_H) for x, y in curve]
    d = [f"M {pts[0][0]:.2f},{_MAIN_BASELINE:.2f}"]
    d += [f"L {x:.2f},{y:.2f}" for x, y in pts]
    d.append(f"L {pts[-1][0]:.2f},{_MAIN_BASELINE:.2f} Z")
    return (
        f'<path d="{" ".join(d)}" fill="{theme["curve_fill"]}" stroke="{theme["curve_stroke"]}" '
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


def _freq_bars(frame: Frame, theme: dict = _LIGHT_THEME) -> str:
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
        stroke = HIGHLIGHT_COLOR if is_highlight else theme["bar_stroke"]
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


def _target_dots(frame: Frame, theme: dict = _LIGHT_THEME) -> str:
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
        parts.append(
            f'<path d="{d}" fill="none" stroke="{theme["axis_stroke"]}" stroke-width="1" stroke-dasharray="2 3"/>'
        )
    for x, y, c in pts:
        parts.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{PALETTE[c]}" '
            f'stroke="{theme["bar_stroke"]}" stroke-width="0.6"/>'
        )
    # Strip caption (left edge) shows the zoomed range so readers can tell
    # that the strip's bottom edge isn't necessarily 0.
    parts.append(
        f'<text class="strip-label" x="{PAD_X:.2f}" y="{_MAIN_Y0 + 8:.2f}" '
        f'text-anchor="start">target rate &#8712; [{tmin:.2f}, {tmax:.2f}]'
        f"</text>"
    )
    return "    " + "\n    ".join(parts)


def _baseline_line(theme: dict = _LIGHT_THEME) -> str:
    return (
        f'    <line x1="{PAD_X}" y1="{_MAIN_BASELINE:.2f}" '
        f'x2="{PAD_X + BIN_W}" y2="{_MAIN_BASELINE:.2f}" '
        f'stroke="{theme["axis_stroke"]}" stroke-width="1"/>'
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


def _render_style(stop_after_stage: int, theme: dict = _LIGHT_THEME) -> str:
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

    binlabel = theme["binlabel"]
    tick_label = theme["tick_label"]
    base = dedent(
        f"""\
        text {{ font-family: -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif; }}
        .title {{ font-size: 14px; font-weight: 600; fill: {theme["title"]}; }}
        .callout {{ font-size: 11px; fill: {theme["callout"]}; }}
        .chip {{ font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 12px; fill: #f9fafb; }}
        .binlabel {{ font-size: 10px; fill: {theme["binlabel"]}; }}
        .binlabel-rot {{ font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 10px; fill: {binlabel}; }}
        .tick-label {{ font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 9px; fill: {tick_label}; }}
        .overrep-label {{ font-size: 11px; fill: {theme["overrep"]}; font-weight: 600; }}
        .merge-label {{ font-size: 9px; fill: {theme["merge"]}; font-weight: 600; }}
        .strip-label {{ font-size: 9px; fill: {theme["strip_label"]}; font-style: italic; }}
        .threshold-label {{
            font-family: ui-monospace, "SF Mono", Menlo, monospace;
            font-size: 10px; fill: {theme["threshold"]}; font-weight: 600;
        }}
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


# =============================================================================
# Dual-strip renderer (QualitativeDiscretizer)
# =============================================================================
#
# Two compressed feature strips stacked vertically: top = categorical (Port),
# bottom = ordinal (AgeGroup). Each strip is DUAL_MAIN_H tall (vs MAIN_H=160
# for the single-strip renderer). The header (title + callout + metric chip)
# is shared and uses the same y-positions as the single-strip renderer.

DUAL_VIEW_H = 490
DUAL_MAIN_H = 110
DUAL_TARGET_STRIP_H = 44  # top portion per strip for target-rate dots
# bar zone per strip = DUAL_MAIN_H - DUAL_TARGET_STRIP_H = 66 px

# Top strip occupies the same vertical slot as a single-strip animation so the
# header coordinates (_MAIN_Y0, _CALLOUT_Y0, _TITLE_Y0) are reused unchanged.
_D_TOP_Y0 = _MAIN_Y0  # 88 — starts right after the callout
_D_TOP_BASELINE = _D_TOP_Y0 + DUAL_MAIN_H  # 198

# 72 px below the top baseline leaves room for rotated -45° labels (which
# extend ~55–65 px downward) plus a small gap before the separator.
_D_SEP_Y = _D_TOP_BASELINE + 72  # 270 — horizontal separator line
_D_BOT_Y0 = _D_SEP_Y + 20  # 290 — bottom strip starts here
_D_BOT_BASELINE = _D_BOT_Y0 + DUAL_MAIN_H  # 400


def render_dual_svg(dual_frames: list, stop_after_stage: int) -> str:
    """Render a list of DualFrames to an animated SVG with two stacked strips."""
    total_stages = stop_after_stage
    stage_groups = "\n".join(_d_render_frame(df, total_stages) for df in dual_frames)
    style = _render_style(stop_after_stage)
    return _doc(VIEW_W, DUAL_VIEW_H, style, stage_groups)


def _d_render_frame(df, total_stages: int) -> str:
    """Render one DualFrame: shared header + two independent feature strips."""
    sep_label_y = _D_SEP_Y + 14
    parts = [
        _stage_caption(df, total_stages),
        _metric_chip(df),
        _callout(df),
        # Top strip (categorical)
        _d_render_strip(df.top, _D_TOP_Y0, _D_TOP_BASELINE, df.top_opacity),
        # Separator + bottom strip label
        f'    <line x1="{PAD_X}" y1="{_D_SEP_Y:.2f}" '
        f'x2="{PAD_X + BIN_W + 10 + NAN_W:.2f}" y2="{_D_SEP_Y:.2f}" '
        f'stroke="#d1d5db" stroke-width="1"/>',
        f'    <text class="strip-label" x="{PAD_X}" y="{sep_label_y:.2f}" '
        f'text-anchor="start">AgeGroup (ordinal)</text>',
        # Bottom strip (ordinal)
        _d_render_strip(df.bot, _D_BOT_Y0, _D_BOT_BASELINE, df.bot_opacity),
    ]
    inner = "\n".join(p for p in parts if p)
    return f'  <g class="stage stage-{df.stage}">\n{inner}\n  </g>'


def _d_render_strip(frame: Frame, y0: float, baseline: float, opacity: float = 1.0) -> str:
    """Render one feature strip at explicit vertical coordinates."""
    has_target = frame.target_strip_max is not None
    bar_zone_h = DUAL_MAIN_H - DUAL_TARGET_STRIP_H if has_target else DUAL_MAIN_H
    parts = [
        _d_target_dots(frame, y0, bar_zone_h),
        _d_freq_bars(frame, baseline, bar_zone_h),
        _d_merge_arrows(frame, baseline, bar_zone_h),
        _d_nan_strip(frame, baseline, bar_zone_h),
        f'<line x1="{PAD_X}" y1="{baseline:.2f}" '
        f'x2="{PAD_X + BIN_W}" y2="{baseline:.2f}" stroke="#9ca3af" stroke-width="1"/>',
        _d_min_freq_line(frame, baseline, bar_zone_h),
    ]
    inner = "\n    ".join(p for p in parts if p)
    return f'    <g opacity="{opacity:.2f}">\n    {inner}\n    </g>'


def _d_freq_bars(frame: Frame, baseline: float, bar_zone_h: float) -> str:
    if not frame.bins:
        return ""
    max_freq = frame.bar_max_freq if frame.bar_max_freq > 0 else max(b.freq for b in frame.bins)
    parts: list[str] = []
    bar_pad = 1.0
    for i, b in enumerate(frame.bins):
        x = PAD_X + b.x_start * BIN_W + bar_pad / 2
        w = max((b.x_end - b.x_start) * BIN_W - bar_pad, 1.0)
        h = (b.freq / max_freq) * bar_zone_h if max_freq > 0 else 0.0
        y = baseline - h
        is_hi = i in frame.highlight_bins
        stroke = HIGHLIGHT_COLOR if is_hi else "#374151"
        sw = 2.0 if is_hi else 0.6
        parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
            f'rx="2" fill="{PALETTE[b.color_id]}" stroke="{stroke}" stroke-width="{sw}"/>'
        )
        cx = PAD_X + ((b.x_start + b.x_end) / 2) * BIN_W
        cy = baseline + 8
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
    return "\n    ".join(parts)


def _d_target_dots(frame: Frame, y0: float, bar_zone_h: float) -> str:
    if frame.target_strip_max is None or not frame.bins:
        return ""
    strip_top = y0 + 6
    strip_bot = y0 + DUAL_TARGET_STRIP_H - 8
    tmin = frame.target_strip_min
    tmax = frame.target_strip_max
    span = tmax - tmin if tmax > tmin else 1.0
    pts: list[tuple[float, float, int]] = []
    for b in frame.bins:
        if b.freq <= 0:
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
    parts.append(
        f'<text class="strip-label" x="{PAD_X:.2f}" y="{y0 + 8:.2f}" '
        f'text-anchor="start">target rate &#8712; [{tmin:.2f}, {tmax:.2f}]</text>'
    )
    return "\n    ".join(parts)


def _d_merge_arrows(frame: Frame, baseline: float, bar_zone_h: float) -> str:
    if not frame.merge_arrows:
        return ""
    bar_zone_top = baseline - bar_zone_h
    y_anchor = bar_zone_top + bar_zone_h * 0.18
    y_arc = bar_zone_top + bar_zone_h * 0.02
    parts: list[str] = []
    for arr in frame.merge_arrows:
        x1 = PAD_X + arr.from_x * BIN_W
        x2 = PAD_X + arr.to_x * BIN_W
        mx = (x1 + x2) / 2
        dx = x2 - x1
        if dx == 0:
            continue
        shrink = 6.0
        x2_end = x2 - shrink * (1 if dx > 0 else -1)
        d = f"M {x1:.2f},{y_anchor:.2f} Q {mx:.2f},{y_arc:.2f} {x2_end:.2f},{y_anchor:.2f}"
        parts.append(
            f'<path d="{d}" fill="none" stroke="{HIGHLIGHT_COLOR}" '
            f'stroke-width="1.5" stroke-dasharray="4 3" marker-end="url(#arrowhead)"/>'
        )
        parts.append(
            f'<text class="merge-label" x="{mx:.2f}" y="{y_arc - 2:.2f}" '
            f'text-anchor="middle">{_escape(arr.label)}</text>'
        )
    return "\n    ".join(parts)


def _d_nan_strip(frame: Frame, baseline: float, bar_zone_h: float) -> str:
    if frame.nan_bin is None:
        return ""
    x_nan = PAD_X + BIN_W + 10
    max_freq = frame.bar_max_freq if frame.bar_max_freq > 0 else 1.0
    h = (frame.nan_bin.freq / max_freq) * bar_zone_h if max_freq > 0 else 0.0
    h = max(h, 6.0)
    y = baseline - h
    anchor_x = x_nan + NAN_W / 2
    anchor_y = baseline + 8
    return (
        f'<rect x="{x_nan:.2f}" y="{y:.2f}" width="{NAN_W:.2f}" height="{h:.2f}" '
        f'rx="3" fill="url(#nan-hatch)" stroke="{PALETTE[7]}" stroke-width="1" opacity="0.85"/>\n    '
        f'<text class="binlabel-rot" x="{anchor_x:.2f}" y="{anchor_y:.2f}" '
        f'text-anchor="end" transform="rotate(-45 {anchor_x:.2f} {anchor_y:.2f})">'
        f"NaN ({frame.nan_bin.freq * 100:.1f}%)</text>"
    )


def _d_min_freq_line(frame: Frame, baseline: float, bar_zone_h: float) -> str:
    if frame.min_freq_y_norm is None:
        return ""
    y_norm = max(0.0, min(1.0, frame.min_freq_y_norm))
    y = baseline - y_norm * bar_zone_h
    return (
        f'<line x1="{PAD_X}" y1="{y:.2f}" x2="{VIEW_W - PAD_X}" y2="{y:.2f}" '
        f'stroke="#ef4444" stroke-width="1" stroke-dasharray="5 3"/>\n    '
        f'<text class="threshold-label" x="{VIEW_W - PAD_X - 4:.2f}" y="{y - 4:.2f}" '
        f'text-anchor="end">{_escape(frame.min_freq_label)}</text>'
    )


# =============================================================================
# Table renderer (combinations search)
# =============================================================================
#
# A static input strip (the ordered post-discretizer bins) sits at the top; a
# table below fills with consecutive groupings ranked by Tschuprow's T. Each
# data row is a full-width segmented bar — one coloured block per group, spanning
# its members' slots and coloured by its leader (leftmost) bin — plus the T
# value. The selected (winning) row gets a gold background. Own layout constants
# and `_t_*` helpers; zero changes to the existing single/dual renderers.

TABLE_VIEW_H = 396
TABLE_RANK_W = 34  # left gutter for the rank label
TABLE_T_W = 72  # right column for the T value
TABLE_INPUT_H = 30  # height of the input-strip cells
TABLE_INPUT_LABEL_H = 16  # horizontal labels under the input cells
TABLE_HEADER_H = 20  # table column-header row
TABLE_ROW_H = 26  # one candidate row

_T_INPUT_Y0 = _MAIN_Y0  # 88 — input strip starts right after the callout
_T_CELL_X0 = PAD_X + TABLE_RANK_W
_T_CELL_X1 = VIEW_W - PAD_X - TABLE_T_W
_T_CELL_W = _T_CELL_X1 - _T_CELL_X0
_T_TABLE_Y0 = _T_INPUT_Y0 + TABLE_INPUT_H + TABLE_INPUT_LABEL_H + 22
_T_ROWS_Y0 = _T_TABLE_Y0 + TABLE_HEADER_H

GOLD_BG = "#fef3c7"
GOLD_STROKE = "#d97706"


def render_table_svg(table_frames: list, stop_after_stage: int) -> str:
    """Render a list of TableFrames to an animated SVG (combinations table)."""
    total_stages = stop_after_stage
    stage_groups = "\n".join(_t_render_frame(tf, total_stages) for tf in table_frames)
    style = _render_style(stop_after_stage)
    return _doc(VIEW_W, TABLE_VIEW_H, style, stage_groups)


def _t_render_frame(frame, total_stages: int) -> str:
    parts = [
        _stage_caption(frame, total_stages),
        _metric_chip(frame),
        _callout(frame),
        _t_input_strip(frame),
        _t_table(frame),
    ]
    inner = "\n".join(p for p in parts if p)
    return f'  <g class="stage stage-{frame.stage}">\n{inner}\n  </g>'


def _t_input_strip(frame) -> str:
    """The ordered post-discretizer bins as a coloured cell row, with a short
    horizontal label under each cell. Identical across stages so it reads as the
    fixed input the search groups over."""
    bins = frame.input_bins
    if not bins:
        return ""
    y = _T_INPUT_Y0
    parts: list[str] = [
        f'<text class="strip-label" x="{PAD_X:.2f}" y="{y - 4:.2f}" '
        f'text-anchor="start">ordered bins (QuantitativeDiscretizer output)</text>'
    ]
    for b in bins:
        x = _T_CELL_X0 + b.x_start * _T_CELL_W
        w = (b.x_end - b.x_start) * _T_CELL_W
        parts.append(
            f'<rect x="{x + 1:.2f}" y="{y:.2f}" width="{w - 2:.2f}" height="{TABLE_INPUT_H:.2f}" '
            f'rx="2" fill="{PALETTE[b.color_id]}" stroke="#374151" stroke-width="0.6"/>'
        )
        cx = _T_CELL_X0 + ((b.x_start + b.x_end) / 2) * _T_CELL_W
        parts.append(
            f'<text class="binlabel" x="{cx:.2f}" y="{y + TABLE_INPUT_H + 11:.2f}" '
            f'text-anchor="middle">{_escape(b.label)}</text>'
        )
    return "    " + "\n    ".join(parts)


def _t_table(frame) -> str:
    bins = frame.input_bins
    n_bins = len(bins)
    parts: list[str] = []

    # Column headers
    hy = _T_TABLE_Y0 + 13
    parts.append(f'<text class="strip-label" x="{PAD_X:.2f}" y="{hy:.2f}" text-anchor="start">#</text>')
    parts.append(
        f'<text class="strip-label" x="{_T_CELL_X0:.2f}" y="{hy:.2f}" '
        f'text-anchor="start">candidate grouping (adjacent bins sharing a colour are merged)</text>'
    )
    parts.append(
        f'<text class="strip-label" x="{VIEW_W - PAD_X:.2f}" y="{hy:.2f}" text-anchor="end">Tschuprow\'s T</text>'
    )
    if frame.top_k:
        parts.append(
            f'<text class="tick-label" x="{_T_CELL_X1:.2f}" y="{_T_TABLE_Y0 - 4:.2f}" '
            f'text-anchor="end">top_k = {frame.top_k} / {frame.n_total}</text>'
        )

    # Data rows
    bar_h = TABLE_ROW_H - 12
    for i, row in enumerate(frame.rows):
        ry = _T_ROWS_Y0 + i * TABLE_ROW_H
        cy = ry + TABLE_ROW_H / 2
        if row.is_winner:
            parts.append(
                f'<rect x="{PAD_X - 4:.2f}" y="{ry + 1:.2f}" width="{VIEW_W - 2 * PAD_X + 8:.2f}" '
                f'height="{TABLE_ROW_H - 2:.2f}" rx="4" fill="{GOLD_BG}" '
                f'stroke="{GOLD_STROKE}" stroke-width="1"/>'
            )
        # rank
        prefix = "&#9733; " if row.is_winner else ""  # ★ on the winner
        parts.append(
            f'<text class="binlabel" x="{PAD_X:.2f}" y="{cy + 3.5:.2f}" '
            f'text-anchor="start">{prefix}#{row.rank + 1}</text>'
        )
        # segmented bar — one block per group
        bar_y = ry + (TABLE_ROW_H - bar_h) / 2
        for group in row.groups:
            lo = min(group)
            hi = max(group)
            leader = group[0]
            gx0 = _T_CELL_X0 + (lo / n_bins) * _T_CELL_W
            gx1 = _T_CELL_X0 + ((hi + 1) / n_bins) * _T_CELL_W
            parts.append(
                f'<rect x="{gx0 + 1:.2f}" y="{bar_y:.2f}" width="{gx1 - gx0 - 2:.2f}" '
                f'height="{bar_h:.2f}" rx="2" fill="{PALETTE[bins[leader].color_id]}" '
                f'stroke="#374151" stroke-width="0.6"/>'
            )
        # T value
        parts.append(
            f'<text class="tick-label" x="{VIEW_W - PAD_X:.2f}" y="{cy + 3.5:.2f}" '
            f'text-anchor="end">{row.tschuprowt:.3f}</text>'
        )
    return "    " + "\n    ".join(parts)


# =============================================================================
# Hero renderer (README full-pipeline animation)
# =============================================================================
#
# The feature strip (Discretizers' idiom) at the regular single-strip
# coordinates, with the ranked-groupings table (Carvers' idiom) below it. The
# strip reuses the single-strip helpers as-is on `frame.strip`; the table rows
# span the strip's value-axis x-spans so each candidate row lines up vertically
# under the bins it groups. Own `_h_*` helpers; zero changes to the existing
# single/dual/table renderers.

_H_SEP_Y = _MAIN_BASELINE + 72  # below the rotated bin labels
_H_CAPTION_Y = _H_SEP_Y + 14  # table caption + top-k badge line
_H_ROWS_Y0 = _H_SEP_Y + 22  # first candidate row
HERO_VIEW_H = int(_H_ROWS_Y0 + 5 * TABLE_ROW_H + 10)


def render_hero_svg(hero_frames: list, stop_after_stage: int, dark: bool = False) -> str:
    """Render a list of HeroFrames to an animated SVG (strip + table hybrid)."""
    theme = _DARK_THEME if dark else _HERO_LIGHT_THEME
    total_stages = stop_after_stage
    stage_groups = "\n".join(_h_render_frame(hf, total_stages, theme) for hf in hero_frames)
    style = _render_style(stop_after_stage, theme)
    return _doc(VIEW_W, HERO_VIEW_H, style, stage_groups, theme)


def _h_render_frame(frame, total_stages: int, theme: dict = _LIGHT_THEME) -> str:
    strip = frame.strip
    parts = [
        _stage_caption(frame, total_stages),
        _metric_chip(frame, theme),
        _callout(frame),
        _main_display(strip, theme),
        _target_dots(strip, theme),
        _nan_strip(strip),
        _baseline_line(theme),
        _min_freq_line(strip, theme),
        _h_winner_cuts(frame, theme),
        _h_table(frame, theme),
    ]
    inner = "\n".join(p for p in parts if p)
    return f'  <g class="stage stage-{frame.stage}">\n{inner}\n  </g>'


def _h_winner_cuts(frame, theme: dict = _LIGHT_THEME) -> str:
    """Gold dashed verticals over the bar zone at the winning grouping's
    interior cut points."""
    if not frame.winner_cuts:
        return ""
    bar_zone_h = _bar_zone_h(frame.strip)
    y_top = _MAIN_BASELINE - bar_zone_h
    parts: list[str] = []
    for cut in frame.winner_cuts:
        x = PAD_X + cut * BIN_W
        parts.append(
            f'<line x1="{x:.2f}" y1="{y_top - 4:.2f}" x2="{x:.2f}" '
            f'y2="{_MAIN_BASELINE:.2f}" stroke="{theme["gold_stroke"]}" stroke-width="2" '
            f'stroke-dasharray="5 3"/>'
        )
    return "    " + "\n    ".join(parts)


def _h_table(frame, theme: dict = _LIGHT_THEME) -> str:
    """Candidate rows below the strip. Each row is a segmented bar whose blocks
    span the grouped bins' value-axis x-spans (so they align under the strip);
    rank + T sit in the right gutter under the NaN strip. Winner row gold."""
    bins = frame.strip.bins
    parts: list[str] = [
        f'<line x1="{PAD_X}" y1="{_H_SEP_Y:.2f}" '
        f'x2="{PAD_X + BIN_W + 10 + NAN_W:.2f}" y2="{_H_SEP_Y:.2f}" '
        f'stroke="{theme["sep_stroke"]}" stroke-width="1"/>',
        f'<text class="strip-label" x="{PAD_X:.2f}" y="{_H_CAPTION_Y:.2f}" '
        f'text-anchor="start">Carver — consecutive groupings ranked by Tschuprow\'s T (best first)</text>',
    ]
    if frame.top_k:
        parts.append(
            f'<text class="tick-label" x="{VIEW_W - PAD_X:.2f}" y="{_H_CAPTION_Y:.2f}" '
            f'text-anchor="end">top {frame.top_k} of {frame.n_total}</text>'
        )
    bar_h = TABLE_ROW_H - 12
    for i, row in enumerate(frame.rows):
        ry = _H_ROWS_Y0 + i * TABLE_ROW_H
        cy = ry + TABLE_ROW_H / 2
        if row.is_winner:
            parts.append(
                f'<rect x="{PAD_X - 4:.2f}" y="{ry + 1:.2f}" width="{VIEW_W - 2 * PAD_X + 8:.2f}" '
                f'height="{TABLE_ROW_H - 2:.2f}" rx="4" fill="{theme["gold_bg"]}" '
                f'stroke="{theme["gold_stroke"]}" stroke-width="1"/>'
            )
        bar_y = ry + (TABLE_ROW_H - bar_h) / 2
        for group in row.groups:
            leader = max(group, key=lambda j: bins[j].freq)
            gx0 = PAD_X + bins[min(group)].x_start * BIN_W
            gx1 = PAD_X + bins[max(group)].x_end * BIN_W
            parts.append(
                f'<rect x="{gx0 + 1:.2f}" y="{bar_y:.2f}" width="{gx1 - gx0 - 2:.2f}" '
                f'height="{bar_h:.2f}" rx="2" fill="{PALETTE[bins[leader].color_id]}" '
                f'stroke="{theme["bar_stroke"]}" stroke-width="0.6"/>'
            )
        prefix = "&#9733; " if row.is_winner else ""  # ★ on the winner
        parts.append(
            f'<text class="tick-label" x="{VIEW_W - PAD_X:.2f}" y="{cy + 3.5:.2f}" '
            f'text-anchor="end">{prefix}#{row.rank + 1} &#183; {row.tschuprowt:.3f}</text>'
        )
    return "    " + "\n    ".join(parts)


_DOC = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {view_w} {view_h}"
     width="{view_w}" height="{view_h}" role="img"
     aria-label="AutoCarver discretizer animation">
  <defs>
    <pattern id="nan-hatch" patternUnits="userSpaceOnUse" width="6" height="6"
             patternTransform="rotate(45)">
      <rect width="6" height="6" fill="{nan_hatch_bg}"/>
      <line x1="0" y1="0" x2="0" y2="6" stroke="{nan_hatch_stroke}" stroke-width="1.2"/>
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


def _doc(view_w: int, view_h: int, style: str, stage_groups: str, theme: dict = _LIGHT_THEME) -> str:
    return _DOC.format(
        view_w=view_w,
        view_h=view_h,
        style=style,
        stage_groups=stage_groups,
        nan_hatch_bg=theme["nan_hatch_bg"],
        nan_hatch_stroke=theme["nan_hatch_stroke"],
    )
