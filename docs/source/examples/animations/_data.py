"""Frame data for the ContinuousDiscretizer animation.

Three stages, all values computed from the data and the real discretizer:

  0. Raw continuous distribution — scipy KDE of synthetic Fare (+ NaN aside)
  1. Over-rep value detected — same curve + vertical orange line at Fare = 0
  2. After ContinuousDiscretizer — modalities and labels from
     `ContinuousDiscretizer.fit_transform()`, plotted as equal-width bars whose
     heights are their actual relative frequencies.

A horizontal `min_freq` reference line is drawn on every stage at the same
y-coordinate (relative to the Stage 2 max bar height) so the threshold is
comparable across the morph.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from AutoCarver.combinations.binary.binary_combination_evaluators import _top_k_partitions_chi2_dp
from AutoCarver.discretizers import (
    CategoricalDiscretizer,
    ContinuousDiscretizer,
    OrdinalDiscretizer,
    QuantitativeDiscretizer,
)
from AutoCarver.discretizers.utils.frequency_ci import is_significantly_below
from AutoCarver.features import Features

from ._engine import Bin, ComboRow, DualFrame, Frame, HeroFrame, MergeArrow, TableFrame

# --- Synthesis parameters ----------------------------------------------------
SEED = 7
N_ROWS = 1500
FARE_MAX = 35.0  # any synthetic value above this is routed to NaN
P_NAN = 0.03
P_ZERO_AMONG_NONNAN = 0.10
LOGNORMAL_MU = 2.0
LOGNORMAL_SIGMA = 0.7
MIN_FREQ = 0.07
KDE_BANDWIDTH = 0.04
SINGLETON_STRIP_PCT = 0.01  # fixed 1% of strip width for over-rep singletons
TICK_STEP = 5  # x-axis tick spacing in fare units (Stage 0/1)

QUANTILE_PALETTE = (1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6)

# Logistic mapping fare → P(survived). Plausible Titanic-flavoured story:
# higher fare → higher survival. Used only to give the Stage-2 dots a
# meaningful spread (ContinuousDiscretizer itself doesn't see y).
CONT_LOGISTIC_SLOPE = 0.15
CONT_LOGISTIC_MID = 12.0


def _zoom_target_range(rates: list[float]) -> tuple[float, float]:
    """Padded (min, max) range used to scale target-rate dots.

    Anchoring the strip at 0 collapses dot differences when all rates cluster
    in a narrow band. Padding by ~15 % of the observed span keeps the dots off
    the strip edges while preserving differences. The minimum total span of
    0.05 stops the strip from degenerating when every modality has near-equal
    target rate.
    """
    if not rates:
        return 0.0, 1.0
    lo, hi = min(rates), max(rates)
    span = hi - lo
    pad = max(span * 0.15, 0.025)
    return max(0.0, lo - pad), min(1.0, hi + pad)


def continuous_binary_frames() -> list[Frame]:
    fare, y = _synthesize()
    nan_mask = np.isnan(fare)
    n = len(fare)
    nonan_fare = fare[~nan_mask]
    nan_freq = float(nan_mask.sum()) / n

    # ----- Density curve (Stages 0 & 1) -------------------------------------
    kde = gaussian_kde(nonan_fare, bw_method=KDE_BANDWIDTH)
    x_samples = np.linspace(0.0, FARE_MAX, 160)
    density = kde(x_samples)
    density = density / density.max()
    density_curve = tuple((float(x) / FARE_MAX, float(y)) for x, y in zip(x_samples, density))

    overrep_markers = ((0.0, "Fare = 0 spike → own modality"),)

    # ----- Stage 2: real ContinuousDiscretizer.fit_transform() --------------
    df = pd.DataFrame({"Fare": fare})
    features = Features(numericals=["Fare"])
    cd = ContinuousDiscretizer(quantitatives=features.quantitatives, min_freq=MIN_FREQ)
    transformed = cd.fit_transform(df)
    feature = features.quantitatives[0]
    counts = transformed["Fare"].value_counts(dropna=False)

    # Real per-bin P(y=1), computed from the transformed labels — drives the
    # Stage-2 target-rate dots. CD itself doesn't see y; this is purely a
    # readability hint that the carver downstream will exploit these bins.
    target_rate_by_label = (
        pd.DataFrame({"Fare": transformed["Fare"], "y": y.to_numpy()}).groupby("Fare", observed=True)["y"].mean()
    )

    s2_bins = _bins_on_value_axis(feature, counts, target_rate_by_label, n)
    # Zoom on the Stage-2 per-bin rates. Reserving the strip on every stage
    # (not just Stage 2) keeps the min_freq line at a consistent y across the
    # morph — `_target_dots` already short-circuits on bin-less frames, so
    # Stages 0/1 stay visually clean (no dots, no caption).
    target_strip_min, target_strip_max = _zoom_target_range([b.target for b in s2_bins])

    # Equally-spaced x-axis ticks shown under the density curve in Stage 0/1.
    xaxis_ticks = tuple((v / FARE_MAX, str(int(v))) for v in range(0, int(FARE_MAX) + 1, TICK_STEP))

    # Scale used by the bar heights AND the min_freq reference line, so the
    # line sits at the same y across every stage of the morph.
    bar_max_freq = max(b.freq for b in s2_bins)
    min_freq_y_norm = MIN_FREQ / bar_max_freq
    min_freq_label = f"min_freq = {MIN_FREQ:.2f}"

    nan_bin_0 = Bin(
        label="NaN",
        x_start=0.0,
        x_end=1.0,
        freq=nan_freq,
        target=0.0,
        color_id=7,
        is_nan=True,
    )

    q = int(round(1.0 / MIN_FREQ))

    common = {
        "min_freq_y_norm": min_freq_y_norm,
        "min_freq_label": min_freq_label,
        "bar_max_freq": bar_max_freq,
        "target_strip_max": target_strip_max,
        "target_strip_min": target_strip_min,
    }

    return [
        Frame(
            0,
            "Raw feature",
            bins=(),
            nan_bin=nan_bin_0,
            metric="—",
            callout=(
                f"Raw continuous distribution of Fare (synthetic, n={N_ROWS}). "
                "Density estimated by Gaussian KDE; NaN held aside."
            ),
            density_curve=density_curve,
            tick_values=xaxis_ticks,
            **common,
        ),
        Frame(
            1,
            "Over-represented value detected",
            bins=(),
            nan_bin=nan_bin_0,
            metric="—",
            callout=(
                f"ContinuousDiscretizer flags values occurring ≥ 1/q ({1 / q:.2f}) as "
                "their own modality. Fare = 0 is the over-rep spike (orange)."
            ),
            density_curve=density_curve,
            overrep_markers=overrep_markers,
            tick_values=xaxis_ticks,
            **common,
        ),
        Frame(
            2,
            "After ContinuousDiscretizer",
            bins=s2_bins,
            nan_bin=nan_bin_0,
            metric="—",
            callout=(
                f"{len(s2_bins)} modalities from `ContinuousDiscretizer.fit_transform()`. "
                "Bar widths = real value ranges; heights = real frequencies."
            ),
            **common,
        ),
    ]


# --- Synthesis ---------------------------------------------------------------


def _synthesize() -> tuple[np.ndarray, pd.Series]:
    rng = np.random.default_rng(SEED)
    nan_mask = rng.random(N_ROWS) < P_NAN
    n_nonnan = int((~nan_mask).sum())
    base = rng.lognormal(LOGNORMAL_MU, LOGNORMAL_SIGMA, size=n_nonnan)
    is_zero = rng.random(n_nonnan) < P_ZERO_AMONG_NONNAN
    values = np.where(is_zero, 0.0, base)
    # Anything above the display range joins NaN — same effect as truncating
    # the distribution at FARE_MAX and pushing the tail into the NaN bucket.
    values[values > FARE_MAX] = np.nan
    fare = np.full(N_ROWS, np.nan)
    fare[~nan_mask] = values

    # Plausible Titanic-flavoured target: higher fare → higher P(survived).
    # NaN rows get the logistic centre as a baseline so they survive at ~50 %.
    fare_for_p = np.where(np.isnan(fare), CONT_LOGISTIC_MID, fare)
    p = 1.0 / (1.0 + np.exp(-CONT_LOGISTIC_SLOPE * (fare_for_p - CONT_LOGISTIC_MID)))
    y = (rng.random(N_ROWS) < p).astype(int)
    return fare, pd.Series(y, name="Survived")


# --- Build bins from the fitted feature --------------------------------------


def _bins_from_fitted_feature(
    feature,
    counts: pd.Series,
    n_total: int,
) -> tuple[Bin, ...]:
    """Equal-width bars, one per CD modality. Label = compact form of
    `feature.labels[i]`. Frequency = real value_counts on the transformed col."""
    labels = list(feature.labels)
    n_bins = len(labels)
    bins: list[Bin] = []
    for i, raw_label in enumerate(labels):
        freq = float(counts.get(raw_label, 0)) / n_total
        bins.append(
            Bin(
                label=_clean_label(raw_label),
                x_start=i / n_bins,
                x_end=(i + 1) / n_bins,
                freq=freq,
                target=0.0,
                color_id=QUANTILE_PALETTE[i % len(QUANTILE_PALETTE)],
            )
        )
    return tuple(bins)


def _bins_on_value_axis(
    feature,
    counts: pd.Series,
    target_rates: pd.Series,
    n_total: int,
) -> tuple[Bin, ...]:
    """Bars on the value axis [0, FARE_MAX] — widths reflect each modality's
    real value range. Singleton (over-rep) modalities get a fixed strip-width
    percentage (`SINGLETON_STRIP_PCT`) regardless of FARE_MAX, so they always
    look slim. The next bin's x_start is shifted to avoid overlap.
    `target_rates` is indexed by the same raw modality labels as `counts`.
    """
    labels = list(feature.labels)
    boundaries = [float(v) for v in feature.values]
    # Singleton width expressed back in fare units so it composes with the
    # cursor-on-value-axis bookkeeping below.
    singleton_w_fare = SINGLETON_STRIP_PCT * FARE_MAX
    bins: list[Bin] = []
    x_cursor = 0.0  # fare units; ensures bars never overlap
    for i, raw_label in enumerate(labels):
        upper = boundaries[i]
        lower = boundaries[i - 1] if i > 0 else float("-inf")
        true_hi = FARE_MAX if np.isinf(upper) else min(upper, FARE_MAX)
        true_lo = 0.0 if (np.isinf(lower) or lower < 0) else lower
        is_singleton = true_hi - true_lo < 0.5
        if is_singleton:
            lo = max(x_cursor, true_lo)
            hi = lo + singleton_w_fare
        else:
            lo = max(x_cursor, true_lo)
            hi = true_hi
            if hi < lo:  # singleton ate this bin's space; give it a tiny slot
                hi = lo + singleton_w_fare / 2
        x_cursor = hi
        freq = float(counts.get(raw_label, 0)) / n_total
        target = float(target_rates.get(raw_label, 0.0))
        bins.append(
            Bin(
                label=_clean_label(raw_label),
                x_start=lo / FARE_MAX,
                x_end=hi / FARE_MAX,
                freq=freq,
                target=target,
                color_id=QUANTILE_PALETTE[i % len(QUANTILE_PALETTE)],
            )
        )
    return tuple(bins)


# --- Label cleanup ----------------------------------------------------------

_SCI_NUM = re.compile(r"-?\d+\.\d+e[+-]?\d+")


def _clean_label(label: str) -> str:
    """Convert CD's scientific-notation interval labels into compact form.

    Examples
    --------
    >>> _clean_label('(-inf, 0.00e+00]')
    '≤ 0'
    >>> _clean_label('(0.00e+00, 2.89e+00]')
    '(0, 2.89]'
    >>> _clean_label('(1.98e+01, inf)')
    '> 19.8'
    """
    label = _SCI_NUM.sub(lambda m: _fmt(float(m.group(0))), label)
    if m := re.fullmatch(r"\(-inf, (\S+)\]", label):
        return f"≤ {m.group(1)}"
    if m := re.fullmatch(r"\((\S+), inf\)", label):
        return f"> {m.group(1)}"
    if m := re.fullmatch(r"\((\S+), (\S+)\]", label):
        return f"({m.group(1)}, {m.group(2)}]"
    return label


def _fmt(val: float) -> str:
    if val == int(val):
        return str(int(val))
    return f"{val:.2f}".rstrip("0").rstrip(".")


# =============================================================================
# CategoricalDiscretizer animation
# =============================================================================

CAT_SEED = 7
CAT_N_ROWS = 1500
CAT_MIN_FREQ = 0.07
CAT_P_NAN = 0.18

# (category, raw_weight, target_rate). Weights are relative; they are renormalized
# to (1 - CAT_P_NAN). Belfast and Boston are below CAT_MIN_FREQ → grouped.
CAT_SPEC = (
    ("Southampton", 0.45, 0.30),
    ("Cherbourg", 0.18, 0.55),
    ("Queenstown", 0.12, 0.40),
    ("Belfast", 0.04, 0.70),
    ("Boston", 0.03, 0.50),
)
CAT_NAN_TARGET_RATE = 0.20  # ignored in animation, used only to make `y` realistic

# Each modality gets a stable color across stages so the eye can track reorder.
CAT_COLOR_BY_LABEL = {
    "Southampton": 1,
    "Cherbourg": 2,
    "Queenstown": 3,
    "Belfast": 4,
    "Boston": 5,
}
OTHER_COLOR = 0  # __OTHER__ default bin uses the neutral grey

# Approx max bar height as a fraction of the bar zone — keeps the bars from
# touching the target-rate strip.
CAT_BAR_HEAD_ROOM = 0.92


def categorical_binary_frames() -> list[Frame]:
    df, y = _synthesize_categorical()
    n = len(df)

    # ---- Real fit_transform() so the final state matches the actual library --
    features = Features(categoricals=["Port"])
    cd = CategoricalDiscretizer(
        categoricals=features.categoricals,
        min_freq=CAT_MIN_FREQ,
    )
    transformed = cd.fit_transform(df, y)
    feature = features.categoricals[0]
    final_labels = list(feature.labels)  # already sorted by target rate
    final_values = list(feature.values)  # parallel: includes "__OTHER__"
    final_counts = transformed["Port"].value_counts(dropna=False)

    # ---- Stage 0: raw modalities, in frequency-desc order -------------------
    raw_counts = df["Port"].value_counts(dropna=False)
    nan_freq = float(df["Port"].isna().sum()) / n

    raw_rates = _target_rate_per_modality(df["Port"], y)

    raw_modalities = [m for m in raw_counts.index if not pd.isna(m)]
    raw_modalities.sort(key=lambda m: -int(raw_counts[m]))
    rare_modalities = [m for m, _, _ in CAT_SPEC if int(raw_counts.get(m, 0)) / n < CAT_MIN_FREQ]

    bar_max_freq = max(int(raw_counts[m]) / n for m in raw_modalities) / CAT_BAR_HEAD_ROOM

    s0_bins = _equal_slot_bins(
        labels=raw_modalities,
        freqs=[int(raw_counts[m]) / n for m in raw_modalities],
        targets=[raw_rates[m] for m in raw_modalities],
        color_ids=[CAT_COLOR_BY_LABEL[m] for m in raw_modalities],
    )
    highlight_s0 = tuple(i for i, b in enumerate(s0_bins) if b.label in rare_modalities)

    # ---- Stage 1: after rare-modality merge (target order untouched) --------
    # Same x-positions for non-rare bars; rare ones replaced by a single
    # __OTHER__ bar at the last slot. Width grows so it covers the union of
    # the rare slots — visually communicates "these merged".
    kept = [m for m in raw_modalities if m not in rare_modalities]
    rare_count = sum(int(raw_counts[m]) for m in rare_modalities)
    rare_target_rate = (
        sum(int(raw_counts[m]) * raw_rates[m] for m in rare_modalities) / rare_count if rare_count else 0.0
    )
    other_label_s1 = ", ".join(rare_modalities)  # matches feature.labels style
    s1_bins = _stage1_bins(
        kept_labels=kept,
        kept_freqs=[int(raw_counts[m]) / n for m in kept],
        kept_targets=[raw_rates[m] for m in kept],
        kept_color_ids=[CAT_COLOR_BY_LABEL[m] for m in kept],
        other_label=other_label_s1,
        other_freq=rare_count / n,
        other_target=rare_target_rate,
        n_total_slots=len(raw_modalities),
    )

    # ---- Stage 2: after target sort (real labels from fit_transform) --------
    # Map each post-fit label to its target rate: kept modalities use their raw
    # rate, __OTHER__ uses the weighted average computed for stage 1.
    label_to_target = {m: raw_rates[m] for m in kept}
    label_to_target[other_label_s1] = rare_target_rate
    s2_bins = _stage2_bins(
        labels=final_labels,
        values=final_values,
        counts=final_counts,
        n_total=n,
        label_to_target=label_to_target,
    )

    nan_bin = Bin(
        label="NaN",
        x_start=0.0,
        x_end=1.0,
        freq=nan_freq,
        target=0.0,
        color_id=7,
        is_nan=True,
    )

    min_freq_y_norm = CAT_MIN_FREQ / bar_max_freq
    min_freq_label = f"min_freq = {CAT_MIN_FREQ:.2f}"
    # Zoom the target-rate strip to the actual span of rates that appear in
    # any stage, so neighbouring modalities don't visually collapse onto each
    # other when the spread is narrow.
    target_strip_min, target_strip_max = _zoom_target_range([b.target for b in s0_bins + s1_bins + s2_bins])
    common = {
        "min_freq_y_norm": min_freq_y_norm,
        "min_freq_label": min_freq_label,
        "bar_max_freq": bar_max_freq,
        "target_strip_max": target_strip_max,
        "target_strip_min": target_strip_min,
    }

    return [
        Frame(
            0,
            "Raw feature",
            bins=s0_bins,
            nan_bin=nan_bin,
            metric="—",
            callout=(
                f"Raw categorical distribution of Port (synthetic, n={CAT_N_ROWS}). "
                "Dots above bars = P(y=1) per modality."
            ),
            highlight_bins=highlight_s0,
            **common,
        ),
        Frame(
            1,
            "Rare modalities grouped",
            bins=s1_bins,
            nan_bin=nan_bin,
            metric="—",
            callout=(
                "Modalities significantly under min_freq (Wilson CI) collapse "
                f"into the default bin '__OTHER__' = '{other_label_s1}'."
            ),
            **common,
        ),
        Frame(
            2,
            "After CategoricalDiscretizer",
            bins=s2_bins,
            nan_bin=nan_bin,
            metric="—",
            callout=(f"{len(s2_bins)} modalities, reordered by ascending P(y=1) — the dot trace is now monotonic."),
            **common,
        ),
    ]


# --- Categorical synthesis ---------------------------------------------------


def _synthesize_categorical() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(CAT_SEED)
    labels = [s[0] for s in CAT_SPEC]
    weights = np.array([s[1] for s in CAT_SPEC], dtype=float)
    weights = weights / weights.sum() * (1 - CAT_P_NAN)
    probs = np.append(weights, 1 - weights.sum())
    choices = rng.choice(labels + [None], size=CAT_N_ROWS, p=probs)
    rates = {s[0]: s[2] for s in CAT_SPEC}
    rates[None] = CAT_NAN_TARGET_RATE
    y = np.array([rng.random() < rates[v] for v in choices]).astype(int)
    return pd.DataFrame({"Port": choices}), pd.Series(y, name="Survived")


def _target_rate_per_modality(x: pd.Series, y: pd.Series) -> dict:
    rates = {}
    for m in x.dropna().unique():
        mask = x == m
        rates[m] = float(y[mask].mean())
    return rates


# --- Bin builders ------------------------------------------------------------


def _equal_slot_bins(
    labels: list,
    freqs: list[float],
    targets: list[float],
    color_ids: list[int],
) -> tuple[Bin, ...]:
    n = len(labels)
    return tuple(
        Bin(
            label=str(labels[i]),
            x_start=i / n,
            x_end=(i + 1) / n,
            freq=freqs[i],
            target=targets[i],
            color_id=color_ids[i],
        )
        for i in range(n)
    )


def _stage1_bins(
    kept_labels: list,
    kept_freqs: list[float],
    kept_targets: list[float],
    kept_color_ids: list[int],
    other_label: str,
    other_freq: float,
    other_target: float,
    n_total_slots: int,
) -> tuple[Bin, ...]:
    """Stage-1 layout: kept bars keep their stage-0 slots; the trailing slot(s)
    previously occupied by rare modalities are merged into a single wider
    __OTHER__ bar."""
    bins: list[Bin] = []
    for i, (lbl, f, t, c) in enumerate(zip(kept_labels, kept_freqs, kept_targets, kept_color_ids)):
        bins.append(
            Bin(
                label=str(lbl),
                x_start=i / n_total_slots,
                x_end=(i + 1) / n_total_slots,
                freq=f,
                target=t,
                color_id=c,
            )
        )
    other_start = len(kept_labels) / n_total_slots
    bins.append(
        Bin(
            label=f"__OTHER__\n({other_label})",
            x_start=other_start,
            x_end=1.0,
            freq=other_freq,
            target=other_target,
            color_id=OTHER_COLOR,
        )
    )
    return tuple(bins)


def _stage2_bins(
    labels: list[str],
    values: list[str],
    counts: pd.Series,
    n_total: int,
    label_to_target: dict[str, float],
) -> tuple[Bin, ...]:
    """Stage-2 layout: equal-width slots in target-rate-sorted order. Colors
    travel with each modality (so reorder reads as movement, not recolor)."""
    n = len(labels)
    bins: list[Bin] = []
    for i, (lbl, val) in enumerate(zip(labels, values)):
        freq = float(counts.get(lbl, 0)) / n_total
        if val == "__OTHER__":
            color = OTHER_COLOR
            display = f"__OTHER__\n({lbl})"
        else:
            color = CAT_COLOR_BY_LABEL.get(lbl, OTHER_COLOR)
            display = str(lbl)
        bins.append(
            Bin(
                label=display,
                x_start=i / n,
                x_end=(i + 1) / n,
                freq=freq,
                target=label_to_target.get(lbl, 0.0),
                color_id=color,
            )
        )
    return tuple(bins)


# =============================================================================
# OrdinalDiscretizer animation
# =============================================================================

ORD_SEED = 7
ORD_N_ROWS = 1500
ORD_MIN_FREQ = 0.07
ORD_MIN_FREQ_ALPHA = 0.05
ORD_P_NAN = 0.06

# Synthetic Titanic-flavoured `AgeGroup` ordinal feature. Declared order is the
# domain-meaningful chronological ranking (NOT target-rate ranking). `teen` and
# `elderly` sit below `min_freq`; the real `OrdinalDiscretizer.fit_transform`
# picks the merge directions from the sample's target rates.
ORD_ORDER = ("child", "teen", "adult", "middle_aged", "elderly")
# Target rates picked so the Stage-2 *merged* groups land at clearly separated
# rates. Because rare modalities get absorbed by much larger neighbours, the
# merged-group rate is dominated by the kept (large) bin — so the kept bins'
# rates (adult, middle_aged) must themselves differ to keep the dots readable
# after merging.
ORD_SPEC = (
    ("child", 0.12, 0.20),
    ("teen", 0.04, 0.75),
    ("adult", 0.42, 0.40),
    ("middle_aged", 0.33, 0.60),
    ("elderly", 0.03, 0.85),
)
ORD_NAN_TARGET_RATE = 0.50

# Each raw modality gets a stable colour across stages so the eye can track it
# as it gets absorbed into its merge target.
ORD_COLOR_BY_LABEL = {
    "child": 1,
    "teen": 2,
    "adult": 3,
    "middle_aged": 4,
    "elderly": 5,
}

ORD_BAR_HEAD_ROOM = 0.92


def ordinal_binary_frames() -> list[Frame]:
    df, y = _synthesize_ordinal()
    n = len(df)

    # ---- Real fit_transform() so the final state matches the actual library --
    features = Features(ordinals={"AgeGroup": list(ORD_ORDER)})
    od = OrdinalDiscretizer(ordinals=features.ordinals, min_freq=ORD_MIN_FREQ)
    od.fit_transform(df, y)
    feature = features.ordinals[0]
    # feature.content maps each kept modality to the list of raw modalities it
    # absorbed (e.g. {'adult': ['teen', 'adult'], ...}). Source of truth for
    # the merge arrows + Stage 2 bar spans.
    content: dict[str, list[str]] = dict(feature.content)

    # ---- Stage 0: raw bars in DECLARED ORDINAL ORDER ------------------------
    raw_counts = df["AgeGroup"].value_counts(dropna=False)
    nan_freq = float(df["AgeGroup"].isna().sum()) / n
    raw_rates = _target_rate_per_modality(df["AgeGroup"], y)

    rare_modalities = [
        m
        for m in ORD_ORDER
        if is_significantly_below(
            int(raw_counts.get(m, 0)),
            n,
            ORD_MIN_FREQ,
            ORD_MIN_FREQ_ALPHA,
        )
    ]

    bar_max_freq = max(int(raw_counts.get(m, 0)) / n for m in ORD_ORDER) / ORD_BAR_HEAD_ROOM

    s0_bins = _equal_slot_bins(
        labels=list(ORD_ORDER),
        freqs=[int(raw_counts.get(m, 0)) / n for m in ORD_ORDER],
        targets=[raw_rates[m] for m in ORD_ORDER],
        color_ids=[ORD_COLOR_BY_LABEL[m] for m in ORD_ORDER],
    )
    highlight_s0 = tuple(i for i, b in enumerate(s0_bins) if b.label in rare_modalities)

    # ---- Stage 1: same bars + merge-direction arrows ------------------------
    arrows = _ordinal_merge_arrows(content, ORD_ORDER, s0_bins)

    # ---- Stage 2: merged bars spanning absorbed slots -----------------------
    s2_bins = _ordinal_stage2_bins(
        feature_labels=list(feature.labels),
        feature_values=list(feature.values),
        content=content,
        order=ORD_ORDER,
        raw_counts=raw_counts,
        raw_rates=raw_rates,
        n_total=n,
    )

    nan_bin = Bin(
        label="NaN",
        x_start=0.0,
        x_end=1.0,
        freq=nan_freq,
        target=0.0,
        color_id=7,
        is_nan=True,
    )

    min_freq_y_norm = ORD_MIN_FREQ / bar_max_freq
    min_freq_label = f"min_freq = {ORD_MIN_FREQ:.2f}"
    # Stage 2's merged bars carry weighted-average target rates; collect every
    # rate that the animation ever paints so the strip span stays consistent
    # across stages and stays zoomed on real differences.
    target_strip_min, target_strip_max = _zoom_target_range([b.target for b in s0_bins + s2_bins])
    common = {
        "min_freq_y_norm": min_freq_y_norm,
        "min_freq_label": min_freq_label,
        "bar_max_freq": bar_max_freq,
        "target_strip_max": target_strip_max,
        "target_strip_min": target_strip_min,
    }

    return [
        Frame(
            0,
            "Raw feature",
            bins=s0_bins,
            nan_bin=nan_bin,
            metric="—",
            callout=(
                f"Raw ordinal AgeGroup (synthetic, n={ORD_N_ROWS}). Bars in "
                "declared order; dots = P(y=1). Rare modalities outlined orange."
            ),
            highlight_bins=highlight_s0,
            **common,
        ),
        Frame(
            1,
            "Merge direction chosen",
            bins=s0_bins,
            nan_bin=nan_bin,
            metric="—",
            callout=(
                "Each rare modality merges with the adjacent neighbour whose "
                "target rate is closest (or its only neighbour at the edges)."
            ),
            highlight_bins=highlight_s0,
            merge_arrows=arrows,
            **common,
        ),
        Frame(
            2,
            "After OrdinalDiscretizer",
            bins=s2_bins,
            nan_bin=nan_bin,
            metric="—",
            callout=(
                f"{len(s2_bins)} groups, ordinal order preserved. Each merged "
                "bar spans the slots of its absorbed modalities."
            ),
            **common,
        ),
    ]


def _synthesize_ordinal() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(ORD_SEED)
    labels = [s[0] for s in ORD_SPEC]
    weights = np.array([s[1] for s in ORD_SPEC], dtype=float)
    weights = weights / weights.sum() * (1 - ORD_P_NAN)
    probs = np.append(weights, 1 - weights.sum())
    choices = rng.choice(labels + [None], size=ORD_N_ROWS, p=probs)
    rates = {s[0]: s[2] for s in ORD_SPEC}
    rates[None] = ORD_NAN_TARGET_RATE
    y = np.array([rng.random() < rates[v] for v in choices]).astype(int)
    return pd.DataFrame({"AgeGroup": choices}), pd.Series(y, name="Survived")


def _ordinal_merge_arrows(
    content: dict[str, list[str]],
    order: tuple[str, ...],
    s0_bins: tuple[Bin, ...],
) -> tuple[MergeArrow, ...]:
    """One arrow per absorbed→kept pair. `from_x`/`to_x` are bar-centre
    coordinates (normalized to [0, 1] along the bin strip)."""
    centre = {b.label: (b.x_start + b.x_end) / 2 for b in s0_bins}
    arrows: list[MergeArrow] = []
    for kept, raw_members in content.items():
        for raw in raw_members:
            if raw == kept:
                continue
            arrows.append(
                MergeArrow(
                    from_x=centre[raw],
                    to_x=centre[kept],
                    label=f"{raw} → {kept}",
                )
            )
    return tuple(arrows)


# =============================================================================
# QuantitativeDiscretizer animation
# =============================================================================
#
# Composite pipeline: ContinuousDiscretizer + OrdinalDiscretizer's merge. Five
# stages, building on the CD prefix (Stages 0/1/2) and adding the OD-style
# merge (Stages 3/4). The CD example uses a single zero spike; that scenario's
# post-CD bins all clear Wilson CI, so the OD merge is a no-op. We need
# multiple over-rep "class fares" to force some quantile bins below threshold
# — that's what QD's pipeline is built for.

QD_SEED = 7
QD_N_ROWS = 1500
QD_FARE_MAX = 35.0
QD_MIN_FREQ = 0.07
QD_MIN_FREQ_ALPHA = 0.05
QD_P_NAN = 0.03
# Probability that a row's fare is a "class fare" spike (vs lognormal body).
# `p_spike=0.78` is high on purpose: it shrinks each between-spike segment
# enough that `find_quantiles` hits its `new_q < 2` fallback (one boundary at
# the segment max), so every spike lands in a *pure singleton* bin — the spike
# value is the only thing in `(segment_max, spike]`. A lower p_spike leaves
# lognormal values just below a spike, so the spike shares its bin (only
# Fare=0 stays a clean singleton because nothing lies below 0).
#
# Post-CD this yields 9 bins, 4 of them pure spike singletons (0, 7.25, 13,
# 26.55) and 3 of them rare (Wilson CI below QD_MIN_FREQ): the sparse segment
# `(13, 25.9]`, the tail `(26.55, 33.9]`, and the always-empty `> 33.9` bin
# that `GroupedList(... + [inf])` appends. OrdinalDiscretizer absorbs each rare
# bin into the dominant (singleton) neighbour, collapsing to 6 bins.
QD_P_SPIKE = 0.78
QD_SPIKES = (0.0, 7.25, 13.0, 26.55)


def quantitative_binary_frames() -> list[Frame]:
    """Five stages: KDE → over-rep markers → after-CD bars with rare outlined
    → merge-direction arrows → after-QD merged bars. Stages 0-2 are a strict
    visual prefix of the ContinuousDiscretizer animation (same x-axis, same bar
    idiom); Stages 3-4 reuse the OrdinalDiscretizer arrow + merged-span idiom.
    """
    fare, y = _synthesize_quantitative()
    nan_mask = np.isnan(fare)
    n = len(fare)
    nan_freq = float(nan_mask.sum()) / n

    # ----- Stage 0/1 density curve ------------------------------------------
    kde = gaussian_kde(fare[~nan_mask], bw_method=KDE_BANDWIDTH)
    x_samples = np.linspace(0.0, QD_FARE_MAX, 160)
    density = kde(x_samples)
    density = density / density.max()
    density_curve = tuple((float(x) / QD_FARE_MAX, float(d)) for x, d in zip(x_samples, density))
    overrep_markers = tuple((s / QD_FARE_MAX, f"Fare = {_fmt(s)}") for s in QD_SPIKES)

    # ----- Stage 2: ContinuousDiscretizer alone (so we can highlight the
    # post-CD rare bins before OD absorbs them) ------------------------------
    df = pd.DataFrame({"Fare": fare})
    features_cd = Features(numericals=["Fare"])
    cd = ContinuousDiscretizer(quantitatives=features_cd.quantitatives, min_freq=QD_MIN_FREQ)
    transformed_cd = cd.fit_transform(df.copy())
    feature_cd = features_cd.quantitatives[0]
    counts_cd = transformed_cd["Fare"].value_counts(dropna=False)
    rates_cd = (
        pd.DataFrame({"Fare": transformed_cd["Fare"], "y": y.to_numpy()}).groupby("Fare", observed=True)["y"].mean()
    )
    s2_bins_all = _qd_bins_on_value_axis(
        feature_cd.labels,
        feature_cd.values,
        counts_cd,
        rates_cd,
        n,
    )
    # Drop the always-empty > fare_max bin that CD appends internally via
    # GroupedList(... + [inf]). It is an implementation detail, not a meaningful
    # modality to show in the animation.
    cd_keep = [i for i, b in enumerate(s2_bins_all) if b.freq > 0]
    s2_bins = tuple(s2_bins_all[i] for i in cd_keep)
    cd_labels = [feature_cd.labels[i] for i in cd_keep]
    cd_values = [feature_cd.values[i] for i in cd_keep]

    rare_idx = tuple(
        i
        for i, lbl in enumerate(cd_labels)
        if is_significantly_below(
            int(counts_cd.get(lbl, 0)),
            n,
            QD_MIN_FREQ,
            QD_MIN_FREQ_ALPHA,
        )
    )

    # ----- Stage 4: full QuantitativeDiscretizer (CD + OD merge) ------------
    features_qd = Features(numericals=["Fare"])
    qd = QuantitativeDiscretizer(quantitatives=features_qd.quantitatives, min_freq=QD_MIN_FREQ)
    qd.fit(df.copy(), y)
    feature_qd = features_qd.quantitatives[0]
    transformed_qd = qd.transform(df.copy())
    counts_qd = transformed_qd["Fare"].value_counts(dropna=False)
    rates_qd = (
        pd.DataFrame({"Fare": transformed_qd["Fare"], "y": y.to_numpy()}).groupby("Fare", observed=True)["y"].mean()
    )
    # Each post-OD bin is a contiguous run of pre-OD bins (post boundaries ⊆ pre
    # boundaries). Group them so the merged bar can inherit the *dominant*
    # (anchor) pre-OD bin's colour and so arrows point rare → anchor — not the
    # `feature.content` "kept = rightmost boundary" direction, which would point
    # a singleton into the rare segment that absorbed its boundary.
    groups = _qd_merge_groups(
        pre_uppers=[float(v) for v in cd_values],
        pre_freqs=[b.freq for b in s2_bins],
        post_uppers=[float(v) for v in feature_qd.values],
    )
    s4_bins = _qd_merged_bins(
        feature_qd.labels,
        feature_qd.values,
        counts_qd,
        rates_qd,
        n,
        groups=groups,
        s2_bins=s2_bins,
    )

    # ----- Stage 3: merge arrows on the pre-OD layout -----------------------
    arrows = _qd_merge_arrows(groups, s2_bins)

    # ----- Frame-wide knobs (consistent across the morph) -------------------
    # Skip empty bins (e.g. CD's `> max_fare` 0-row bin): their target=0
    # would force the zoom to start at 0, collapsing the rest of the dots.
    target_strip_min, target_strip_max = _zoom_target_range([b.target for b in s2_bins + s4_bins if b.freq > 0])
    bar_max_freq = max(b.freq for b in s2_bins + s4_bins)
    min_freq_y_norm = QD_MIN_FREQ / bar_max_freq
    xaxis_ticks = tuple((v / QD_FARE_MAX, str(int(v))) for v in range(0, int(QD_FARE_MAX) + 1, TICK_STEP))
    nan_bin = Bin(
        label="NaN",
        x_start=0.0,
        x_end=1.0,
        freq=nan_freq,
        target=0.0,
        color_id=7,
        is_nan=True,
    )

    q = int(round(1.0 / QD_MIN_FREQ))
    common = {
        "min_freq_y_norm": min_freq_y_norm,
        "min_freq_label": f"min_freq = {QD_MIN_FREQ:.2f}",
        "bar_max_freq": bar_max_freq,
        "target_strip_max": target_strip_max,
        "target_strip_min": target_strip_min,
    }

    return [
        Frame(
            0,
            "Raw feature",
            bins=(),
            nan_bin=nan_bin,
            metric="—",
            callout=(
                f"Raw continuous Fare (synthetic, n={QD_N_ROWS}) with multiple "
                "class-fare spikes (0, 7.25, 13, 26.55) on a lognormal body."
            ),
            density_curve=density_curve,
            tick_values=xaxis_ticks,
            **common,
        ),
        Frame(
            1,
            "Over-represented values detected",
            bins=(),
            nan_bin=nan_bin,
            metric="—",
            callout=(
                f"ContinuousDiscretizer flags values occurring ≥ 1/q ({1 / q:.2f}) "
                "as their own singleton bin — all four class-fare spikes qualify."
            ),
            density_curve=density_curve,
            overrep_markers=overrep_markers,
            tick_values=xaxis_ticks,
            **common,
        ),
        Frame(
            2,
            "After ContinuousDiscretizer",
            bins=s2_bins,
            nan_bin=nan_bin,
            metric="—",
            callout=(
                f"{len(s2_bins)} CD bins: 4 thin spike singletons + quantile "
                "bins. Those with Wilson upper bound below min_freq are orange."
            ),
            highlight_bins=rare_idx,
            **common,
        ),
        Frame(
            3,
            "Merge direction chosen",
            bins=s2_bins,
            nan_bin=nan_bin,
            metric="—",
            callout=(
                "OrdinalDiscretizer merges each rare bin into its dominant "
                "neighbour — arrows point from the sparse bin to the bin that "
                "absorbs it."
            ),
            highlight_bins=rare_idx,
            merge_arrows=arrows,
            **common,
        ),
        Frame(
            4,
            "After QuantitativeDiscretizer",
            bins=s4_bins,
            nan_bin=nan_bin,
            metric="—",
            callout=(
                f"{len(s4_bins)} bins after CD + OD. Each merged bar spans its "
                "absorbed Stage-2 slots and keeps the dominant bin's colour."
            ),
            **common,
        ),
    ]


def _synthesize_quantitative() -> tuple[np.ndarray, pd.Series]:
    rng = np.random.default_rng(QD_SEED)
    nan_mask = rng.random(QD_N_ROWS) < QD_P_NAN
    n_nonnan = int((~nan_mask).sum())
    base = rng.lognormal(LOGNORMAL_MU, LOGNORMAL_SIGMA, size=n_nonnan)
    spike_assign = rng.random(n_nonnan) < QD_P_SPIKE
    pick = rng.choice(QD_SPIKES, size=n_nonnan)
    values = np.where(spike_assign, pick, base)
    values[values > QD_FARE_MAX] = np.nan
    fare = np.full(QD_N_ROWS, np.nan)
    fare[~nan_mask] = values

    # Same logistic-on-fare target as the CD example, for visual continuity.
    rng2 = np.random.default_rng(QD_SEED + 1)
    fare_for_p = np.where(np.isnan(fare), CONT_LOGISTIC_MID, fare)
    p = 1.0 / (1.0 + np.exp(-CONT_LOGISTIC_SLOPE * (fare_for_p - CONT_LOGISTIC_MID)))
    y = (rng2.random(QD_N_ROWS) < p).astype(int)
    return fare, pd.Series(y, name="Survived")


def _qd_value_axis_layout(
    labels: list[str],
    values: list,
    value_max: float = QD_FARE_MAX,
) -> list[tuple[float, float]]:
    """Shared (lo, hi] value-axis placement for Stage 2 and Stage 4 bars.

    Singleton bins (value range < 0.5 fare units — the over-rep spikes) get a
    fixed `SINGLETON_STRIP_PCT` width so they stay visible; the cursor bookkeeping
    shifts the next bar right so bars never overlap.
    """
    boundaries = [float(v) for v in values]
    singleton_w = SINGLETON_STRIP_PCT * value_max
    spans: list[tuple[float, float]] = []
    x_cursor = 0.0
    for i in range(len(labels)):
        upper = boundaries[i]
        lower = boundaries[i - 1] if i > 0 else float("-inf")
        true_hi = value_max if np.isinf(upper) else min(upper, value_max)
        true_lo = 0.0 if (np.isinf(lower) or lower < 0) else lower
        lo = max(x_cursor, true_lo)
        if true_hi - true_lo < 0.5:  # singleton (over-rep spike)
            hi = lo + singleton_w
        else:
            hi = true_hi if true_hi > lo else lo + singleton_w / 2
        x_cursor = hi
        spans.append((lo / value_max, hi / value_max))
    return spans


def _qd_bins_on_value_axis(
    labels: list[str],
    values: list,
    counts: pd.Series,
    target_rates: pd.Series,
    n_total: int,
    value_max: float = QD_FARE_MAX,
) -> tuple[Bin, ...]:
    """Stage 2 bars: one per CD modality, coloured by position in the palette."""
    spans = _qd_value_axis_layout(labels, values, value_max)
    bins: list[Bin] = []
    for i, raw_label in enumerate(labels):
        x_start, x_end = spans[i]
        bins.append(
            Bin(
                label=_clean_label(raw_label),
                x_start=x_start,
                x_end=x_end,
                freq=float(counts.get(raw_label, 0)) / n_total,
                target=float(target_rates.get(raw_label, 0.0)),
                color_id=QUANTILE_PALETTE[i % len(QUANTILE_PALETTE)],
            )
        )
    return tuple(bins)


def _qd_merged_bins(
    labels: list[str],
    values: list,
    counts: pd.Series,
    target_rates: pd.Series,
    n_total: int,
    groups: list[tuple[int, list[int]]],
    s2_bins: tuple[Bin, ...],
) -> tuple[Bin, ...]:
    """Stage 4 bars. Each post-OD bin inherits the colour of its anchor — the
    dominant (highest-frequency) pre-OD bin it absorbed — so the surviving bin's
    colour carries across the morph instead of an absorbed bin's colour."""
    spans = _qd_value_axis_layout(labels, values)
    bins: list[Bin] = []
    for i, raw_label in enumerate(labels):
        x_start, x_end = spans[i]
        anchor = groups[i][0]
        color_id = s2_bins[anchor].color_id if anchor >= 0 else (QUANTILE_PALETTE[i % len(QUANTILE_PALETTE)])
        bins.append(
            Bin(
                label=_clean_label(raw_label),
                x_start=x_start,
                x_end=x_end,
                freq=float(counts.get(raw_label, 0)) / n_total,
                target=float(target_rates.get(raw_label, 0.0)),
                color_id=color_id,
            )
        )
    return tuple(bins)


def _qd_merge_groups(
    pre_uppers: list[float],
    pre_freqs: list[float],
    post_uppers: list[float],
) -> list[tuple[int, list[int]]]:
    """Map each post-OD bin to the pre-OD bins it absorbed.

    Post-OD boundaries are a subset of the pre-OD boundaries, so each post-OD
    bin is a contiguous run of pre-OD bins. Within each run the highest-frequency
    bin is the *anchor* (the modality that visually wins the merge); the others
    are the rare bins absorbed into it. Returns one
    `(anchor_pre_idx, [member_pre_idx, ...])` per post-OD bin (anchor = -1 when a
    post-OD bin has no members, which shouldn't happen for well-formed input).
    """
    groups: list[tuple[int, list[int]]] = []
    pre_i = 0
    for post_up in post_uppers:
        members: list[int] = []
        while pre_i < len(pre_uppers) and _le_boundary(pre_uppers[pre_i], post_up):
            members.append(pre_i)
            pre_i += 1
        anchor = max(members, key=lambda j: pre_freqs[j]) if members else -1
        groups.append((anchor, members))
    return groups


def _qd_merge_arrows(
    groups: list[tuple[int, list[int]]],
    s2_bins: tuple[Bin, ...],
) -> tuple[MergeArrow, ...]:
    """One arrow per absorbed (non-anchor) pre-OD bin, pointing from its centre
    to the anchor bin's centre. Labelled with the absorbed bin's interval."""
    arrows: list[MergeArrow] = []
    for anchor, members in groups:
        if anchor < 0 or len(members) <= 1:
            continue
        anchor_c = (s2_bins[anchor].x_start + s2_bins[anchor].x_end) / 2
        for m in members:
            if m == anchor:
                continue
            m_c = (s2_bins[m].x_start + s2_bins[m].x_end) / 2
            arrows.append(
                MergeArrow(
                    from_x=m_c,
                    to_x=anchor_c,
                    label=s2_bins[m].label,
                )
            )
    return tuple(arrows)


# =============================================================================
# QualitativeDiscretizer animation
# =============================================================================
#
# Composite: CategoricalDiscretizer on Port (top strip) then OrdinalDiscretizer
# on AgeGroup (bottom strip). Each strip reuses the exact same synthetic data
# and fitted frames as the standalone CD and OD animations — no new synthesis.
# Four stages:
#   0 — raw state on both strips
#   1 — after CD (top strip transforms; bottom dimmed at 60 % opacity)
#   2 — OD merge arrows on bottom strip; top stays at CD result
#   3 — after QD: bottom shows merged bars; both strips at full opacity


def qualitative_binary_frames() -> list[DualFrame]:
    cat = categorical_binary_frames()  # [raw, rare_grouped, sorted]
    ord_ = ordinal_binary_frames()  # [raw, arrows, merged]

    # Stages 0-2 walk through all CD steps (bottom strip dimmed).
    # Stages 3-4 walk through all OD steps (top strip at full opacity — already done).
    return [
        DualFrame(
            stage=0,
            title="Raw features",
            top=cat[0],
            bot=ord_[0],
            callout=(
                "Port (categorical, top) and AgeGroup (ordinal, bottom) in their raw state. "
                "Rare modalities are outlined orange on both strips."
            ),
        ),
        DualFrame(
            stage=1,
            title="Rare modalities grouped",
            top=cat[1],
            bot=ord_[0],
            bot_opacity=0.6,
            callout=(
                "CategoricalDiscretizer: rare Port modalities (Belfast, Boston) collapse "
                "into __OTHER__. AgeGroup is unchanged (dimmed — not yet processed)."
            ),
        ),
        DualFrame(
            stage=2,
            title="After CategoricalDiscretizer",
            top=cat[2],
            bot=ord_[0],
            bot_opacity=0.6,
            callout=("Port bars reordered by ascending P(y=1) — dot trace is now monotonic. AgeGroup still unchanged."),
        ),
        DualFrame(
            stage=3,
            title="OrdinalDiscretizer — merge direction",
            top=cat[2],
            bot=ord_[1],
            callout=(
                "OrdinalDiscretizer: each rare AgeGroup modality merges with the adjacent "
                "neighbour whose target rate is closest. Port is already done."
            ),
        ),
        DualFrame(
            stage=4,
            title="After QualitativeDiscretizer",
            top=cat[2],
            bot=ord_[2],
            callout=("Port: 4 modalities sorted by target rate. AgeGroup: 3 groups, ordinal order preserved."),
        ),
    ]


def _le_boundary(a: float, b: float) -> bool:
    if np.isinf(b):
        return True
    if np.isinf(a):
        return False
    return a <= b + 1e-9


def _ordinal_stage2_bins(
    feature_labels: list[str],
    feature_values: list[str],
    content: dict[str, list[str]],
    order: tuple[str, ...],
    raw_counts: pd.Series,
    raw_rates: dict,
    n_total: int,
) -> tuple[Bin, ...]:
    """Stage-2 layout: bars span the union of their absorbed modalities' Stage-0
    slots, so the merge reads as "this region collapses into one bar". Colour =
    the kept modality's colour. Target rate = frequency-weighted average over
    the absorbed modalities."""
    slot_w = 1.0 / len(order)
    slot_index = {m: i for i, m in enumerate(order)}
    bins: list[Bin] = []
    for label, kept in zip(feature_labels, feature_values):
        members = list(content[kept])
        member_slots = sorted(slot_index[m] for m in members)
        x_start = member_slots[0] * slot_w
        x_end = (member_slots[-1] + 1) * slot_w
        member_counts = [int(raw_counts.get(m, 0)) for m in members]
        total = sum(member_counts)
        freq = total / n_total if n_total else 0.0
        target = sum(c * raw_rates[m] for c, m in zip(member_counts, members)) / total if total else 0.0
        bins.append(
            Bin(
                label=str(label),
                x_start=x_start,
                x_end=x_end,
                freq=freq,
                target=target,
                color_id=ORD_COLOR_BY_LABEL[kept],
            )
        )
    return tuple(bins)


# =============================================================================
# Combinations animation (base of every carver)
# =============================================================================
#
# Starts from the QuantitativeDiscretizer stage-4 output (6 ordered bins) and
# shows the *core carver step*: enumerate every consecutive grouping of those
# bins, score each by association with the binary target (Tschuprow's T), and
# keep the best viable one. The real `_top_k_partitions_chi2_dp` ranks them;
# the table fills best-first in growing top-K batches (the progressive-doubling
# search), and the selected combination is highlighted.
#
# Scope: this animates the *non-NaN* consecutive search on the 6 QD bins (NaN
# fan-out is a secondary step). Fidelity note: a real BinaryCarver re-discretizes
# at `half_min_freq` before the search, so its bins are finer than these 6 — we
# use the QD bins for visual continuity with the QuantitativeDiscretizer
# animation (its stage 4 is exactly this strip).

COMBI_MAX_N_MOD = 5  # max groups per combination (carver tip range: 5–7)
COMBI_SHOW_ROWS = 8  # ranked rows displayed in the table
COMBI_TOPK_BATCHES = (2, 4, 8)  # rows revealed per stage (progressive top-K)


def combinations_binary_frames() -> list[TableFrame]:
    fare, y = _synthesize_quantitative()
    df = pd.DataFrame({"Fare": fare})

    # ----- Real QD fit: the 6 ordered bins fed to the combination search ------
    features = Features(numericals=["Fare"])
    qd = QuantitativeDiscretizer(quantitatives=features.quantitatives, min_freq=QD_MIN_FREQ)
    qd.fit(df.copy(), y)
    feature = features.quantitatives[0]
    transformed = qd.transform(df.copy())
    labels = list(feature.labels)
    n_bins = len(labels)

    # Per-modality (n0, n1) in label order, plus per-bin freq / target rate.
    xtab = pd.crosstab(transformed["Fare"], y).reindex(labels, fill_value=0)
    n0 = xtab[0].to_numpy(dtype=float)
    n1 = xtab[1].to_numpy(dtype=float)
    nobs = int(n0.sum() + n1.sum())

    input_bins = tuple(
        Bin(
            label=_clean_label(lbl),
            x_start=i / n_bins,
            x_end=(i + 1) / n_bins,
            freq=(n0[i] + n1[i]) / nobs,
            target=float(n1[i] / (n0[i] + n1[i])) if (n0[i] + n1[i]) else 0.0,
            color_id=QUANTILE_PALETTE[i % len(QUANTILE_PALETTE)],
        )
        for i, lbl in enumerate(labels)
    )

    # ----- Rank every consecutive grouping by Tschuprow's T (real DP) ---------
    ranked = _top_k_partitions_chi2_dp(
        n0, n1, max_n_mod=COMBI_MAX_N_MOD, raw_index=labels, sort_by="tschuprowt", top_k=10_000
    )
    n_total = len(ranked)
    pos = {lbl: i for i, lbl in enumerate(labels)}

    winner_rank = next(
        (rk for rk, r in enumerate(ranked) if _combo_viable(r["combination"], pos, n0, n1, nobs)),
        None,
    )

    rows: list[ComboRow] = []
    for rk, r in enumerate(ranked[:COMBI_SHOW_ROWS]):
        groups = tuple(tuple(pos[m] for m in g) for g in r["combination"])
        rows.append(
            ComboRow(
                rank=rk,
                groups=groups,
                tschuprowt=float(r["tschuprowt"]),
                is_winner=(rk == winner_rank),
            )
        )

    winner_t = ranked[winner_rank]["tschuprowt"] if winner_rank is not None else None
    metric = f"T = {winner_t:.3f}" if winner_t is not None else "—"

    # ----- Stage 0: input strip only; Stages 1..n: growing top-K batches ------
    frames: list[TableFrame] = [
        TableFrame(
            stage=0,
            title="Carver input: ordered bins",
            input_bins=input_bins,
            rows=(),
            top_k=0,
            n_total=n_total,
            metric="—",
            callout=(
                f"The carver starts from {n_bins} ordered bins (QuantitativeDiscretizer output) and tries "
                f"every consecutive grouping into ≤ {COMBI_MAX_N_MOD} groups — {n_total} in total."
            ),
        )
    ]
    for s, k in enumerate(COMBI_TOPK_BATCHES, start=1):
        revealed = rows[:k]
        is_last = s == len(COMBI_TOPK_BATCHES)
        frames.append(
            TableFrame(
                stage=s,
                title=("Best grouping selected" if is_last else "Ranking groupings by association"),
                input_bins=input_bins,
                rows=tuple(
                    ComboRow(rk.rank, rk.groups, rk.tschuprowt, is_winner=rk.is_winner and is_last) for rk in revealed
                ),
                top_k=k,
                n_total=n_total,
                metric=metric if is_last else "—",
                callout=(
                    (
                        "Highest-T grouping that passes the viability checks (min_freq + distinct rates) "
                        "is kept — here the 2-group split."
                    )
                    if is_last
                    else (
                        f"Consecutive groupings ranked by Tschuprow's T (top {k} shown); each row is one "
                        "candidate — adjacent bins sharing a colour are merged."
                    )
                ),
            )
        )
    return frames


# =============================================================================
# README hero animation (full pipeline: Discretizers + Carvers)
# =============================================================================
#
# The first canvas combining both idioms: the feature strip (Discretizers) on
# top and the ranked-groupings table (Carvers) below. One feature travels the
# full pipeline: raw → discretized → ranked → carved.
#
# Data story: synthetic Titanic-flavoured Age vs Survived with a NON-monotone
# survival curve (children high, young adults low, middle-aged higher, elderly
# lowest) and 20 % NaN. The non-monotone shape matters: with a monotone target
# Tschuprow's T always picks a 2-bucket split (the sqrt(r-1) penalty beats any
# finer cut), while the N-shaped plateaus give an honest 4-bucket winner — the
# "binning captures non-linear risk" pitch.

HERO_SEED = 7
HERO_N_ROWS = 1500
HERO_AGE_MAX = 80.0
HERO_P_NAN = 0.20
HERO_MIN_FREQ = 0.07
HERO_MAX_N_MOD = 5
HERO_SHOW_ROWS = 5
HERO_TICK_STEP = 10
HERO_KDE_BANDWIDTH = 0.15  # Age has no spikes; Fare's 0.04 over-sharpens here
HERO_P_CHILD = 0.15  # mixture weight of the child bump
# (upper_age, P(survived)) plateaus — the N-shaped risk the carver recovers.
HERO_RATE_PLATEAUS = ((10.0, 0.72), (32.0, 0.36), (50.0, 0.52), (float("inf"), 0.20))
HERO_NAN_TARGET_RATE = 0.35


def readme_binary_frames() -> list[HeroFrame]:
    """Five stages: raw KDE → discretized bins → ranked groupings table →
    winner selected (gold cuts) → carved buckets. Strip frames reuse the
    single-strip layout; the table reuses the combinations row idiom."""
    age, y = _synthesize_hero()
    nan_mask = np.isnan(age)
    n = len(age)
    nan_freq = float(nan_mask.sum()) / n

    # ----- Stage 0 density curve ---------------------------------------------
    kde = gaussian_kde(age[~nan_mask], bw_method=HERO_KDE_BANDWIDTH)
    x_samples = np.linspace(0.0, HERO_AGE_MAX, 160)
    density = kde(x_samples)
    density = density / density.max()
    density_curve = tuple((float(x) / HERO_AGE_MAX, float(d)) for x, d in zip(x_samples, density))
    xaxis_ticks = tuple((v / HERO_AGE_MAX, str(int(v))) for v in range(0, int(HERO_AGE_MAX) + 1, HERO_TICK_STEP))

    # ----- Stage 1: real QuantitativeDiscretizer ------------------------------
    df = pd.DataFrame({"Age": age})
    features = Features(numericals=["Age"])
    qd = QuantitativeDiscretizer(quantitatives=features.quantitatives, min_freq=HERO_MIN_FREQ)
    qd.fit(df.copy(), y)
    feature = features.quantitatives[0]
    transformed = qd.transform(df.copy())
    labels = list(feature.labels)
    values = list(feature.values)

    xtab = pd.crosstab(transformed["Age"], y).reindex(labels, fill_value=0)
    n0 = xtab[0].to_numpy(dtype=float)
    n1 = xtab[1].to_numpy(dtype=float)
    nobs = int(n0.sum() + n1.sum())
    counts = transformed["Age"].value_counts(dropna=False)
    rates = pd.DataFrame({"Age": transformed["Age"], "y": y.to_numpy()}).groupby("Age", observed=True)["y"].mean()
    s1_bins = _qd_bins_on_value_axis(labels, values, counts, rates, n, HERO_AGE_MAX)

    # ----- Stages 2-3: rank every consecutive grouping (real DP) --------------
    ranked = _top_k_partitions_chi2_dp(
        n0, n1, max_n_mod=HERO_MAX_N_MOD, raw_index=labels, sort_by="tschuprowt", top_k=10_000
    )
    n_total = len(ranked)
    pos = {lbl: i for i, lbl in enumerate(labels)}
    winner_rank = next(
        (rk for rk, r in enumerate(ranked) if _combo_viable(r["combination"], pos, n0, n1, nobs)),
        None,
    )
    rows = tuple(
        ComboRow(
            rank=rk,
            groups=tuple(tuple(pos[m] for m in g) for g in r["combination"]),
            tschuprowt=float(r["tschuprowt"]),
        )
        for rk, r in enumerate(ranked[:HERO_SHOW_ROWS])
    )
    winner_groups = rows[winner_rank].groups
    winner_t = float(ranked[winner_rank]["tschuprowt"])
    metric = f"T = {winner_t:.3f}"
    rows_with_winner = tuple(ComboRow(r.rank, r.groups, r.tschuprowt, is_winner=(r.rank == winner_rank)) for r in rows)
    # Interior cut points of the winning grouping, in strip coordinates.
    winner_cuts = tuple(s1_bins[max(g)].x_end for g in winner_groups[:-1])

    # ----- Stage 4: carved buckets (winner groups merged over s1 spans) -------
    s4_bins = _hero_merged_bins(winner_groups, s1_bins, values)
    # The stage-4 table row indexes the *merged* strip bins (one block per
    # bucket) — the stage-1 indices in `winner_groups` would point past them.
    winner_row_s4 = ComboRow(
        rank=winner_rank,
        groups=tuple((i,) for i in range(len(s4_bins))),
        tschuprowt=winner_t,
        is_winner=True,
    )

    # ----- Frame-wide knobs (consistent across the morph) ---------------------
    target_strip_min, target_strip_max = _zoom_target_range([b.target for b in s1_bins + s4_bins])
    bar_max_freq = max(b.freq for b in s1_bins + s4_bins)
    nan_bin = Bin(label="NaN", x_start=0.0, x_end=1.0, freq=nan_freq, target=0.0, color_id=7, is_nan=True)
    strip_common = {
        "nan_bin": nan_bin,
        "metric": "—",
        "min_freq_y_norm": HERO_MIN_FREQ / bar_max_freq,
        "min_freq_label": f"min_freq = {HERO_MIN_FREQ:.2f}",
        "bar_max_freq": bar_max_freq,
        "target_strip_max": target_strip_max,
        "target_strip_min": target_strip_min,
    }
    raw_strip = Frame(0, "", bins=(), density_curve=density_curve, tick_values=xaxis_ticks, **strip_common)
    binned_strip = Frame(1, "", bins=s1_bins, **strip_common)
    carved_strip = Frame(4, "", bins=s4_bins, **strip_common)

    return [
        HeroFrame(
            stage=0,
            title="Raw feature",
            strip=raw_strip,
            rows=(),
            top_k=0,
            n_total=n_total,
            callout=(
                f"A raw continuous feature: Age (synthetic Titanic-flavoured, n={HERO_N_ROWS}) — "
                "skewed, non-monotone risk, 20% missing."
            ),
        ),
        HeroFrame(
            stage=1,
            title="Discretizers — split into clean ordered bins",
            strip=binned_strip,
            rows=(),
            top_k=0,
            n_total=n_total,
            callout=(
                f"QuantitativeDiscretizer splits Age into {len(s1_bins)} ordered bins above min_freq. "
                "Categorical, ordinal and datetime features get the same treatment."
            ),
        ),
        HeroFrame(
            stage=2,
            title="Carvers — rank every consecutive grouping",
            strip=binned_strip,
            rows=rows,
            top_k=HERO_SHOW_ROWS,
            n_total=n_total,
            callout=(
                f"BinaryCarver tries every consecutive grouping into ≤ {HERO_MAX_N_MOD} buckets — "
                f"{n_total} candidates ranked by Tschuprow's T. Adjacent bins sharing a colour form one bucket."
            ),
        ),
        HeroFrame(
            stage=3,
            title="Best viable grouping selected",
            strip=binned_strip,
            rows=rows_with_winner,
            top_k=HERO_SHOW_ROWS,
            n_total=n_total,
            metric=metric,
            winner_cuts=winner_cuts,
            callout=(
                "The highest-T grouping that stays viable on a held-out dev set "
                "(min_freq + distinct target rates) wins — gold cuts on the strip."
            ),
        ),
        HeroFrame(
            stage=4,
            title="Carved feature",
            strip=carved_strip,
            rows=(winner_row_s4,),
            top_k=0,
            n_total=n_total,
            metric=metric,
            callout=(
                f"{len(s4_bins)} interpretable buckets capture the non-monotone age–survival risk — "
                "ready for scorecards, WOE or one-hot encoding."
            ),
        ),
    ]


def _synthesize_hero() -> tuple[np.ndarray, pd.Series]:
    rng = np.random.default_rng(HERO_SEED)
    nan_mask = rng.random(HERO_N_ROWS) < HERO_P_NAN
    n_nonnan = int((~nan_mask).sum())
    is_child = rng.random(n_nonnan) < HERO_P_CHILD
    child = rng.normal(8, 4, n_nonnan).clip(0.5, 14)
    adult = rng.lognormal(np.log(30), 0.40, n_nonnan).clip(15, HERO_AGE_MAX)
    age = np.full(HERO_N_ROWS, np.nan)
    age[~nan_mask] = np.where(is_child, child, adult)

    rng2 = np.random.default_rng(HERO_SEED + 1)
    p = np.array([_hero_rate(a) for a in age])
    y = (rng2.random(HERO_N_ROWS) < p).astype(int)
    return age, pd.Series(y, name="Survived")


def _hero_rate(a: float) -> float:
    if np.isnan(a):
        return HERO_NAN_TARGET_RATE
    for upper, rate in HERO_RATE_PLATEAUS:
        if a <= upper:
            return rate
    return HERO_RATE_PLATEAUS[-1][1]


def _hero_merged_bins(
    winner_groups: tuple[tuple[int, ...], ...],
    s1_bins: tuple[Bin, ...],
    values: list,
) -> tuple[Bin, ...]:
    """One bar per winning bucket: union span of its members' Stage-1 slots,
    summed frequency, frequency-weighted target rate, anchor (highest-freq
    member) colour — same conventions as `_qd_merged_bins`."""
    # Same 3-significant-digit rounding as the scientific notation that
    # `_clean_label` parses, so merged labels match the Stage-1 bin labels.
    boundaries = [float(f"{float(v):.2e}") for v in values]
    bins: list[Bin] = []
    for group in winner_groups:
        members = [s1_bins[i] for i in group]
        freq = sum(b.freq for b in members)
        target = sum(b.freq * b.target for b in members) / freq if freq else 0.0
        anchor = max(members, key=lambda b: b.freq)
        lower = boundaries[min(group) - 1] if min(group) > 0 else float("-inf")
        upper = boundaries[max(group)]
        if np.isinf(lower):
            label = f"≤ {_fmt(upper)}"
        elif np.isinf(upper):
            label = f"> {_fmt(lower)}"
        else:
            label = f"({_fmt(lower)}, {_fmt(upper)}]"
        bins.append(
            Bin(
                label=label,
                x_start=members[0].x_start,
                x_end=members[-1].x_end,
                freq=freq,
                target=target,
                color_id=anchor.color_id,
            )
        )
    return tuple(bins)


def _combo_viable(
    combination: list[list[str]],
    pos: dict[str, int],
    n0: np.ndarray,
    n1: np.ndarray,
    nobs: int,
) -> bool:
    """Mirror `test_viability` for one grouping: no group significantly below
    min_freq (Wilson CI) and consecutive groups have distinct target rates."""
    counts: list[float] = []
    rates: list[float] = []
    for g in combination:
        idx = [pos[m] for m in g]
        c = float(sum(n0[j] + n1[j] for j in idx))
        counts.append(c)
        rates.append(float(sum(n1[j] for j in idx) / c) if c else 0.0)
    min_freq_ok = not bool(np.any(is_significantly_below(np.array(counts), nobs, QD_MIN_FREQ, QD_MIN_FREQ_ALPHA)))
    rate_arr = np.array(rates)
    distinct = not bool(np.any(np.isclose(rate_arr[1:], rate_arr[:-1]))) if len(rate_arr) > 1 else True
    return min_freq_ok and distinct
