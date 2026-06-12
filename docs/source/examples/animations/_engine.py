"""Frame data structures and the build_animation() entry point.

A `Frame` is a snapshot of one pipeline stage; `build_animation` returns the list
of frames for a given (feature_type, target_type, stop_after_stage) variant. See
`docs/animation_examples_plan.md` for the conceptual model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

FeatureType = Literal["continuous", "ordinal", "categorical", "quantitative", "qualitative", "combinations", "readme"]
TargetType = Literal["binary", "multiclass", "continuous"]


@dataclass(frozen=True)
class Bin:
    """One bar on the feature strip at one stage.

    `x_start` / `x_end` are normalized to [0, 1] along the strip's drawing area.
    `color_id` is an index into the palette and is preserved across merges so a
    bin "swallowing" its neighbour keeps the same color in the same region.
    """

    label: str
    x_start: float
    x_end: float
    freq: float
    target: float  # P(y=1) for binary
    color_id: int
    is_nan: bool = False


@dataclass(frozen=True)
class Ghost:
    """One losing candidate grouping (Stage 3 only)."""

    bins: tuple[Bin, ...]
    metric: str  # e.g. "T = 0.175"


@dataclass(frozen=True)
class MergeArrow:
    """Dashed arrow drawn inside the feature strip — visualizes which sub-zone
    of an already-merged bin came from a now-absorbed neighbour.
    """

    from_x: float  # normalized start x in [0, 1] (inside the feature strip)
    to_x: float  # normalized end x in [0, 1]
    label: str  # e.g. "absorbed [0, 0]"


@dataclass(frozen=True)
class Frame:
    stage: int
    title: str
    bins: tuple[Bin, ...]  # empty when rendering a density curve instead
    nan_bin: Bin | None
    metric: str
    ghosts: tuple[Ghost, ...] = field(default_factory=tuple)  # kept for rollback
    callout: str = ""
    highlight_bins: tuple[int, ...] = ()
    merge_arrows: tuple[MergeArrow, ...] = ()
    feature_as_histogram: bool = False
    density_curve: tuple[tuple[float, float], ...] = ()  # (x_norm, y_norm) samples
    overrep_markers: tuple[tuple[float, str], ...] = ()  # (x_norm, label) verticals
    tick_values: tuple[tuple[float, str], ...] = ()  # (x_norm, text) — unused now
    min_freq_y_norm: float | None = None  # y position of min_freq reference line
    min_freq_label: str = ""  # e.g. "min_freq = 0.07"
    bar_max_freq: float = 1.0  # freq used to scale bar heights (Stage 2)
    # When set, the renderer reserves the top strip of MAIN_H for target-rate
    # dots scaled to [target_strip_min, target_strip_max] (one dot per bin,
    # connected by a light line). Bars + min_freq line scale to the remaining
    # lower zone. Zoom the strip range (rather than anchoring at 0) to keep
    # small differences between modality target rates readable.
    target_strip_max: float | None = None
    target_strip_min: float = 0.0


@dataclass(frozen=True)
class DualFrame:
    """Two stacked feature strips for QualitativeDiscretizer.

    `top` = categorical strip (Port), `bot` = ordinal strip (AgeGroup).
    `bot_opacity` is set to 0.6 on the stage where only the top strip is active,
    to draw the eye to the transformation that is happening.
    """

    stage: int
    title: str
    top: Frame
    bot: Frame
    callout: str = ""
    metric: str = "—"
    top_opacity: float = 1.0
    bot_opacity: float = 1.0


@dataclass(frozen=True)
class ComboRow:
    """One consecutive-grouping candidate in the combinations table.

    `groups` is the partition of the input bins as index tuples (each group a
    run of contiguous input-bin indices, e.g. ((0, 1, 2, 3), (4, 5))). The row
    is drawn as a segmented bar: each group is one coloured block spanning its
    members' slots, coloured by its leader (leftmost) input bin so the block
    reads as "these adjacent bins merged".
    """

    rank: int
    groups: tuple[tuple[int, ...], ...]
    tschuprowt: float
    is_winner: bool = False


@dataclass(frozen=True)
class TableFrame:
    """One stage of the combinations animation.

    `input_bins` is the ordered post-discretizer strip (drawn once at the top,
    identical across stages). `rows` are the top-K ranked groupings revealed so
    far — each stage reveals a larger top-K batch (`top_k`), so the table fills
    best-first while the badge evokes the progressive-doubling DP search.
    """

    stage: int
    title: str
    input_bins: tuple[Bin, ...]
    rows: tuple[ComboRow, ...]
    top_k: int
    n_total: int  # total consecutive groupings the search ranks
    metric: str = "—"
    callout: str = ""


@dataclass(frozen=True)
class HeroFrame:
    """One stage of the README hero animation — a feature strip (Discretizers'
    idiom) stacked over the ranked-groupings table (Carvers' idiom).

    `strip` is rendered with the regular single-strip helpers at the regular
    single-strip coordinates; `rows` fill the table below it. `winner_cuts`
    are normalized x positions where the winning grouping cuts the strip
    (gold dashed verticals, shown when the winner is selected).
    """

    stage: int
    title: str
    strip: Frame
    rows: tuple[ComboRow, ...]
    top_k: int
    n_total: int
    metric: str = "—"
    callout: str = ""
    winner_cuts: tuple[float, ...] = ()


def build_animation(
    feature: FeatureType,
    target: TargetType,
    stop_after_stage: int,
) -> list[Frame]:
    """Return the truncated frame sequence for one example variant.

    The engine produces all 6 stage-frames (0..5) and slices to
    `stop_after_stage` — guarantees the discretizer animation is a strict
    visual prefix of the matching carver animation.
    """
    if not 0 <= stop_after_stage <= 4:
        raise ValueError(f"stop_after_stage must be in [0, 4], got {stop_after_stage}")

    if (feature, target) == ("continuous", "binary"):
        from ._data import continuous_binary_frames

        frames = continuous_binary_frames()
    elif (feature, target) == ("categorical", "binary"):
        from ._data import categorical_binary_frames

        frames = categorical_binary_frames()
    elif (feature, target) == ("ordinal", "binary"):
        from ._data import ordinal_binary_frames

        frames = ordinal_binary_frames()
    elif (feature, target) == ("quantitative", "binary"):
        from ._data import quantitative_binary_frames

        frames = quantitative_binary_frames()
    elif (feature, target) == ("qualitative", "binary"):
        from ._data import qualitative_binary_frames

        frames = qualitative_binary_frames()
    elif (feature, target) == ("combinations", "binary"):
        from ._data import combinations_binary_frames

        frames = combinations_binary_frames()
    elif (feature, target) == ("readme", "binary"):
        from ._data import readme_binary_frames

        frames = readme_binary_frames()
    else:
        raise NotImplementedError(f"variant ({feature!r}, {target!r}) not yet implemented")

    return frames[: stop_after_stage + 1]
