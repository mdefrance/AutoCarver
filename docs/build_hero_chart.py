"""Generate the AutoCarver hero chart (raw vs carved ``age`` on the Adult census data).

Two-panel figure: left is the raw ``age`` feature, a noisy per-value share of
incomes above 50K across dozens of distinct ages; right is the same feature
after ``BinaryCarver`` with Cramér's V (``CramervCombinations``), a handful of
auditable buckets that keep the rise-then-fall shape of income with age, checked
on a dev sample. Used as the README and docs hero image and the social card.

Cramér's V is used because the default Tschuprow's T keeps only 2 buckets here
(young vs the rest): its per-group penalty favours one dominant split.

Writes to ``docs/source/_static/``:

  hero_chart_light.svg   light-mode SVG (transparent bg, dark ink)
  hero_chart_dark.svg    dark-mode SVG (transparent bg, light ink)
  hero_chart.png         1200x630 social-card PNG (light mode, opaque bg)

Usage::

    uv run python docs/build_hero_chart.py [--data path/to/adult.csv]

Without ``--data``, loads the Adult data with ``sklearn.datasets.fetch_openml``
("adult", version 2), which downloads once and then reads scikit-learn's cache.
"""

import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from AutoCarver import BinaryCarver, Features
from AutoCarver.combinations import CramervCombinations

STATIC = Path(__file__).resolve().parent / "source" / "_static"
TARGET = "income_above_50k"
FEATURE = "age"

# same palette as docs/build_logo.py; GOLD is reserved for the "decision" accent
NAVY = "#2E3A59"
NAVY_DARKMODE = "#E2E8F0"
BLUE = "#5B8FF9"  # carved bars on dark backgrounds — NAVY is invisible there
GOLD = "#F6BD16"
RAW_BAR = "#9CA6B8"  # light neutral — deliberately less confident than the carved bars

LIGHT = {"ink": NAVY, "carved_bar": NAVY}
DARK = {"ink": NAVY_DARKMODE, "carved_bar": BLUE}

N_RAW_BINS = 30
FIG_W_IN, FIG_H_IN = 12, 5
SOCIAL_W, SOCIAL_H = 1200, 630


def load_data(data_path: str | None) -> pd.DataFrame:
    """Reads the Adult data from ``--data``, else through ``fetch_openml`` (cached by scikit-learn)."""
    data = pd.read_csv(data_path) if data_path is not None else fetch_openml("adult", version=2, as_frame=True).frame
    data[TARGET] = (data["class"] == ">50K").astype(int)
    return data


def fit_carver(data: pd.DataFrame) -> BinaryCarver:
    train, dev = train_test_split(data, test_size=0.33, random_state=42, stratify=data[TARGET])
    carver = BinaryCarver(
        features=Features(numericals=[FEATURE]),
        combination_evaluator=CramervCombinations(),
        config={"ordinal_encoding": False},
    )
    carver.fit(train, train[TARGET], X_dev=dev, y_dev=dev[TARGET])
    return carver, train


def raw_age_stats(train: pd.DataFrame) -> dict:
    """Per-bin count + share of incomes above 50K for a ~30-bin raw age histogram."""
    ages, targets = train[FEATURE].to_numpy(), train[TARGET].to_numpy()

    counts, edges = np.histogram(ages, bins=N_RAW_BINS)
    bin_idx = np.clip(np.digitize(ages, edges[1:-1]), 0, N_RAW_BINS - 1)
    rate = np.full(N_RAW_BINS, np.nan)
    for b in range(N_RAW_BINS):
        in_bin = bin_idx == b
        if in_bin.any():
            rate[b] = targets[in_bin].mean()

    return {
        "edges": edges,
        "counts": counts,
        "centers": (edges[:-1] + edges[1:]) / 2,
        "rate": rate,
    }


def carved_age_stats(carver: BinaryCarver) -> dict:
    """Per-bucket label, frequency, and share of incomes above 50K, in carving order."""
    labels = carver.features(FEATURE).labels
    summary = carver.summary.reset_index()
    summary = summary[summary["feature"] == str(carver.features(FEATURE))].set_index("label").loc[labels]
    return {
        "labels": _age_ranges(list(carver.features(FEATURE).values)),
        "frequency": summary["frequency"].to_numpy(),
        "rate": summary["target_mean"].to_numpy(),
    }


def _age_ranges(upper_bounds: list[float]) -> list[str]:
    """Readable whole-year ranges from the carved upper bounds, e.g. [27, 31, inf] -> ≤ 27, 28–31, 32+."""
    labels, lower = [], None
    for upper in upper_bounds:
        if lower is None:
            labels.append(f"≤ {int(upper)}")
        elif upper == float("inf"):
            labels.append(f"{int(lower) + 1}+")
        else:
            labels.append(f"{int(lower) + 1}–{int(upper)}")
        lower = upper
    return labels


def _wrap(label: str) -> str:
    return "\n".join(textwrap.wrap(str(label), width=13)) or str(label)


def draw(raw: dict, carved: dict, palette: dict, *, figsize: tuple[float, float]) -> plt.Figure:
    ink = palette["ink"]
    rate_max = np.ceil(max(np.nanmax(raw["rate"]), carved["rate"].max()) * 10) / 10

    fig, (ax_raw, ax_carved) = plt.subplots(1, 2, figsize=figsize)

    # --- left: raw ---
    ax_raw.bar(raw["centers"], raw["counts"], width=np.diff(raw["edges"]), color=RAW_BAR, edgecolor=ink, linewidth=0.3)
    ax_raw.set_title("Raw age: noisy target signal across every distinct value", color=ink, fontsize=11)
    ax_raw.set_xlabel("Age", color=ink)
    ax_raw.set_ylabel("People", color=ink)
    rate_ax_raw = ax_raw.twinx()
    rate_ax_raw.plot(raw["centers"], raw["rate"], color=GOLD, linewidth=1, marker="o", markersize=3)
    rate_ax_raw.set_ylim(0, rate_max)
    rate_ax_raw.set_ylabel("Income above 50K", color=ink)

    # --- right: carved ---
    x = np.arange(len(carved["labels"]))
    ax_carved.bar(x, carved["frequency"], color=palette["carved_bar"], edgecolor=ink, linewidth=0.3)
    ax_carved.set_title(
        f"Carved with Cramér's V: {len(carved['labels'])} buckets, income rises then falls", color=ink, fontsize=11
    )
    ax_carved.set_xticks(x)
    ax_carved.set_xticklabels([_wrap(label) for label in carved["labels"]], fontsize=7.5, color=ink)
    ax_carved.set_ylabel("Frequency", color=ink)
    rate_ax_carved = ax_carved.twinx()
    rate_ax_carved.plot(x, carved["rate"], color=GOLD, linewidth=0, marker="o", markersize=6, zorder=3)
    rate_ax_carved.set_ylim(0, rate_max)
    rate_ax_carved.set_ylabel("Income above 50K", color=ink)

    for ax in (ax_raw, ax_carved):
        ax.tick_params(colors=ink, labelsize=8)
        ax.grid(axis="y", color=ink, alpha=0.12, linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_color(ink)
            spine.set_alpha(0.4)
    for rate_ax in (rate_ax_raw, rate_ax_carved):
        rate_ax.tick_params(colors=ink, labelsize=8)
        for spine in rate_ax.spines.values():
            spine.set_color(ink)
            spine.set_alpha(0.4)

    fig.tight_layout()
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=None, help="local Adult CSV path (default: fetch_openml, cached)")
    args = parser.parse_args()

    plt.rcParams["svg.hashsalt"] = "autocarver"

    data = load_data(args.data)
    carver, train = fit_carver(data)
    raw = raw_age_stats(train)
    carved = carved_age_stats(carver)

    STATIC.mkdir(parents=True, exist_ok=True)
    for name, palette in (("light", LIGHT), ("dark", DARK)):
        fig = draw(raw, carved, palette, figsize=(FIG_W_IN, FIG_H_IN))
        out = STATIC / f"hero_chart_{name}.svg"
        fig.savefig(out, format="svg", transparent=True, metadata={"Date": None})
        plt.close(fig)
        print(f"wrote {out.name}")

    fig = draw(raw, carved, LIGHT, figsize=(SOCIAL_W / 100, SOCIAL_H / 100))
    fig.patch.set_facecolor("white")
    out = STATIC / "hero_chart.png"
    fig.savefig(out, format="png", dpi=100, transparent=False, facecolor="white", metadata={"Software": None})
    plt.close(fig)
    print(f"wrote {out.name} ({SOCIAL_W}x{SOCIAL_H})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
