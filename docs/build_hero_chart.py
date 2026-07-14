"""Generate the AutoCarver hero chart (raw vs carved ``Age`` on Titanic).

Two-panel figure: left is the raw ``Age`` feature — a noisy per-value
survival rate across dozens of distinct ages; right is the same feature
after ``BinaryCarver`` — a handful of auditable buckets with a clearly
monotonic survival rate. Used as the README hero image below the pipeline
animation, the article's key figure, and the LinkedIn post image.

Note: the quick-start's Titanic CSV has no missing values in any column
(verified — this mirror is already cleaned), so this chart does not depict
a NaN bucket.

Writes to ``docs/source/_static/``:

  hero_chart_light.svg   light-mode SVG (transparent bg, dark ink)
  hero_chart_dark.svg    dark-mode SVG (transparent bg, light ink)
  hero_chart.png         1200x630 social-card PNG (light mode, opaque bg)

Usage::

    uv run python docs/build_hero_chart.py [--data path/to/titanic.csv]

Without ``--data``, downloads the quick-start Titanic CSV and caches it to
``docs/source/_static/.titanic_cache.csv`` (gitignored via ``*.csv``) so
reruns and README rendering never depend on the network.
"""

import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from AutoCarver import BinaryCarver, Features
from AutoCarver.discretizers.utils.base_discretizer import ProcessingConfig

STATIC = Path(__file__).resolve().parent / "source" / "_static"
CACHE = STATIC / ".titanic_cache.csv"
TITANIC_URL = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
TARGET = "Survived"
FEATURE = "Age"

# same palette as docs/build_logo.py; GOLD is reserved for the "decision" accent
NAVY = "#2E3A59"
NAVY_DARKMODE = "#E2E8F0"
GOLD = "#F6BD16"
RAW_BAR = "#9CA6B8"  # light neutral — deliberately less confident than the carved bars

LIGHT = {"ink": NAVY}
DARK = {"ink": NAVY_DARKMODE}

N_RAW_BINS = 30
FIG_W_IN, FIG_H_IN = 12, 5
SOCIAL_W, SOCIAL_H = 1200, 630


def load_data(data_path: str | None) -> pd.DataFrame:
    """Reads Titanic data from ``--data``, else the cache, else the quick-start URL."""
    if data_path is not None:
        return pd.read_csv(data_path)
    if CACHE.exists():
        return pd.read_csv(CACHE)
    data = pd.read_csv(TITANIC_URL)
    STATIC.mkdir(parents=True, exist_ok=True)
    data.to_csv(CACHE, index=False)
    return data


def fit_carver(data: pd.DataFrame) -> BinaryCarver:
    train, dev = train_test_split(data, test_size=0.33, random_state=42, stratify=data[TARGET])
    features = Features(
        categoricals=["Sex"],
        numericals=["Age", "Fare", "Siblings/Spouses Aboard", "Parents/Children Aboard"],
        ordinals={"Pclass": ["1", "2", "3"]},
    )
    carver = BinaryCarver(features=features, config=ProcessingConfig(ordinal_encoding=False))
    carver.fit(train, train[TARGET], X_dev=dev, y_dev=dev[TARGET])
    return carver, train


def raw_age_stats(train: pd.DataFrame) -> dict:
    """Per-bin count + survival rate for a ~30-bin raw Age histogram."""
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
    """Per-bucket label, frequency, and survival rate, in carving order."""
    labels = carver.features(FEATURE).labels
    summary = carver.summary.reset_index()
    summary = summary[summary["feature"] == str(carver.features(FEATURE))].set_index("label").loc[labels]
    return {
        "labels": labels,
        "frequency": summary["frequency"].to_numpy(),
        "rate": summary["target_mean"].to_numpy(),
    }


def _wrap(label: str) -> str:
    return "\n".join(textwrap.wrap(str(label), width=13)) or str(label)


def draw(raw: dict, carved: dict, palette: dict, *, figsize: tuple[float, float]) -> plt.Figure:
    ink = palette["ink"]
    rate_max = np.ceil(max(np.nanmax(raw["rate"]), carved["rate"].max()) * 10) / 10

    fig, (ax_raw, ax_carved) = plt.subplots(1, 2, figsize=figsize)

    # --- left: raw ---
    ax_raw.bar(raw["centers"], raw["counts"], width=np.diff(raw["edges"]), color=RAW_BAR, edgecolor=ink, linewidth=0.3)
    ax_raw.set_title("Raw Age — noisy target signal across every distinct value", color=ink, fontsize=11)
    ax_raw.set_xlabel("Age", color=ink)
    ax_raw.set_ylabel("Passengers", color=ink)
    rate_ax_raw = ax_raw.twinx()
    rate_ax_raw.plot(raw["centers"], raw["rate"], color=GOLD, linewidth=1, marker="o", markersize=3)
    rate_ax_raw.set_ylim(0, rate_max)
    rate_ax_raw.set_ylabel("Survival rate", color=ink)

    # --- right: carved ---
    x = np.arange(len(carved["labels"]))
    ax_carved.bar(x, carved["frequency"], color=NAVY, edgecolor=ink, linewidth=0.3)
    ax_carved.set_title(
        f"Carved — {len(carved['labels'])} auditable buckets, monotonic survival rate", color=ink, fontsize=11
    )
    ax_carved.set_xticks(x)
    ax_carved.set_xticklabels([_wrap(label) for label in carved["labels"]], fontsize=7.5, color=ink)
    ax_carved.set_ylabel("Frequency", color=ink)
    rate_ax_carved = ax_carved.twinx()
    rate_ax_carved.plot(x, carved["rate"], color=GOLD, linewidth=0, marker="o", markersize=6, zorder=3)
    rate_ax_carved.set_ylim(0, rate_max)
    rate_ax_carved.set_ylabel("Survival rate", color=ink)

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
    parser.add_argument("--data", default=None, help="local Titanic CSV path (default: download + cache)")
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
