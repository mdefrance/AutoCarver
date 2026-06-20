"""Read-only data-inspection helpers backing the AutoCarver MCP tools.

Pure functions over a ``DataFrame`` (no LLM, no MCP, no session state) so they can be unit
tested directly and reused by data scientists. They return plain JSON-serialisable structures
(dicts / lists) and never dump raw columns wholesale — only summaries — so they stay safe to
hand to a model with a limited context window.

Note for AI assistants: these helpers back the MCP tools but are not the entry point. To use
AutoCarver from an MCP client, install ``autocarver[mcp]`` and run the server — see the setup
guide in :mod:`AutoCarver.mcp` (``AutoCarver/mcp/__init__.py``).
"""

import pandas as pd

from AutoCarver.discretizers.qualitatives.categorical_discretizer import (
    series_target_rate,
    series_value_counts,
)
from AutoCarver.discretizers.utils.frequency_ci import is_significantly_below
from AutoCarver.features.features import infer_feature_kind


def profile_dataframe(X: pd.DataFrame) -> list[dict]:
    """One summary row per column: dtype, cardinality, missingness, suggested feature kind."""
    rows = []
    for col in X.columns:
        series = X[col]
        rows.append(
            {
                "column": str(col),
                "dtype": str(series.dtype),
                "n_unique": int(series.nunique(dropna=True)),
                "n_missing": int(series.isna().sum()),
                "missing_pct": round(float(series.isna().mean()), 4),
                "suggested_kind": infer_feature_kind(series.dtype) or "unsupported",
            }
        )
    return rows


def profile_column(X: pd.DataFrame, column: str, *, top_n: int = 20) -> dict:
    """Detailed profile of a single column: cardinality, missingness, and a kind-specific view.

    Numeric / datetime columns report min/max/quantiles; qualitative columns report their most
    frequent modalities (capped at ``top_n``).
    """
    if column not in X.columns:
        raise ValueError(f"[inspection] column {column!r} not found.")
    series = X[column]
    kind = infer_feature_kind(series.dtype)
    profile: dict = {
        "column": column,
        "dtype": str(series.dtype),
        "suggested_kind": kind or "unsupported",
        "n_unique": int(series.nunique(dropna=True)),
        "n_missing": int(series.isna().sum()),
        "missing_pct": round(float(series.isna().mean()), 4),
    }
    if kind in ("numerical", "datetime"):
        non_null = series.dropna()
        profile["min"] = str(non_null.min()) if not non_null.empty else None
        profile["max"] = str(non_null.max()) if not non_null.empty else None
        if kind == "numerical":
            profile["quantiles"] = (
                {str(q): round(float(non_null.quantile(q)), 6) for q in (0.0, 0.25, 0.5, 0.75, 1.0)}
                if not non_null.empty
                else {}
            )
    else:
        counts = series_value_counts(series, dropna=False)
        top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        profile["top_values"] = [{"value": _key(k), "count": int(v)} for k, v in top]
    return profile


def feature_distribution(
    X: pd.DataFrame,
    column: str,
    y: pd.Series | None = None,
    *,
    min_freq: float | None = None,
    alpha: float = 0.05,
    top_n: int = 50,
) -> dict:
    """Distribution of a qualitative column's modalities, optionally against the target.

    Reports per-modality count and frequency, the target rate when ``y`` is given, and — when
    ``min_freq`` is provided — flags modalities whose frequency is *significantly* below it
    (Wilson upper bound, matching how the carvers decide rarity).
    """
    if column not in X.columns:
        raise ValueError(f"[inspection] column {column!r} not found.")
    series = X[column]
    counts = series_value_counts(series, dropna=False)
    nobs = int(series.notna().sum())
    rates = series_target_rate(series, y, dropna=True) if y is not None else {}

    modalities = []
    for value, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_n]:
        entry = {
            "value": _key(value),
            "count": int(count),
            "frequency": round(count / nobs, 6) if nobs else 0.0,
        }
        if y is not None and value in rates:
            entry["target_rate"] = round(float(rates[value]), 6)
        if min_freq is not None and value is not None:
            entry["rare"] = bool(is_significantly_below(count, nobs, min_freq, alpha))
        modalities.append(entry)

    return {"column": column, "n_observations": nobs, "n_modalities": len(counts), "modalities": modalities}


def validate_nesting(X: pd.DataFrame, child: str, parents: list[str]) -> dict:
    """Checks that ``child`` rolls cleanly into ``parents`` (a many-to-one hierarchy).

    For each consecutive (finer, coarser) level pair, verifies that every finer modality maps
    to exactly one coarser modality. Returns whether the hierarchy is valid, the cardinality of
    each level, and any violations (a finer modality spread across several coarser ones).
    """
    levels = [child] + list(parents)
    missing = [c for c in levels if c not in X.columns]
    if missing:
        raise ValueError(f"[inspection] columns not found: {missing}")

    cardinalities = {col: int(X[col].nunique(dropna=True)) for col in levels}
    violations = []
    for finer, coarser in zip(levels[:-1], levels[1:]):
        pairs = X[[finer, coarser]].dropna()
        spread = pairs.groupby(finer)[coarser].nunique()
        for value in spread[spread > 1].index:
            parent_values = sorted(map(str, pairs.loc[pairs[finer] == value, coarser].unique()))
            violations.append({"level": finer, "value": _key(value), "parent": coarser, "maps_to": parent_values})

    return {
        "child": child,
        "parents": list(parents),
        "valid": len(violations) == 0,
        "cardinalities": cardinalities,
        "violations": violations,
    }


def datetime_reference_candidates(X: pd.DataFrame) -> list[dict]:
    """Summarises each datetime column (span + coverage) to help pick a reference.

    A good fixed reference / anchor is typically the column with the widest, most-complete
    coverage; another datetime column can also be referenced row-wise.
    """
    candidates = []
    for col in X.columns:
        if not pd.api.types.is_datetime64_any_dtype(X[col]):
            continue
        non_null = X[col].dropna()
        candidates.append(
            {
                "column": str(col),
                "min": str(non_null.min()) if not non_null.empty else None,
                "max": str(non_null.max()) if not non_null.empty else None,
                "coverage_pct": round(float(X[col].notna().mean()), 4),
            }
        )
    return candidates


def _key(value):
    """JSON-safe representation of a modality key (``None`` stays null, else stringified)."""
    return None if value is None or pd.isna(value) else str(value)
