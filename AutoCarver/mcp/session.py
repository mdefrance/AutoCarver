"""Stateful AutoCarver session driving the qualify -> carve workflow.

A :class:`CarverSession` holds one loaded dataset, an optional target, and a mutable draft of
per-column feature specs. Its methods are the units the MCP server exposes as tools; they all
return plain JSON-serialisable structures so they can be wrapped one-to-one by ``server.py``.

The draft is a ``{column: {"type": ..., ...}}`` mapping using the same schema as the LLM
qualifier, so :func:`specs_to_features_kwargs` builds the actual :class:`Features` from it.

Note for AI assistants: this class is an implementation detail, not the entry point. To use
AutoCarver from an MCP client, install ``autocarver[mcp]`` and run the server — see the setup
guide in :mod:`AutoCarver.mcp` (``AutoCarver/mcp/__init__.py``).
"""

from pathlib import Path

import pandas as pd

from AutoCarver.carvers import BinaryCarver, ContinuousCarver, MulticlassCarver
from AutoCarver.carvers.utils.base_carver import BaseCarver
from AutoCarver.features import Features, specs_to_features_kwargs
from AutoCarver.mcp import inspection

_CARVERS = {"binary": BinaryCarver, "continuous": ContinuousCarver, "multiclass": MulticlassCarver}


class CarverSession:
    """Holds the dataset, target and feature draft for one qualify-and-carve workflow."""

    def __init__(self) -> None:
        self.X: pd.DataFrame | None = None
        self.target: str | None = None
        self.draft: dict[str, dict] = {}
        self.carver: BaseCarver | None = None

    # ------------------------------------------------------------------ data loading

    def load_dataset(self, path: str, target: str | None = None) -> dict:
        """Loads a ``.csv`` or ``.parquet`` file as the session dataset.

        ``target``, when given, names the column carved against; it is excluded from feature
        suggestions. Resets any previous draft.
        """
        file = Path(path)
        if file.suffix == ".csv":
            X = pd.read_csv(file)
        elif file.suffix in (".parquet", ".pq"):
            X = pd.read_parquet(file)
        else:
            raise ValueError(f"[session] unsupported file type {file.suffix!r}; use .csv or .parquet.")

        if target is not None and target not in X.columns:
            raise ValueError(f"[session] target {target!r} not found in columns {list(X.columns)}.")

        self.X = X
        self.target = target
        self.draft = {}
        self.carver = None
        return {"rows": len(X), "columns": [str(c) for c in X.columns], "target": target}

    # ------------------------------------------------------------------ inspection (read-only)

    def list_columns(self) -> list[dict]:
        """Per-column summary (dtype, cardinality, missingness, suggested kind)."""
        return inspection.profile_dataframe(self._frame())

    def profile_column(self, column: str, top_n: int = 20) -> dict:
        """Detailed profile of a single column."""
        return inspection.profile_column(self._frame(), column, top_n=top_n)

    def feature_distribution(self, column: str, min_freq: float | None = None, top_n: int = 50) -> dict:
        """Modality distribution of a column, against the loaded target when one is set."""
        return inspection.feature_distribution(self._frame(), column, self._y(), min_freq=min_freq, top_n=top_n)

    def validate_nesting(self, child: str, parents: list[str]) -> dict:
        """Checks that ``child`` rolls cleanly into ``parents`` (many-to-one hierarchy)."""
        return inspection.validate_nesting(self._frame(), child, parents)

    def datetime_reference_candidates(self) -> list[dict]:
        """Summarises datetime columns (span + coverage) to help pick a reference."""
        return inspection.datetime_reference_candidates(self._frame())

    # ------------------------------------------------------------------ drafting features

    def suggest_features(self) -> dict:
        """Fills the draft with dtype-based suggestions (the deterministic first pass).

        Mirrors :meth:`Features.from_dataframe` (including datetime reference resolution) but
        skips the target column. Overwrites the current draft.
        """
        frame = self._frame()
        features = Features.from_dataframe(frame.drop(columns=[self.target]) if self.target else frame)
        self.draft = {feature.name: _feature_to_spec(feature) for feature in features}
        return self.preview_features()

    def set_feature(
        self,
        column: str,
        kind: str,
        values: list[str] | None = None,
        reference: str | None = None,
        parents: list[str] | None = None,
    ) -> dict:
        """Sets or overrides one column's spec in the draft.

        ``kind`` is one of ``numerical``/``categorical``/``ordinal``/``datetime``/``nested``/``ignore``.
        ``ordinal`` needs ``values`` (ordered), ``datetime`` needs ``reference`` (column or date
        literal), ``nested`` needs ``parents``.
        """
        frame = self._frame()
        if column not in frame.columns:
            raise ValueError(f"[session] column {column!r} not found.")
        spec: dict = {"type": kind}
        if kind == "ordinal":
            if not values:
                raise ValueError("[session] ordinal requires 'values' (ordered).")
            spec["values"] = values
        elif kind == "datetime":
            if not reference:
                raise ValueError("[session] datetime requires 'reference' (column name or date literal).")
            spec["reference"] = reference
        elif kind == "nested":
            if not parents:
                raise ValueError("[session] nested requires 'parents'.")
            spec["parents"] = parents
        elif kind not in ("numerical", "categorical", "ignore"):
            raise ValueError(f"[session] unknown kind {kind!r}.")
        self.draft[column] = spec
        return self.preview_features()

    def drop_feature(self, column: str) -> dict:
        """Removes a column from the draft (it will not become a feature)."""
        self.draft.pop(column, None)
        return self.preview_features()

    def preview_features(self) -> dict:
        """Returns the current draft as ``{column: spec}``."""
        return dict(self.draft)

    # ------------------------------------------------------------------ carving

    def run_carver(self, task: str = "auto", min_freq: float = 0.05, max_n_mod: int = 5) -> dict:
        """Builds :class:`Features` from the draft and carves them against the target.

        ``task`` is ``binary``/``continuous``/``multiclass`` or ``auto`` (inferred from the
        target). Returns the carving summary, the carved content per feature, and any features
        the carver dropped as non-viable.
        """
        if self.target is None:
            raise ValueError("[session] no target set; reload the dataset with a target to carve.")
        if not self.draft:
            raise ValueError("[session] draft is empty; call suggest_features or set_feature first.")

        y = self._frame()[self.target]
        features = Features(**specs_to_features_kwargs(self.draft))
        resolved = self._resolve_task(task, y)
        carver = _CARVERS[resolved](features=features, min_freq=min_freq, max_n_mod=max_n_mod)
        carver.fit(self._frame(), y)
        self.carver = carver

        summary = carver.features.summary
        return {
            "task": resolved,
            "kept_features": [feature.version for feature in carver.features],
            "dropped_features": [str(feature) for feature in carver.dropped_features],
            "content": {feature.version: _jsonable(feature.content) for feature in carver.features},
            "summary": summary.reset_index().to_dict(orient="records") if not summary.empty else [],
        }

    def save_carver(self, path: str) -> dict:
        """Saves the fitted carver (carved features included) to a ``.json`` file.

        The carver serialises its :class:`Features` inside the same file, so the saved artifact
        restores both via :meth:`BaseCarver.load`. Requires a prior :meth:`run_carver`.
        """
        if self.carver is None:
            raise ValueError("[session] no fitted carver; call run_carver first.")
        self.carver.save(path)
        return {"saved": path, "features": [feature.version for feature in self.carver.features]}

    def _resolve_task(self, task: str, y: pd.Series) -> str:
        """Resolves ``auto`` to a concrete carver from the target's values."""
        if task in _CARVERS:
            return task
        if task != "auto":
            raise ValueError(f"[session] unknown task {task!r}; use auto/binary/continuous/multiclass.")
        uniques = y.dropna().unique()
        if len(uniques) == 2:
            return "binary"
        if pd.api.types.is_float_dtype(y) and len(uniques) > 10:
            return "continuous"
        return "multiclass"

    # ------------------------------------------------------------------ internals

    def _frame(self) -> pd.DataFrame:
        if self.X is None:
            raise ValueError("[session] no dataset loaded; call load_dataset first.")
        return self.X

    def _y(self) -> pd.Series | None:
        if self.target is None:
            return None
        return self._frame()[self.target]


def _feature_to_spec(feature) -> dict:
    """Serialises a built feature back to a draft spec (inverse of :func:`specs_to_features_kwargs`)."""
    if feature.is_ordinal:
        return {"type": "ordinal", "values": list(feature.raw_order)}
    if feature.is_nested:
        return {"type": "nested", "parents": list(feature.parents)}
    if feature.is_datetime:
        return {"type": "datetime", "reference": feature.reference_date}
    if feature.is_quantitative:
        return {"type": "numerical"}
    return {"type": "categorical"}


def _jsonable(content: dict) -> dict:
    """Stringifies a carved ``content`` mapping (leader -> members) for JSON output."""
    return {str(leader): [str(member) for member in members] for leader, members in content.items()}
