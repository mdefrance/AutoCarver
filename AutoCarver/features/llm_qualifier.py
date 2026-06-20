"""Provider-agnostic helpers to qualify a DataFrame's columns into :class:`Features` via an LLM.

These helpers never import any provider SDK. The caller supplies ``llm_fn``, a callable taking
the prompt string and returning the model's raw text answer, so any backend can be plugged in.

Anthropic example (latest model id ``claude-opus-4-8``)::

    from anthropic import Anthropic

    client = Anthropic()

    def llm_fn(prompt: str) -> str:
        msg = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    features = qualify_with_llm(X, llm_fn)

OpenAI example::

    from openai import OpenAI

    client = OpenAI()

    def llm_fn(prompt: str) -> str:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content

    features = qualify_with_llm(X, llm_fn)
"""

import json
from collections.abc import Callable

import pandas as pd

from AutoCarver.features.features import Features, FeaturesConfig

# the JSON contract the model must follow, embedded in the prompt and used to parse the answer
_SCHEMA_INSTRUCTIONS = """\
Return ONLY a JSON object (no prose, no markdown fences) mapping every column name to an object
describing its feature type. Each value must have a "type" field, one of:

- "numerical": a quantitative column.
- "categorical": an unordered qualitative column.
- "ordinal": an ordered qualitative column. Add "values": the full list of categories from
  smallest/lowest to largest/highest (strings).
- "datetime": a date/time column. Add "reference": either the name of another datetime column
  to measure elapsed time against, or a fixed date literal like "2020-01-01".
- "nested": a fine-grained qualitative column that rolls up into coarser columns. Add "parents":
  the list of coarser-ward parent column names, from nearest to farthest.
- "ignore": a column that should not become a feature (ids, free text, leakage, etc.).

Example:
{"age": {"type": "numerical"},
 "city": {"type": "categorical"},
 "grade": {"type": "ordinal", "values": ["low", "medium", "high"]},
 "signed_at": {"type": "datetime", "reference": "observed_at"},
 "product": {"type": "nested", "parents": ["category", "division"]},
 "user_id": {"type": "ignore"}}
"""


def build_qualification_prompt(X: pd.DataFrame, *, sample_size: int = 20) -> str:
    """Builds the qualification prompt describing every column of ``X`` for the LLM.

    Each column is summarised by its name, dtype, number of unique values and a small sample of
    values so the model can infer its feature type and any ordering / hierarchy.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame whose columns are described.
    sample_size : int, optional
        Maximum number of sample values shown per column, by default ``20``.
    """
    lines = ["You are qualifying the columns of a tabular dataset for the AutoCarver library.", ""]
    for col in X.columns:
        series = X[col]
        sample = series.dropna().unique()[:sample_size]
        sample_repr = ", ".join(map(str, sample))
        lines.append(f"- {col!r} (dtype={series.dtype}, n_unique={series.nunique()}): {sample_repr}")
    lines += ["", _SCHEMA_INSTRUCTIONS]
    return "\n".join(lines)


def _require(spec: dict, key: str, col: str, kind: str):
    """Returns ``spec[key]`` or raises a clear error naming the column and its kind."""
    if key not in spec:
        raise ValueError(f"[qualify] {kind} column {col!r} is missing its {key!r}.")
    return spec[key]


def specs_to_features_kwargs(mapping: dict) -> dict:
    """Routes a ``{column: {"type": ..., ...}}`` mapping into :class:`Features` kwargs.

    Returns a dict with the ``categoricals``, ``numericals``, ``ordinals``, ``datetimes`` and
    ``nested`` keys expected by ``Features(**kwargs)``. ``ignore`` columns are dropped. Shared
    by the LLM qualifier and the MCP session so the type-routing has a single source of truth.

    Raises
    ------
    ValueError
        If a column declares an unknown / incomplete type.
    """
    categoricals: list[str] = []
    numericals: list[str] = []
    ordinals: dict[str, list[str]] = {}
    datetimes: list[tuple[str, str]] = []
    nested: dict[str, list[str]] = {}

    for col, spec in mapping.items():
        kind = spec.get("type")
        if kind == "numerical":
            numericals.append(col)
        elif kind == "categorical":
            categoricals.append(col)
        elif kind == "ordinal":
            ordinals[col] = [str(value) for value in _require(spec, "values", col, kind)]
        elif kind == "datetime":
            datetimes.append((col, str(_require(spec, "reference", col, kind))))
        elif kind == "nested":
            nested[col] = [str(parent) for parent in _require(spec, "parents", col, kind)]
        elif kind != "ignore":
            raise ValueError(f"[qualify] column {col!r} has unknown type {kind!r}.")

    return {
        "categoricals": categoricals,
        "numericals": numericals,
        "ordinals": ordinals,
        "datetimes": datetimes,
        "nested": nested,
    }


def parse_qualification_response(response: str) -> dict:
    """Parses the LLM's JSON answer into keyword arguments for :class:`Features`.

    Extracts the JSON object from ``response`` and routes it through
    :func:`specs_to_features_kwargs`.

    Raises
    ------
    ValueError
        If no JSON object can be parsed, or a column declares an unknown / incomplete type.
    """
    start, end = response.find("{"), response.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"[qualify] No JSON object found in LLM response: {response!r}")
    try:
        mapping = json.loads(response[start : end + 1])
    except json.JSONDecodeError as error:
        raise ValueError(f"[qualify] Could not parse JSON from LLM response: {error}") from error
    return specs_to_features_kwargs(mapping)


def qualify_with_llm(
    X: pd.DataFrame,
    llm_fn: Callable[[str], str],
    *,
    sample_size: int = 20,
    config: FeaturesConfig | None = None,
) -> Features:
    """Builds a :class:`Features` by asking ``llm_fn`` to qualify every column of ``X``.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame whose columns are qualified.
    llm_fn : Callable[[str], str]
        Callable taking the prompt and returning the model's raw text answer.
    sample_size : int, optional
        Maximum number of sample values shown per column, by default ``20``.
    config : FeaturesConfig, optional
        Collection-level config propagated to each feature, by default ``None``.
    """
    prompt = build_qualification_prompt(X, sample_size=sample_size)
    kwargs = parse_qualification_response(llm_fn(prompt))
    return Features(config=config, **kwargs)
