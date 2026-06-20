"""Tests for Features.from_dataframe and the LLM qualifier helpers."""

import json

import pandas as pd
from pytest import raises, warns

from AutoCarver.features import (
    Features,
    build_qualification_prompt,
    get_names,
    parse_qualification_response,
    qualify_with_llm,
)


def _mixed_dataframe() -> pd.DataFrame:
    """DataFrame covering every auto-detectable dtype plus an unsupported one."""
    return pd.DataFrame(
        {
            "num_i": [1, 2, 3, 4],
            "num_f": [1.0, 2.5, 3.1, 4.2],
            "cat": ["a", "b", "a", "c"],
            "flag": [True, False, True, False],
            "ord": pd.Categorical(["low", "high", "medium", "low"], categories=["low", "medium", "high"], ordered=True),
            "unord": pd.Categorical(["x", "y", "x", "z"]),
            "dt": pd.to_datetime(["2020-01-01", "2020-06-01", "2021-01-01", "2019-01-01"]),
        }
    )


def test_from_dataframe_maps_each_dtype():
    """Each column lands in its expected typed list; ordered category lifts its order."""
    features = Features.from_dataframe(_mixed_dataframe())

    assert get_names(features.quantitatives) == ["num_i", "num_f", "dt"]
    assert get_names(features.categoricals) == ["cat", "flag", "unord"]
    assert get_names(features.ordinals) == ["ord"]
    assert get_names(features.datetimes) == ["dt"]
    # ordered category's order is lifted as the ordinal raw order
    assert features("ord").raw_order == ["low", "medium", "high"]
    # nested is never auto-detected
    assert get_names(features.nested) == []


def test_from_dataframe_warns_and_skips_unsupported_dtype():
    """An unsupported dtype (here: timedelta64) is skipped with a warning."""
    X = pd.DataFrame({"num": [1, 2], "weird": pd.to_timedelta([1, 2], unit="D")})
    with warns(UserWarning, match="weird"):
        features = Features.from_dataframe(X)
    assert "weird" not in features
    assert get_names(features.quantitatives) == ["num"]


def test_from_dataframe_datetime_reference_uses_most_recent_column():
    """With several datetime columns, the most recent one is the shared reference."""
    X = pd.DataFrame(
        {
            "d_old": pd.to_datetime(["2019-01-01", "2019-06-01"]),
            "d_new": pd.to_datetime(["2023-01-01", "2023-06-01"]),
            "d_mid": pd.to_datetime(["2021-01-01", "2021-06-01"]),
        }
    )
    features = Features.from_dataframe(X)

    # d_new is the most recent -> anchor; others reference it row-wise
    assert features("d_old").reference_date == "d_new"
    assert features("d_mid").reference_date == "d_new"
    # the anchor can't reference itself -> fixed literal (its earliest date)
    assert features("d_new").reference_date == "2023-01-01"


def test_from_dataframe_single_datetime_uses_fixed_reference():
    """A lone datetime column has no other column to reference; uses its earliest date."""
    X = pd.DataFrame({"d": pd.to_datetime(["2020-03-01", "2020-01-15", "2020-12-31"])})
    features = Features.from_dataframe(X)
    assert features("d").reference_date == "2020-01-15"


def test_build_qualification_prompt_lists_every_column():
    """The prompt mentions every column name and its dtype."""
    X = _mixed_dataframe()
    prompt = build_qualification_prompt(X)
    for column in X.columns:
        assert repr(column) in prompt
    assert "JSON" in prompt


def test_parse_qualification_response_routes_each_type():
    """Canned JSON (incl. ordinal order, datetime ref, nested parents) maps to Features kwargs."""
    response = "Here is the result:\n" + json.dumps(
        {
            "age": {"type": "numerical"},
            "city": {"type": "categorical"},
            "grade": {"type": "ordinal", "values": ["low", "medium", "high"]},
            "signed_at": {"type": "datetime", "reference": "observed_at"},
            "product": {"type": "nested", "parents": ["category", "division"]},
            "user_id": {"type": "ignore"},
        }
    )
    kwargs = parse_qualification_response(response)

    assert kwargs["numericals"] == ["age"]
    assert kwargs["categoricals"] == ["city"]
    assert kwargs["ordinals"] == {"grade": ["low", "medium", "high"]}
    assert kwargs["datetimes"] == [("signed_at", "observed_at")]
    assert kwargs["nested"] == {"product": ["category", "division"]}


def test_parse_qualification_response_raises_on_malformed():
    """A response without a JSON object raises a clear error."""
    with raises(ValueError, match="No JSON object"):
        parse_qualification_response("sorry, I cannot help with that")


def test_qualify_with_llm_builds_features_from_fake_llm():
    """qualify_with_llm wires prompt -> llm_fn -> parse -> Features."""
    X = pd.DataFrame({"age": [1, 2], "grade": ["low", "high"], "product": ["a", "b"]})
    canned = json.dumps(
        {
            "age": {"type": "numerical"},
            "grade": {"type": "ordinal", "values": ["low", "high"]},
            "product": {"type": "nested", "parents": ["category"]},
        }
    )
    features = qualify_with_llm(X, lambda prompt: canned)

    assert get_names(features.quantitatives) == ["age"]
    assert features("grade").raw_order == ["low", "high"]
    assert get_names(features.nested) == ["product"]
