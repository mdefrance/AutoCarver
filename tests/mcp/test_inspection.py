"""Tests for the read-only MCP inspection helpers."""

import pandas as pd
from pytest import raises

from AutoCarver.mcp import inspection


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cat": ["a", "a", "b", "c", "a"],
            "num": [1.0, 2.0, 3.0, 4.0, 5.0],
            "child": ["A1", "A2", "B1", "B2", "A1"],
            "parent": ["A", "A", "B", "B", "A"],
            "dt": pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01", None]),
        }
    )


def test_profile_dataframe_suggests_kinds():
    rows = {r["column"]: r for r in inspection.profile_dataframe(_frame())}
    assert rows["cat"]["suggested_kind"] == "categorical"
    assert rows["num"]["suggested_kind"] == "numerical"
    assert rows["dt"]["suggested_kind"] == "datetime"
    assert rows["dt"]["n_missing"] == 1


def test_profile_column_numeric_has_quantiles():
    profile = inspection.profile_column(_frame(), "num")
    assert profile["suggested_kind"] == "numerical"
    assert profile["quantiles"]["0.5"] == 3.0


def test_profile_column_qualitative_has_top_values():
    profile = inspection.profile_column(_frame(), "cat")
    assert profile["top_values"][0] == {"value": "a", "count": 3}


def test_profile_column_unknown_raises():
    with raises(ValueError, match="not found"):
        inspection.profile_column(_frame(), "nope")


def test_feature_distribution_reports_count_and_target_rate():
    y = pd.Series([0, 1, 0, 1, 1])
    dist = inspection.feature_distribution(_frame(), "cat", y)
    by_value = {m["value"]: m for m in dist["modalities"]}
    assert by_value["a"]["count"] == 3
    assert by_value["a"]["frequency"] == 0.6
    assert "target_rate" in by_value["a"]


def test_feature_distribution_flags_rare_modalities():
    # Wilson upper bound is conservative, so use a large sample with an unambiguously rare
    # modality: 'rare' at ~2% sits well below min_freq=0.5, 'common' does not.
    X = pd.DataFrame({"cat": ["common"] * 196 + ["rare"] * 4})
    dist = inspection.feature_distribution(X, "cat", min_freq=0.5)
    by_value = {m["value"]: m for m in dist["modalities"]}
    assert by_value["rare"]["rare"] is True
    assert by_value["common"]["rare"] is False


def test_validate_nesting_valid():
    # each child maps to exactly one parent
    X = pd.DataFrame({"child": ["A1", "A2", "B1"], "parent": ["A", "A", "B"]})
    result = inspection.validate_nesting(X, "child", ["parent"])
    assert result["valid"] is True
    assert result["violations"] == []


def test_validate_nesting_detects_violation():
    # A1 sits under both A and B -> not a clean hierarchy
    X = pd.DataFrame({"child": ["A1", "A1", "B1"], "parent": ["A", "B", "B"]})
    result = inspection.validate_nesting(X, "child", ["parent"])
    assert result["valid"] is False
    assert result["violations"][0]["value"] == "A1"
    assert result["violations"][0]["maps_to"] == ["A", "B"]


def test_datetime_reference_candidates():
    candidates = inspection.datetime_reference_candidates(_frame())
    assert len(candidates) == 1
    assert candidates[0]["column"] == "dt"
    assert candidates[0]["coverage_pct"] == 0.8
