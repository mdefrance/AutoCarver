"""Smoke tests for the FastMCP server wiring."""

import asyncio

from AutoCarver.mcp import create_server

_EXPECTED_TOOLS = {
    "load_dataset",
    "list_columns",
    "profile_column",
    "feature_distribution",
    "validate_nesting",
    "datetime_reference_candidates",
    "suggest_features",
    "set_feature",
    "drop_feature",
    "preview_features",
    "run_carver",
    "save_carver",
}


def test_server_registers_all_tools():
    server = create_server()
    tools = asyncio.run(server.list_tools())
    assert {tool.name for tool in tools} == _EXPECTED_TOOLS


def test_server_tools_share_one_session(tmp_path):
    """Tools mutate a single shared session, so load -> suggest -> preview round-trips."""
    import pandas as pd

    path = tmp_path / "d.csv"
    pd.DataFrame({"cat": ["a", "b", "a"], "num": [1.0, 2.0, 3.0], "target": [0, 1, 0]}).to_csv(path, index=False)

    server = create_server()

    async def run():
        await server.call_tool("load_dataset", {"path": str(path), "target": "target"})
        await server.call_tool("suggest_features", {})
        return await server.call_tool("preview_features", {})

    result = run_and_unwrap(run())
    assert result == {"cat": {"type": "categorical"}, "num": {"type": "numerical"}}


def run_and_unwrap(coro):
    """Runs an async tool call and pulls its structured payload across fastmcp result shapes."""
    result = asyncio.run(coro)
    return getattr(result, "structured_content", None) or getattr(result, "data", result)
