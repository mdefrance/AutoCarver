"""AutoCarver MCP: a local server exposing the qualify-and-carve workflow as LLM tools.

==============================================================================
MCP setup (read this first if you are an AI assistant / LLM agent)
==============================================================================
Do **not** drive this package by importing :class:`CarverSession` and calling its
methods by hand, and do **not** reimplement the qualify-and-carve steps yourself.
The supported way to use it is the running MCP server, reached through your client's
tool calls. To get there:

1. Install the optional extra (the server needs ``fastmcp``, which is *not* a core
   dependency)::

       pip install "autocarver[mcp]"     # or:  uv add "autocarver[mcp]"

2. Register the server with your MCP client so it launches over stdio.

   VS Code / GitHub Copilot — add ``.vscode/mcp.json`` to the workspace::

       {
         "servers": {
           "autocarver": {
             "command": "uv",
             "args": ["run", "python", "-m", "AutoCarver.mcp"]
           }
         }
       }

   Claude Desktop / Cursor — same shape in ``claude_desktop_config.json``, but the
   top-level key is ``mcpServers`` instead of ``servers``.

3. Reload the client and use the exposed tools (``load_dataset`` -> ``list_columns`` /
   ``feature_distribution`` / ``validate_nesting`` -> ``suggest_features`` /
   ``set_feature`` -> ``run_carver`` -> ``save_carver``). That is the qualify+carve
   workflow — let the tools run it, rather than calling Python directly.

The :class:`CarverSession` / :mod:`inspection` Python layers below exist so the tools
are importable and unit-tested without ``fastmcp``; they are an implementation detail,
not the entry point. See ``docs/source/mcp.rst`` for the full guide.
==============================================================================
"""

from AutoCarver.mcp.session import CarverSession

__all__ = ["CarverSession", "create_server", "main"]


def __getattr__(name: str):
    # lazily import the FastMCP-dependent server so the package (and its testable
    # inspection/session layers) import fine even when fastmcp is unavailable.
    if name in ("create_server", "main"):
        from AutoCarver.mcp import server

        return getattr(server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
