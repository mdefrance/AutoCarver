"""FastMCP server exposing the AutoCarver qualify-and-carve workflow as tools.

Thin transport layer: every tool wraps one :class:`CarverSession` method. All logic lives in
:mod:`AutoCarver.mcp.session` / :mod:`AutoCarver.mcp.inspection`, which are importable and
testable without FastMCP installed.

Run with ``python -m AutoCarver.mcp`` (stdio transport, the default for MCP clients).
"""

try:
    from fastmcp import FastMCP
except ModuleNotFoundError as error:  # pragma: no cover - exercised only without the extra
    raise ModuleNotFoundError(
        "The AutoCarver MCP server requires the optional 'mcp' extra. Install it with:\n\n"
        '    pip install "autocarver[mcp]"   # or: uv add "autocarver[mcp]"\n\n'
        "then point your MCP client at it (see the 'MCP setup' guide in AutoCarver/mcp/__init__.py)."
    ) from error

from AutoCarver.mcp.session import CarverSession


def create_server() -> FastMCP:
    """Builds a FastMCP server bound to a fresh :class:`CarverSession`."""
    server = FastMCP("AutoCarver")
    session = CarverSession()
    _register_inspection_tools(server, session)
    _register_drafting_tools(server, session)
    return server


def _register_inspection_tools(server: FastMCP, session: CarverSession) -> None:
    """Registers the data-loading and read-only inspection tools."""

    @server.tool
    def load_dataset(path: str, target: str | None = None) -> dict:
        """Load a .csv/.parquet file as the working dataset; optionally name the target column."""
        return session.load_dataset(path, target)

    @server.tool
    def list_columns() -> list[dict]:
        """List every column with its dtype, cardinality, missingness and suggested feature kind."""
        return session.list_columns()

    @server.tool
    def profile_column(column: str, top_n: int = 20) -> dict:
        """Profile one column: cardinality, missingness, quantiles (numeric) or top modalities."""
        return session.profile_column(column, top_n)

    @server.tool
    def feature_distribution(column: str, min_freq: float | None = None, top_n: int = 50) -> dict:
        """Show a column's modality distribution, target rate, and rare-modality flags."""
        return session.feature_distribution(column, min_freq, top_n)

    @server.tool
    def validate_nesting(child: str, parents: list[str]) -> dict:
        """Check that a finest column rolls cleanly into coarser parent columns (many-to-one)."""
        return session.validate_nesting(child, parents)

    @server.tool
    def datetime_reference_candidates() -> list[dict]:
        """Summarise datetime columns (span + coverage) to help choose a reference."""
        return session.datetime_reference_candidates()


def _register_drafting_tools(server: FastMCP, session: CarverSession) -> None:
    """Registers the feature-draft and carving tools."""

    @server.tool
    def suggest_features() -> dict:
        """Fill the feature draft with dtype-based suggestions (skips the target)."""
        return session.suggest_features()

    @server.tool
    def set_feature(
        column: str,
        kind: str,
        values: list[str] | None = None,
        reference: str | None = None,
        parents: list[str] | None = None,
    ) -> dict:
        """Set/override a column's feature kind in the draft (numerical/categorical/ordinal/datetime/nested/ignore)."""
        return session.set_feature(column, kind, values, reference, parents)

    @server.tool
    def drop_feature(column: str) -> dict:
        """Remove a column from the feature draft."""
        return session.drop_feature(column)

    @server.tool
    def preview_features() -> dict:
        """Return the current feature draft as {column: spec}."""
        return session.preview_features()

    @server.tool
    def run_carver(task: str = "auto", min_freq: float = 0.05, max_n_mod: int = 5) -> dict:
        """Build Features from the draft and carve them against the target; return the summary."""
        return session.run_carver(task, min_freq, max_n_mod)

    @server.tool
    def save_carver(path: str) -> dict:
        """Save the fitted carver and its carved features to a .json file (run run_carver first)."""
        return session.save_carver(path)


def main() -> None:
    """Entry point: run the AutoCarver MCP server over stdio."""
    create_server().run()


if __name__ == "__main__":
    main()
