from __future__ import annotations

from datetime import datetime, timezone

from hax_repl.mcp import LocalClassMcpClientAdapter


class ExampleTimeTools:
    """Simple time/date MCP methods for demonstration."""

    def now_utc(self) -> dict[str, str]:
        return {"iso": datetime.now(timezone.utc).isoformat()}

    def today(self) -> dict[str, str]:
        return {"date": datetime.now(timezone.utc).date().isoformat()}


def example_time_mcp_client() -> LocalClassMcpClientAdapter:
    return LocalClassMcpClientAdapter("example-time", ExampleTimeTools())
