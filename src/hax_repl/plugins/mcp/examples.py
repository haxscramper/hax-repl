from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from hax_repl.mcp import LocalClassMcpClientAdapter


class ExampleFilesystemTools:
    """Simple filesystem MCP methods for demonstration."""

    def pwd(self) -> dict[str, str]:
        return {"cwd": str(Path.cwd())}

    def list_dir(self, path: str = ".", max_entries: int = 50) -> dict[str, object]:
        target = Path(path).expanduser()
        entries: list[str] = []
        for child in sorted(target.iterdir(), key=lambda p: p.name):
            suffix = "/" if child.is_dir() else ""
            entries.append(f"{child.name}{suffix}")
            if len(entries) >= max(1, max_entries):
                break
        return {"path": str(target), "entries": entries}


class ExampleTimeTools:
    """Simple time/date MCP methods for demonstration."""

    def now_utc(self) -> dict[str, str]:
        return {"iso": datetime.now(timezone.utc).isoformat()}

    def today(self) -> dict[str, str]:
        return {"date": datetime.now(timezone.utc).date().isoformat()}


def example_fs_mcp_client():
    return LocalClassMcpClientAdapter("example-fs", ExampleFilesystemTools())


def example_time_mcp_client():
    return LocalClassMcpClientAdapter("example-time", ExampleTimeTools())
