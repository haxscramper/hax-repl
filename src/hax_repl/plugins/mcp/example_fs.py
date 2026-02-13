from __future__ import annotations

from pathlib import Path

from hax_repl.mcp import LocalClassMcpClientAdapter
from hax_repl.models import PluginMCPMeta
from hax_repl.plugin_system import PluginDescriptor


class ExampleFilesystemTools:
    """Simple filesystem MCP methods for demonstration."""

    def pwd(self) -> dict[str, str]:
        return {"cwd": str(Path.cwd())}

    def list_dir(self,
                 path: str = ".",
                 max_entries: int = 50) -> dict[str, object]:
        target = Path(path).expanduser()
        entries: list[str] = []
        for child in sorted(target.iterdir(), key=lambda p: p.name):
            suffix = "/" if child.is_dir() else ""
            entries.append(f"{child.name}{suffix}")
            if len(entries) >= max(1, max_entries):
                break
        return {"path": str(target), "entries": entries}


def example_fs_mcp_client() -> LocalClassMcpClientAdapter:
    return LocalClassMcpClientAdapter("example-fs", ExampleFilesystemTools())


def register() -> PluginDescriptor:
    return PluginDescriptor(
        metadata=PluginMCPMeta(
            name="example-fs",
            description="Example filesystem MCP client plugin.",
        ),
        plugin_factory=example_fs_mcp_client,
    )
