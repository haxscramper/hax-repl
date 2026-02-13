from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel


class PluginConfig(BaseModel):
    name: str


@runtime_checkable
class Plugin(Protocol):
    def plugin_name(self) -> str: ...


@dataclass(frozen=True)
class LoadedPlugin:
    group: str
    name: str
    plugin: Any


def load_plugins_or_fail(entrypoint_group: str) -> list[LoadedPlugin]:
    loaded: list[LoadedPlugin] = []
    for ep in importlib.metadata.entry_points(group=entrypoint_group):
        plugin_obj = ep.load()
        loaded.append(LoadedPlugin(group=entrypoint_group, name=ep.name, plugin=plugin_obj))
    return loaded
