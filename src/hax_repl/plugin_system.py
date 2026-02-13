from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from pydantic import BaseModel, Field, TypeAdapter

from hax_repl.models import PluginMetadata


class PluginPathConfig(BaseModel):
    path: str


class PluginLoaderConfig(BaseModel):
    plugins: list[PluginPathConfig] = Field(default_factory=list)


@runtime_checkable
class PluginDescription(Protocol):

    def get_plugin(self) -> Any:
        ...

    def get_metadata(self) -> PluginMetadata:
        ...


@dataclass(frozen=True)
class LoadedPlugin:
    kind: str
    name: str
    path: Path
    description: PluginDescription
    metadata: PluginMetadata


class PluginDescriptor:

    def __init__(self, metadata: PluginMetadata,
                 plugin_factory: Callable[[], Any] | type[Any]) -> None:
        self._metadata = metadata
        self._plugin_factory = plugin_factory

    def get_plugin(self) -> Any:
        if isinstance(self._plugin_factory, type):
            return self._plugin_factory()
        plugin = self._plugin_factory()
        return instantiate_plugin(plugin)

    def get_metadata(self) -> PluginMetadata:
        return self._metadata


def instantiate_plugin(plugin_obj: Any) -> Any:
    if isinstance(plugin_obj, type):
        return plugin_obj()
    return plugin_obj


def load_plugins_from_config_or_fail(
        config_path: Path,
        interpolation_vars: dict[str, str] | None = None) -> list[LoadedPlugin]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    config = PluginLoaderConfig.model_validate(payload)
    variables = _default_interpolation_vars(config_path)
    if interpolation_vars:
        variables.update(interpolation_vars)

    loaded: list[LoadedPlugin] = []
    for plugin_ref in config.plugins:
        resolved_path = _resolve_plugin_path(plugin_ref.path, variables)
        description = _load_plugin_description_from_file_or_fail(resolved_path)
        metadata = _validate_plugin_metadata(description.get_metadata(), resolved_path)
        loaded.append(
            LoadedPlugin(
                kind=metadata.kind,
                name=metadata.name,
                path=resolved_path,
                description=description,
                metadata=metadata,
            ))
    return loaded


def _load_plugin_description_from_file_or_fail(path: Path) -> PluginDescription:
    if not path.exists():
        raise RuntimeError(f"Plugin file not found: {path}")
    if not path.is_file():
        raise RuntimeError(f"Plugin path must be a file: {path}")

    module_name = _module_name_for_plugin_path(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to create import spec for plugin file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    register = getattr(module, "register", None)
    if not callable(register):
        raise RuntimeError(f"Plugin file must define callable register(): {path}")

    description = register()
    if not hasattr(description, "get_plugin") or not callable(
            getattr(description, "get_plugin")):
        raise RuntimeError(f"register() must return an object with get_plugin(): {path}")
    if not hasattr(description, "get_metadata") or not callable(
            getattr(description, "get_metadata")):
        raise RuntimeError(
            f"register() must return an object with get_metadata(): {path}")
    return description


def _validate_plugin_metadata(metadata: Any, path: Path) -> PluginMetadata:
    adapter = TypeAdapter(PluginMetadata)
    try:
        return adapter.validate_python(metadata)
    except Exception as exc:
        raise RuntimeError(f"Invalid plugin metadata in {path}: {exc}") from exc


def _resolve_plugin_path(raw_path: str, interpolation_vars: dict[str, str]) -> Path:
    try:
        formatted = raw_path.format_map(interpolation_vars)
    except KeyError as exc:
        available = ", ".join(sorted(interpolation_vars.keys()))
        raise RuntimeError(
            f"Unknown interpolation variable '{exc.args[0]}' in plugin path '{raw_path}'. "
            f"Available variables: {available}") from exc
    return Path(formatted).expanduser().resolve()


def _default_interpolation_vars(config_path: Path) -> dict[str, str]:
    repo_root = _find_repo_root(config_path.parent) or config_path.parent.resolve()
    return {
        "repo": str(repo_root),
        "config_dir": str(config_path.parent.resolve()),
        "home": str(Path.home()),
        "cwd": str(Path.cwd()),
    }


def _find_repo_root(start: Path) -> Path | None:
    current = start.resolve()
    if (current / ".git").exists():
        return current
    for parent in current.parents:
        if (parent / ".git").exists():
            return parent
    return None


def _module_name_for_plugin_path(path: Path) -> str:
    digest = hashlib.md5(str(path).encode("utf-8")).hexdigest()
    return f"hax_repl_dynamic_plugin_{digest}"
