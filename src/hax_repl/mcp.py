from __future__ import annotations

import importlib
import inspect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field, create_model

from hax_repl.functions import FunctionSpec


class McpInvokeArgs(BaseModel):
    arguments_json: str = Field(default="{}")


class McpClientDescriptor(BaseModel):
    name: str
    module: str
    class_name: str = Field(alias="class")


class McpClient(Protocol):
    def client_name(self) -> str: ...

    def list_tools(self) -> list[FunctionSpec[BaseModel, Any]]: ...

    def invoke(self, tool_name: str, arguments_json: str) -> str: ...


@dataclass(frozen=True)
class RegisteredMcpClient:
    name: str
    client: McpClient
    function_names: list[str]


class LocalClassMcpClientAdapter:
    def __init__(self, name: str, obj: object) -> None:
        self._name = name
        self._obj = obj
        self._methods: dict[str, Any] = {
            method_name: method
            for method_name, method in inspect.getmembers(obj, predicate=callable)
            if not method_name.startswith("_")
        }
        self._function_name_to_method_name: dict[str, str] = {}

    def client_name(self) -> str:
        return self._name

    def list_tools(self) -> list[FunctionSpec[BaseModel, Any]]:
        tools: list[FunctionSpec[BaseModel, Any]] = []
        for method_name in sorted(self._methods.keys()):
            method = self._methods[method_name]
            doc = inspect.getdoc(method) or f"MCP tool {method_name}"
            function_name = _safe_function_name(self._name, method_name)
            self._function_name_to_method_name[function_name] = method_name
            args_model = create_model(
                f"McpArgs_{self._name}_{method_name}",
                arguments_json=(str, Field(default="{}")),
            )
            tools.append(
                FunctionSpec(
                    name=function_name,
                    description=doc,
                    args_model=args_model,
                    result_model=None,
                    impl=self._make_impl(method_name),
                )
            )
        return tools

    def _make_impl(self, method_name: str):
        def _impl(args_model: BaseModel) -> Any:
            return self.invoke(method_name, args_model.model_dump().get("arguments_json", "{}"))

        return _impl

    def invoke(self, tool_name: str, arguments_json: str) -> str:
        if tool_name not in self._methods:
            raise KeyError(f"Unknown MCP tool: {tool_name}")
        method = self._methods[tool_name]
        parsed = json.loads(arguments_json) if arguments_json.strip() else {}
        if not isinstance(parsed, dict):
            parsed = {"value": parsed}
        result = method(**parsed)
        if isinstance(result, str):
            return json.dumps({"result": result}, ensure_ascii=True)
        if isinstance(result, BaseModel):
            return result.model_dump_json()
        return json.dumps(result, ensure_ascii=True)


class DescriptorMcpLoader:
    def load(self, path: Path) -> LocalClassMcpClientAdapter:
        payload = json.loads(path.read_text(encoding="utf-8"))
        descriptor = McpClientDescriptor.model_validate(payload)
        module = importlib.import_module(descriptor.module)
        cls = getattr(module, descriptor.class_name)
        instance = cls()
        return LocalClassMcpClientAdapter(descriptor.name, instance)


def _safe_function_name(client_name: str, method_name: str) -> str:
    def _normalize(token: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", token)
        normalized = normalized.strip("_")
        return normalized or "x"

    prefix = "mcp"
    composed = f"{prefix}_{_normalize(client_name)}_{_normalize(method_name)}"
    if len(composed) <= 64:
        return composed
    return composed[:64].rstrip("_")
