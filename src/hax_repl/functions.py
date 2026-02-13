from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Generic, Protocol, TypeVar

from pydantic import BaseModel

TArgs = TypeVar("TArgs", bound=BaseModel)
TResult = TypeVar("TResult")
FUNCTION_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@dataclass(frozen=True)
class FunctionSpec(Generic[TArgs, TResult]):
    name: str
    description: str
    args_model: type[TArgs]
    result_model: type[BaseModel] | None
    impl: Callable[[TArgs], TResult]


class FunctionProvider(Protocol):
    def provider_name(self) -> str: ...

    def functions(self) -> list[FunctionSpec[BaseModel, Any]]: ...


class FunctionRegistry:
    def __init__(self) -> None:
        self._functions: dict[str, FunctionSpec[BaseModel, Any]] = {}

    def register(self, function_spec: FunctionSpec[BaseModel, Any]) -> None:
        if not FUNCTION_NAME_PATTERN.match(function_spec.name):
            raise ValueError(
                "Invalid function name "
                f"'{function_spec.name}'. Must match {FUNCTION_NAME_PATTERN.pattern}"
            )
        self._functions[function_spec.name] = function_spec

    def all_specs(self) -> list[FunctionSpec[BaseModel, Any]]:
        return [self._functions[name] for name in sorted(self._functions.keys())]

    def names(self) -> list[str]:
        return sorted(self._functions.keys())

    def get(self, name: str) -> FunctionSpec[BaseModel, Any]:
        if name not in self._functions:
            raise KeyError(f"Unknown function: {name}")
        return self._functions[name]

    def enabled_specs(self, enabled_function_names: list[str] | None = None) -> list[FunctionSpec[BaseModel, Any]]:
        if enabled_function_names is None or not enabled_function_names:
            return self.all_specs()
        specs: list[FunctionSpec[BaseModel, Any]] = []
        for name in enabled_function_names:
            specs.append(self.get(name))
        return specs

    def schema_hash_for(self, function_spec: FunctionSpec[BaseModel, Any]) -> str:
        payload = {
            "name": function_spec.name,
            "description": function_spec.description,
            "args_schema": function_spec.args_model.model_json_schema(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.md5(encoded.encode("utf-8")).hexdigest()

    def invoke_json(self, function_name: str, arguments_json: str) -> str:
        spec = self.get(function_name)
        try:
            raw_args = json.loads(arguments_json)
        except json.JSONDecodeError:
            raw_args = {}
        args_model = spec.args_model.model_validate(raw_args)
        result = spec.impl(args_model)
        if isinstance(result, BaseModel):
            return result.model_dump_json()
        if isinstance(result, str):
            return json.dumps({"result": result}, ensure_ascii=True)
        return json.dumps(result, ensure_ascii=True)

    def to_tool_specs(
        self, enabled_function_names: list[str] | None = None
    ) -> list[dict[str, object]]:
        tools: list[dict[str, object]] = []
        for spec in self.enabled_specs(enabled_function_names):
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": spec.args_model.model_json_schema(),
                    },
                }
            )
        return tools

    def list_with_schema_hashes(
        self, enabled_function_names: list[str] | None = None
    ) -> list[tuple[str, str]]:
        result: list[tuple[str, str]] = []
        for spec in self.enabled_specs(enabled_function_names):
            result.append((spec.name, self.schema_hash_for(spec)))
        return result


class PythonEvalArgs(BaseModel):
    expression: str


class PythonEvalResult(BaseModel):
    result_repr: str


def _python_eval_impl(args: PythonEvalArgs) -> PythonEvalResult:
    result = eval(args.expression)
    return PythonEvalResult(result_repr=repr(result))


class BuiltinFunctionProvider:
    def provider_name(self) -> str:
        return "builtin"

    def functions(self) -> list[FunctionSpec[BaseModel, Any]]:
        return [
            FunctionSpec(
                name="python_eval",
                description="Evaluate a Python expression and return repr(result).",
                args_model=PythonEvalArgs,
                result_model=PythonEvalResult,
                impl=_python_eval_impl,
            )
        ]
