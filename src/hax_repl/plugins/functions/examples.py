from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from hax_repl.functions import FunctionProvider, FunctionSpec


class EchoTextArgs(BaseModel):
    text: str
    uppercase: bool = False


class EchoTextResult(BaseModel):
    text: str
    length: int


class SumNumbersArgs(BaseModel):
    numbers: list[float] = Field(default_factory=list)


class SumNumbersResult(BaseModel):
    total: float
    count: int


class ListDirectoryArgs(BaseModel):
    path: str = "."
    include_hidden: bool = False
    max_entries: int = 50


class ListDirectoryResult(BaseModel):
    path: str
    entries: list[str]


class ReadTextFileArgs(BaseModel):
    path: str
    max_chars: int = 4000


class ReadTextFileResult(BaseModel):
    path: str
    content: str
    truncated: bool


def _echo_text_impl(args: EchoTextArgs) -> EchoTextResult:
    text = args.text.upper() if args.uppercase else args.text
    return EchoTextResult(text=text, length=len(text))


def _sum_numbers_impl(args: SumNumbersArgs) -> SumNumbersResult:
    total = float(sum(args.numbers))
    return SumNumbersResult(total=total, count=len(args.numbers))


def _list_directory_impl(args: ListDirectoryArgs) -> ListDirectoryResult:
    target_path = Path(args.path).expanduser()
    entries: list[str] = []
    for child in sorted(target_path.iterdir(), key=lambda p: p.name):
        if not args.include_hidden and child.name.startswith("."):
            continue
        suffix = "/" if child.is_dir() else ""
        entries.append(f"{child.name}{suffix}")
        if len(entries) >= max(args.max_entries, 1):
            break
    return ListDirectoryResult(path=str(target_path), entries=entries)


def _read_text_file_impl(args: ReadTextFileArgs) -> ReadTextFileResult:
    target_path = Path(args.path).expanduser()
    content = target_path.read_text(encoding="utf-8", errors="replace")
    truncated = False
    if len(content) > args.max_chars:
        content = content[: args.max_chars]
        truncated = True
    return ReadTextFileResult(path=str(target_path), content=content, truncated=truncated)


class ExampleFunctionProvider(FunctionProvider):
    def provider_name(self) -> str:
        return "example-functions"

    def functions(self) -> list[FunctionSpec[BaseModel, Any]]:
        return [
            FunctionSpec(
                name="echo_text",
                description="Echo text and return its length.",
                args_model=EchoTextArgs,
                result_model=EchoTextResult,
                impl=_echo_text_impl,
            ),
            FunctionSpec(
                name="sum_numbers",
                description="Sum numeric values in a list.",
                args_model=SumNumbersArgs,
                result_model=SumNumbersResult,
                impl=_sum_numbers_impl,
            ),
            FunctionSpec(
                name="list_directory",
                description="List files and directories in a path.",
                args_model=ListDirectoryArgs,
                result_model=ListDirectoryResult,
                impl=_list_directory_impl,
            ),
            FunctionSpec(
                name="read_text_file",
                description="Read a UTF-8 text file with optional truncation.",
                args_model=ReadTextFileArgs,
                result_model=ReadTextFileResult,
                impl=_read_text_file_impl,
            ),
        ]


class JsonUtilityArgs(BaseModel):
    value_json: str


class JsonUtilityResult(BaseModel):
    pretty_json: str


def _pretty_json_impl(args: JsonUtilityArgs) -> JsonUtilityResult:
    payload = json.loads(args.value_json)
    return JsonUtilityResult(pretty_json=json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))


class JsonFunctionProvider(FunctionProvider):
    def provider_name(self) -> str:
        return "json-functions"

    def functions(self) -> list[FunctionSpec[BaseModel, Any]]:
        return [
            FunctionSpec(
                name="pretty_json",
                description="Format JSON string with indentation and sorted keys.",
                args_model=JsonUtilityArgs,
                result_model=JsonUtilityResult,
                impl=_pretty_json_impl,
            )
        ]
