from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from hax_repl.functions import FunctionProvider, FunctionSpec


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
