from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import httpx

LOGGER = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_KEY_ENV = "HAXSCRAMPER_LLM_REPL_KEY"


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list["ToolCall"] | None = None


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments_json: str


@dataclass(frozen=True)
class ChatCompletionResult:
    text: str
    tool_calls: list[ToolCall]


class OpenRouterClient:

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._api_key = os.getenv(OPENROUTER_KEY_ENV, "").strip()
        if not self._api_key:
            raise RuntimeError(
                f"Environment variable {OPENROUTER_KEY_ENV} is required for LLM requests."
            )

    @property
    def model_name(self) -> str:
        return self._model_name

    def stream_chat(self, messages: list[ChatMessage]) -> Iterable[str]:
        request_payload = {
            "model": self._model_name,
            "messages": [_serialize_message(m) for m in messages],
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        with httpx.stream(
                "POST",
                OPENROUTER_URL,
                headers=headers,
                json=request_payload,
                timeout=120.0,
        ) as response:
            _raise_for_status_with_details(response)
            for raw_line in response.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.strip()
                if not line.startswith("data: "):
                    continue
                data = line.removeprefix("data: ").strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    LOGGER.warning(
                        "Skipping malformed SSE chunk from OpenRouter.")
                    continue
                token = _extract_content_delta(event)
                if token:
                    yield token

    def complete_chat_with_tools(
        self,
        messages: list[ChatMessage],
        tool_specs: Sequence[dict[str, object]],
    ) -> ChatCompletionResult:
        request_payload: dict[str, Any] = {
            "model": self._model_name,
            "messages": [_serialize_message(m) for m in messages],
            "stream": False,
        }
        if tool_specs:
            request_payload["tools"] = list(tool_specs)
            request_payload["tool_choice"] = "auto"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        response = httpx.post(
            OPENROUTER_URL,
            headers=headers,
            json=request_payload,
            timeout=120.0,
        )
        _raise_for_status_with_details(response)
        payload = response.json()
        choices = payload.get("choices", [])
        if not choices:
            return ChatCompletionResult(text="", tool_calls=[])
        message = choices[0].get("message", {})
        text = _extract_message_text(message)
        tool_calls = _extract_tool_calls(message)
        return ChatCompletionResult(text=text, tool_calls=tool_calls)


def _serialize_message(message: ChatMessage) -> dict[str, object]:
    payload: dict[str, object] = {"role": message.role}
    if message.content is not None:
        payload["content"] = message.content
    if message.name is not None:
        payload["name"] = message.name
    if message.tool_call_id is not None:
        payload["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        payload["tool_calls"] = [{
            "id": call.id,
            "type": "function",
            "function": {
                "name": call.name,
                "arguments": call.arguments_json
            },
        } for call in message.tool_calls]
    return payload


def _extract_content_delta(event: dict[str, object]) -> str:
    choices = event.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    delta = first.get("delta")
    if not isinstance(delta, dict):
        return ""
    content = delta.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _extract_message_text(message: object) -> str:
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _extract_tool_calls(message: object) -> list[ToolCall]:
    if not isinstance(message, dict):
        return []
    raw_tool_calls = message.get("tool_calls")
    if not isinstance(raw_tool_calls, list):
        return []
    calls: list[ToolCall] = []
    for raw in raw_tool_calls:
        if not isinstance(raw, dict):
            continue
        call_id = raw.get("id")
        function_data = raw.get("function")
        if not isinstance(call_id, str) or not isinstance(function_data, dict):
            continue
        name = function_data.get("name")
        arguments = function_data.get("arguments")
        if not isinstance(name, str) or not isinstance(arguments, str):
            continue
        calls.append(ToolCall(id=call_id, name=name, arguments_json=arguments))
    return calls


def _raise_for_status_with_details(response: httpx.Response) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        snippet = response.text[:2000]
        detail = snippet.strip() or "<empty response body>"
        raise RuntimeError(
            f"OpenRouter request failed with status={response.status_code}. "
            f"Response body: {detail}") from exc
