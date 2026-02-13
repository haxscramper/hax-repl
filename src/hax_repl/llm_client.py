from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Iterable

import httpx

LOGGER = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_KEY_ENV = "HAXSCRAMPER_LLM_REPL_KEY"


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


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
            "messages": [{"role": m.role, "content": m.content} for m in messages],
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
            response.raise_for_status()
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
                    LOGGER.warning("Skipping malformed SSE chunk from OpenRouter.")
                    continue
                token = _extract_content_delta(event)
                if token:
                    yield token


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
