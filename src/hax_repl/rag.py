from __future__ import annotations

import json
from typing import Protocol, Sequence

from pydantic import BaseModel, Field


class RagChunk(BaseModel):
    source_id: str
    text: str
    score: float


class RagResult(BaseModel):
    provider: str
    index: str
    chunks: Sequence[RagChunk] = Field(default_factory=list)


class RagInvocationRecord(BaseModel):
    result: RagResult


class RagProvider(Protocol):
    def list_indices(self) -> Sequence[str]: ...

    def query(self, index_name: str, query_text: str, options_json: str = "{}") -> RagResult: ...

    def update_index(self, index_name: str, sources: Sequence[str], options_json: str = "{}") -> None: ...


class RagRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, RagProvider] = {}

    def register(self, provider_name: str, provider: RagProvider) -> None:
        self._providers[provider_name] = provider

    def get(self, provider_name: str) -> RagProvider:
        if provider_name not in self._providers:
            raise KeyError(f"Unknown RAG provider: {provider_name}")
        return self._providers[provider_name]

    def list_providers(self) -> list[str]:
        return sorted(self._providers.keys())

    def list_indices(self, provider_name: str) -> Sequence[str]:
        provider = self.get(provider_name)
        return provider.list_indices()

    def query(
        self,
        *,
        provider_name: str,
        index_name: str,
        query_text: str,
        options_json: str = "{}",
    ) -> RagResult:
        provider = self.get(provider_name)
        return provider.query(index_name=index_name, query_text=query_text, options_json=options_json)

    def update_index(
        self,
        *,
        provider_name: str,
        index_name: str,
        sources: Sequence[str],
        options_json: str = "{}",
    ) -> None:
        provider = self.get(provider_name)
        provider.update_index(index_name=index_name, sources=sources, options_json=options_json)


def options_get_int(options_json: str, key: str, default: int) -> int:
    try:
        payload = json.loads(options_json) if options_json.strip() else {}
    except json.JSONDecodeError:
        return default
    raw = payload.get(key, default)
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    return default
