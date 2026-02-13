from __future__ import annotations

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
