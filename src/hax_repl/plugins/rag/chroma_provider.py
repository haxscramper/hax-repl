from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

import httpx

from hax_repl.models import PluginRagMeta
from hax_repl.plugin_system import PluginDescriptor
from hax_repl.plugins.rag._chunking import chunk_text
from hax_repl.rag import options_get_int, RagChunk, RagProvider, RagResult

OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
OPENROUTER_KEY_ENV = "HAXSCRAMPER_LLM_REPL_KEY"
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"


class ChromaVectorRagProvider(RagProvider):

    def __init__(self) -> None:
        import chromadb

        app_dir = Path.home() / ".local" / "share" / "haxllm" / "rag" / "chroma"
        app_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(app_dir))
        self._provider_name = "chroma"
        api_key = os.getenv(OPENROUTER_KEY_ENV, "").strip()
        if not api_key:
            raise RuntimeError(
                f"{OPENROUTER_KEY_ENV} is required for Chroma vector embeddings.")
        self._api_key = api_key
        model = os.getenv("HAX_REPL_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip()
        if "/" not in model:
            self._embedding_model = f"openai/{model}"
        else:
            self._embedding_model = model

    def list_indices(self) -> Sequence[str]:
        return [collection.name for collection in self._client.list_collections()]

    def query(self,
              index_name: str,
              query_text: str,
              options_json: str = "{}") -> RagResult:
        top_k = max(1, options_get_int(options_json, "top_k", 5))
        collection = self._client.get_or_create_collection(name=index_name)
        embedding = self._embed(query_text)
        result = collection.query(query_embeddings=[embedding], n_results=top_k)
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        chunks: list[RagChunk] = []
        for idx, doc in enumerate(documents):
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            distance = distances[idx] if idx < len(distances) else 0.0
            score = 1.0 / (1.0 + float(distance))
            source_id = str(metadata.get("source_id", f"{index_name}:{idx}"))
            chunks.append(RagChunk(source_id=source_id, text=str(doc), score=score))
        return RagResult(provider=self._provider_name, index=index_name, chunks=chunks)

    def update_index(self,
                     index_name: str,
                     sources: Sequence[str],
                     options_json: str = "{}") -> None:
        collection = self._client.get_or_create_collection(name=index_name)
        chunk_size = max(200, options_get_int(options_json, "chunk_size", 800))
        overlap = max(0, options_get_int(options_json, "chunk_overlap", 120))

        doc_ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, str]] = []
        for source in sources:
            path = Path(source).expanduser()
            if not path.exists() or not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for idx, chunk in enumerate(
                    chunk_text(text, chunk_size=chunk_size, overlap=overlap)):
                doc_id = f"{path}:{idx}"
                doc_ids.append(doc_id)
                documents.append(chunk)
                metadatas.append({"source_id": str(path)})
        if not documents:
            return
        embeddings = self._embed_many(documents)
        collection.upsert(ids=doc_ids,
                          documents=documents,
                          embeddings=embeddings,
                          metadatas=metadatas)

    def _embed(self, text: str) -> list[float]:
        return self._embed_many([text])[0]

    def _embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._embedding_model,
            "input": texts,
        }
        response = httpx.post(
            OPENROUTER_EMBEDDINGS_URL,
            headers=headers,
            json=payload,
            timeout=120.0,
        )
        self._raise_for_status_with_details(response)
        data = response.json().get("data")
        if not isinstance(data, list):
            raise RuntimeError("OpenRouter embeddings response missing 'data' list.")
        embeddings: list[list[float]] = []
        for item in data:
            if not isinstance(item, dict):
                raise RuntimeError(
                    "OpenRouter embeddings response contains non-object item.")
            raw_embedding = item.get("embedding")
            if not isinstance(raw_embedding, list):
                raise RuntimeError(
                    "OpenRouter embeddings response item missing 'embedding' list.")
            embeddings.append([float(value) for value in raw_embedding])
        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"OpenRouter embeddings count mismatch: expected {len(texts)}, got {len(embeddings)}."
            )
        return embeddings

    @staticmethod
    def _raise_for_status_with_details(response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = response.text[:2000].strip() or "<empty response body>"
            try:
                parsed = json.loads(body)
                body = json.dumps(parsed, ensure_ascii=True)
            except json.JSONDecodeError:
                pass
            raise RuntimeError(
                f"OpenRouter embeddings request failed with status={response.status_code}. Response body: {body}"
            ) from exc


def register() -> PluginDescriptor:
    return PluginDescriptor(
        metadata=PluginRagMeta(
            name="chroma",
            description="Chroma vector-store RAG provider.",
        ),
        plugin_factory=ChromaVectorRagProvider,
    )
