from __future__ import annotations

from pathlib import Path
from typing import Sequence

from hax_repl.models import PluginRagMeta
from hax_repl.plugin_system import PluginDescriptor
from hax_repl.plugins.rag._chunking import chunk_text
from hax_repl.rag import RagChunk, RagProvider, RagResult, options_get_int


class TantivyFullTextRagProvider(RagProvider):

    def __init__(self) -> None:
        import tantivy

        base_dir = Path.home(
        ) / ".local" / "share" / "haxllm" / "rag" / "tantivy"
        base_dir.mkdir(parents=True, exist_ok=True)
        self._base_dir = base_dir
        self._tantivy = tantivy
        self._provider_name = "tantivy"

    def list_indices(self) -> Sequence[str]:
        return [p.name for p in self._base_dir.iterdir() if p.is_dir()]

    def query(self,
              index_name: str,
              query_text: str,
              options_json: str = "{}") -> RagResult:
        index = self._open_index(index_name)
        query_parser = index.parse_query(query_text, ["body"])
        searcher = index.searcher()
        top_k = max(1, options_get_int(options_json, "top_k", 5))
        top_docs = searcher.search(query_parser, top_k)
        chunks: list[RagChunk] = []
        for score, address in top_docs:
            doc = searcher.doc(address)
            source_id = str(
                doc.get_first("source_id") or f"{index_name}:{address}")
            text = str(doc.get_first("body") or "")
            chunks.append(
                RagChunk(source_id=source_id, text=text, score=float(score)))
        return RagResult(provider=self._provider_name,
                         index=index_name,
                         chunks=chunks)

    def update_index(self,
                     index_name: str,
                     sources: Sequence[str],
                     options_json: str = "{}") -> None:
        index = self._open_or_create_index(index_name)
        writer = index.writer()
        chunk_size = max(200, options_get_int(options_json, "chunk_size", 800))
        overlap = max(0, options_get_int(options_json, "chunk_overlap", 120))
        for source in sources:
            path = Path(source).expanduser()
            if not path.exists() or not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for idx, chunk in enumerate(
                    chunk_text(text, chunk_size=chunk_size, overlap=overlap)):
                writer.add_document({
                    "source_id": f"{path}:{idx}",
                    "body": chunk
                })
        writer.commit()

    def _open_or_create_index(self, index_name: str):
        dir_path = self._base_dir / index_name
        dir_path.mkdir(parents=True, exist_ok=True)
        schema_builder = self._tantivy.SchemaBuilder()
        schema_builder.add_text_field("source_id", stored=True)
        schema_builder.add_text_field("body", stored=True)
        schema = schema_builder.build()
        if any(dir_path.iterdir()):
            return self._tantivy.Index.open(str(dir_path))
        return self._tantivy.Index(schema, path=str(dir_path))

    def _open_index(self, index_name: str):
        dir_path = self._base_dir / index_name
        if not dir_path.exists():
            return self._open_or_create_index(index_name)
        return self._tantivy.Index.open(str(dir_path))


def register() -> PluginDescriptor:
    return PluginDescriptor(
        metadata=PluginRagMeta(
            name="tantivy",
            description="Tantivy full-text RAG provider.",
        ),
        plugin_factory=TantivyFullTextRagProvider,
    )
