from __future__ import annotations

from dataclasses import dataclass
import platform
import re
import shlex
from typing import Callable, Sequence

from hax_repl.rag import RagResult

MACRO_PATTERN = re.compile(r"\$\((.*?)\)")


@dataclass(frozen=True)
class MacroExpansionResult:
    expanded_text: str
    rag_records: list[RagResult]


class MacroExpander:
    """POC macro expander placeholder.

    For phases 0-2 we only parse macros and return their body as-is.
    """

    def expand(
        self,
        prompt: str,
        *,
        rag_query: Callable[[str, str, str], RagResult] | None = None,
    ) -> MacroExpansionResult:
        rag_records: list[RagResult] = []

        def _replace(match: re.Match[str]) -> str:
            body = match.group(1).strip()
            if body == "get-os":
                return platform.platform()
            if body.startswith("rag:"):
                result = self._expand_rag_macro(body, rag_query=rag_query)
                if result is None:
                    return body
                rag_records.append(result)
                return self._rag_result_to_text(result)
            return body

        expanded = MACRO_PATTERN.sub(_replace, prompt)
        return MacroExpansionResult(expanded_text=expanded, rag_records=rag_records)

    def _expand_rag_macro(
        self,
        body: str,
        *,
        rag_query: Callable[[str, str, str], RagResult] | None = None,
    ) -> RagResult | None:
        if rag_query is None:
            return None
        # Expected: rag:provider/index "query text"
        rest = body.removeprefix("rag:").strip()
        if not rest:
            return None
        try:
            parts = shlex.split(rest)
        except ValueError:
            return None
        if len(parts) < 2:
            return None
        provider_index = parts[0]
        query_text = " ".join(parts[1:])
        if "/" not in provider_index:
            return None
        provider_name, index_name = provider_index.split("/", 1)
        return rag_query(provider_name, index_name, query_text)

    def _rag_result_to_text(self, result: RagResult) -> str:
        lines: list[str] = [
            f"<rag provider=\"{result.provider}\" index=\"{result.index}\">"
        ]
        for chunk in result.chunks:
            lines.append(
                f"- [{chunk.source_id}] score={chunk.score:.4f} text={chunk.text[:240]}")
        lines.append("</rag>")
        return "\n".join(lines)
