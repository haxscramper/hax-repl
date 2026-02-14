from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Sequence

from hax_repl.models import ContentHashID, ContextHashID


def _md5_for_json(payload: object) -> str:
    encoded = json.dumps(payload,
                         sort_keys=True,
                         separators=(",", ":"),
                         ensure_ascii=True)
    return hashlib.md5(encoded.encode("utf-8")).hexdigest()


def content_hash_for_prompt(
    *,
    original_prompt: str,
    augmented_prompt: str,
    included_context_ids: Sequence[str],
) -> ContentHashID:
    return ContentHashID(md5=_md5_for_json({
        "kind": "prompt",
        "original_prompt": original_prompt,
        "augmented_prompt": augmented_prompt,
        "included_context_ids": list(included_context_ids),
    }))


def content_hash_for_response(
    *,
    text: str,
    function_calls_json: Sequence[str],
    function_results_json: Sequence[str],
    thinking_text: str,
) -> ContentHashID:
    return ContentHashID(md5=_md5_for_json({
        "kind": "response",
        "text": text,
        "function_calls_json": list(function_calls_json),
        "function_results_json": list(function_results_json),
        "thinking_text": thinking_text,
    }))


def context_hash(
    *,
    content_hash: ContentHashID,
    model_name: str,
    rag_provenance: Sequence[str],
) -> ContextHashID:
    return ContextHashID(md5=_md5_for_json({
        "content_hash": content_hash.md5,
        "model_name": model_name,
        "rag_provenance": list(rag_provenance),
    }))


@dataclass(frozen=True)
class SeparatedResponseText:
    visible_text: str
    thinking_text: str


def split_thinking_blocks(full_text: str) -> SeparatedResponseText:
    start_tag = "<think>"
    end_tag = "</think>"

    visible_parts: list[str] = []
    thinking_parts: list[str] = []
    i = 0
    in_think = False
    buffer: list[str] = []

    while i < len(full_text):
        if not in_think and full_text.startswith(start_tag, i):
            if buffer:
                visible_parts.append("".join(buffer))
                buffer = []
            in_think = True
            i += len(start_tag)
            continue
        if in_think and full_text.startswith(end_tag, i):
            if buffer:
                thinking_parts.append("".join(buffer))
                buffer = []
            in_think = False
            i += len(end_tag)
            continue
        buffer.append(full_text[i])
        i += 1

    if buffer:
        if in_think:
            thinking_parts.append("".join(buffer))
        else:
            visible_parts.append("".join(buffer))

    return SeparatedResponseText(
        visible_text="".join(visible_parts),
        thinking_text="".join(thinking_parts),
    )
