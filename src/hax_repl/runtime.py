from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Callable, Sequence

from hax_repl.hashing import (
    content_hash_for_prompt,
    content_hash_for_response,
    context_hash,
    split_thinking_blocks,
)
from hax_repl.llm_client import ChatMessage, OpenRouterClient
from hax_repl.message_store import MessageStore
from hax_repl.models import (
    ContextHashID,
    ModelName,
    PromptMessage,
    ResponseMessage,
    RoleName,
    SessionFile,
    SessionName,
)
from hax_repl.session_store import SessionStore

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueryStats:
    elapsed_ms: int
    query_chars: int
    response_chars: int
    model_name: str
    time_until_first_token_ms: int
    model_thinking_ms: int


@dataclass(frozen=True)
class RuntimeResponse:
    prompt_message: PromptMessage
    response_message: ResponseMessage
    stats: QueryStats


class AppRuntime:
    def __init__(self, session_name: str | None, model_name: str = "anthropic/claude-sonnet-4.5") -> None:
        app_dir = Path.home() / ".local" / "share" / "haxllm"
        sessions_dir = app_dir / "sessions"
        db_path = app_dir / "messages.sqlite3"

        self._session_store = SessionStore(sessions_dir)
        self._message_store = MessageStore(db_path)
        self._model_name = model_name
        self._client = OpenRouterClient(model_name=model_name)

        resolved_name: SessionName = self._session_store.resolve_session_name(session_name)
        self._session: SessionFile = self._session_store.load_or_create(resolved_name)
        self._default_role = RoleName(value="user")
        self._default_agent = "default-agent"

    @property
    def session(self) -> SessionFile:
        return self._session

    @property
    def prompt_state_label(self) -> str:
        return f"{self._session.session.value}/{self._default_agent}|{self._default_role.value})"

    @property
    def model_name(self) -> str:
        return self._model_name

    def next_query_index(self) -> int:
        return len(self._session.turns) + 1

    def _conversation_messages(self) -> list[ChatMessage]:
        messages: list[ChatMessage] = []
        for turn in self._session.turns:
            prompt = self._message_store.get(turn.prompt_id)
            if isinstance(prompt, PromptMessage):
                messages.append(ChatMessage(role=prompt.role.value, content=prompt.augmented_prompt))
            if turn.response_id is not None:
                response = self._message_store.get(turn.response_id)
                if isinstance(response, ResponseMessage):
                    messages.append(ChatMessage(role="assistant", content=response.text))
        return messages

    def send_prompt(
        self,
        original_prompt: str,
        enabled_functions: Sequence[str] | None = None,
        on_visible_token: Callable[[str], None] | None = None,
    ) -> RuntimeResponse:
        enabled_function_list = list(enabled_functions or [])
        included_context_ids = [cid.md5 for cid in self._session.message_ids]
        augmented_prompt = original_prompt

        prompt_content_id = content_hash_for_prompt(
            original_prompt=original_prompt,
            augmented_prompt=augmented_prompt,
            enabled_functions=enabled_function_list,
            included_context_ids=included_context_ids,
        )
        prompt_context_id = context_hash(
            content_hash=prompt_content_id,
            model_name=self._model_name,
            function_schema_hashes=[],
            rag_provenance=[],
        )
        prompt_message = PromptMessage(
            context_id=prompt_context_id,
            content_id=prompt_content_id,
            role=self._default_role,
            original_prompt=original_prompt,
            augmented_prompt=augmented_prompt,
            included_context=[ContextHashID(md5=cid) for cid in included_context_ids],
            enabled_functions=enabled_function_list,
        )

        llm_messages = self._conversation_messages()
        llm_messages.append(ChatMessage(role="user", content=augmented_prompt))
        query_chars = sum(len(message.content) for message in llm_messages)

        self._message_store.upsert(prompt_message)
        self._session = self._session_store.append_prompt(self._session, prompt_context_id)

        started = monotonic()
        first_token_at: float | None = None
        first_visible_token_at: float | None = None
        raw_full_text = ""
        visible_so_far = ""
        thinking_so_far = ""
        for chunk in self._client.stream_chat(llm_messages):
            if first_token_at is None:
                first_token_at = monotonic()
            raw_full_text += chunk
            split = split_thinking_blocks(raw_full_text)
            visible_delta = split.visible_text[len(visible_so_far) :]
            thinking_so_far = split.thinking_text
            visible_so_far = split.visible_text
            if visible_delta and on_visible_token is not None:
                if first_visible_token_at is None:
                    first_visible_token_at = monotonic()
                on_visible_token(visible_delta)
            elif visible_delta and first_visible_token_at is None:
                first_visible_token_at = monotonic()

        visible_text = visible_so_far.strip()
        thinking_text = thinking_so_far.strip()
        response_content_id = content_hash_for_response(
            text=visible_text,
            function_calls_json=[],
            function_results_json=[],
            thinking_text=thinking_text,
        )
        response_context_id = context_hash(
            content_hash=response_content_id,
            model_name=self._model_name,
            function_schema_hashes=[],
            rag_provenance=[],
        )
        response_message = ResponseMessage(
            context_id=response_context_id,
            content_id=response_content_id,
            model=ModelName(value=self._model_name),
            in_response_to=prompt_context_id,
            text=visible_text,
            thinking_text=thinking_text,
        )
        self._message_store.upsert(response_message)
        self._session = self._session_store.attach_response_to_last_turn(self._session, response_context_id)

        completed_at = monotonic()
        elapsed_ms = int((completed_at - started) * 1000)
        time_until_first_token_ms = (
            int((first_token_at - started) * 1000) if first_token_at is not None else elapsed_ms
        )
        if first_token_at is None:
            model_thinking_ms = 0
        elif first_visible_token_at is not None:
            model_thinking_ms = max(int((first_visible_token_at - first_token_at) * 1000), 0)
        else:
            model_thinking_ms = max(int((completed_at - first_token_at) * 1000), 0)
        stats = QueryStats(
            elapsed_ms=elapsed_ms,
            query_chars=query_chars,
            response_chars=len(visible_text),
            model_name=self._model_name,
            time_until_first_token_ms=time_until_first_token_ms,
            model_thinking_ms=model_thinking_ms,
        )
        LOGGER.info(
            "Prompt sent successfully elapsed_ms=%s first_token_ms=%s thinking_ms=%s query_chars=%s",
            elapsed_ms,
            time_until_first_token_ms,
            model_thinking_ms,
            query_chars,
        )
        return RuntimeResponse(prompt_message=prompt_message, response_message=response_message, stats=stats)
