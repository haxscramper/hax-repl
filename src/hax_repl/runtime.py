from __future__ import annotations

import json
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
from hax_repl.functions import BuiltinFunctionProvider, FunctionProvider, FunctionRegistry
from hax_repl.llm_client import ChatCompletionResult, ChatMessage, OpenRouterClient, ToolCall
from hax_repl.macro import MacroExpander
from hax_repl.message_store import MessageStore
from hax_repl.models import (
    AnyMessage,
    ContextHashID,
    EnabledFunction,
    FunctionCallRequest,
    FunctionCallResult,
    ModelName,
    PromptMessage,
    ResponseMessage,
    RoleName,
    SessionFile,
    SessionName,
)
from hax_repl.plugin_system import instantiate_plugin, load_plugins_or_fail
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
        self._macro_expander = MacroExpander()
        self._function_registry = self._build_function_registry()

        resolved_name: SessionName = self._session_store.resolve_session_name(session_name)
        self._session: SessionFile = self._session_store.load_or_create(resolved_name)
        self._default_role = RoleName(value="user")
        self._default_agent = "default-agent"

    def _build_function_registry(self) -> FunctionRegistry:
        registry = FunctionRegistry()
        builtin_provider: FunctionProvider = BuiltinFunctionProvider()
        for function_spec in builtin_provider.functions():
            registry.register(function_spec)

        for loaded in load_plugins_or_fail("hax_repl.function_providers"):
            provider_candidate = instantiate_plugin(loaded.plugin)
            if not hasattr(provider_candidate, "functions"):
                raise RuntimeError(f"Invalid function provider plugin: {loaded.name}")
            provider = provider_candidate
            functions = provider.functions()
            for function_spec in functions:
                registry.register(function_spec)
        return registry

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

    def _message_for_context_id(self, context_id: ContextHashID) -> AnyMessage | None:
        return self._message_store.get(context_id)

    def last_prompt_message(self) -> PromptMessage | None:
        for turn in reversed(self._session.turns):
            message = self._message_for_context_id(turn.prompt_id)
            if isinstance(message, PromptMessage):
                return message
        return None

    def last_response_message(self) -> ResponseMessage | None:
        for turn in reversed(self._session.turns):
            if turn.response_id is None:
                continue
            message = self._message_for_context_id(turn.response_id)
            if isinstance(message, ResponseMessage):
                return message
        return None

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

    def _enabled_function_refs(self, enabled_function_names: Sequence[str] | None) -> list[EnabledFunction]:
        specs = self._function_registry.enabled_specs(
            list(enabled_function_names) if enabled_function_names else None
        )
        return [
            EnabledFunction(name=spec.name, schema_hash=self._function_registry.schema_hash_for(spec))
            for spec in specs
        ]

    def _function_schema_hashes(self, refs: Sequence[EnabledFunction]) -> list[str]:
        return [ref.schema_hash for ref in refs]

    def _run_function_calling_loop(
        self,
        *,
        llm_messages: list[ChatMessage],
        enabled_function_names: list[str],
    ) -> tuple[ChatCompletionResult, list[FunctionCallRequest], list[FunctionCallResult]]:
        messages = list(llm_messages)
        tool_specs = self._function_registry.to_tool_specs(enabled_function_names)
        all_requests: list[FunctionCallRequest] = []
        all_results: list[FunctionCallResult] = []
        max_rounds = 6
        last_result = ChatCompletionResult(text="", tool_calls=[])

        for _ in range(max_rounds):
            completion = self._client.complete_chat_with_tools(messages=messages, tool_specs=tool_specs)
            last_result = completion
            if not completion.tool_calls:
                return completion, all_requests, all_results

            messages.append(
                ChatMessage(
                    role="assistant",
                    content=completion.text,
                    tool_calls=completion.tool_calls,
                )
            )
            for tool_call in completion.tool_calls:
                request = FunctionCallRequest(name=tool_call.name, arguments_json=tool_call.arguments_json)
                all_requests.append(request)
                result_json = self._invoke_tool_with_failure_payload(tool_call)
                all_results.append(FunctionCallResult(name=tool_call.name, result_json=result_json))
                messages.append(
                    ChatMessage(
                        role="tool",
                        content=result_json,
                        name=tool_call.name,
                        tool_call_id=tool_call.id,
                    )
                )

        return last_result, all_requests, all_results

    def _invoke_tool_with_failure_payload(self, tool_call: ToolCall) -> str:
        try:
            return self._function_registry.invoke_json(tool_call.name, tool_call.arguments_json)
        except Exception as exc:
            LOGGER.exception("Function call failed: %s", tool_call.name)
            return json.dumps(
                {
                    "ok": False,
                    "error": str(exc),
                    "tool_name": tool_call.name,
                },
                ensure_ascii=True,
            )

    def send_prompt(
        self,
        original_prompt: str,
        enabled_functions: Sequence[str] | None = None,
        on_visible_token: Callable[[str], None] | None = None,
    ) -> RuntimeResponse:
        enabled_function_refs = self._enabled_function_refs(enabled_functions)
        included_context_ids = [cid.md5 for cid in self._session.message_ids]
        augmented_prompt = self._macro_expander.expand(original_prompt)

        prompt_content_id = content_hash_for_prompt(
            original_prompt=original_prompt,
            augmented_prompt=augmented_prompt,
            enabled_functions=enabled_function_refs,
            included_context_ids=included_context_ids,
        )
        prompt_context_id = context_hash(
            content_hash=prompt_content_id,
            model_name=self._model_name,
            function_schema_hashes=self._function_schema_hashes(enabled_function_refs),
            rag_provenance=[],
        )
        prompt_message = PromptMessage(
            context_id=prompt_context_id,
            content_id=prompt_content_id,
            role=self._default_role,
            original_prompt=original_prompt,
            augmented_prompt=augmented_prompt,
            included_context=[ContextHashID(md5=cid) for cid in included_context_ids],
            enabled_functions=enabled_function_refs,
        )

        llm_messages = self._conversation_messages()
        llm_messages.append(ChatMessage(role="user", content=augmented_prompt))
        query_chars = sum(len(message.content) for message in llm_messages)

        self._message_store.upsert(prompt_message)
        self._session = self._session_store.append_prompt(self._session, prompt_context_id)

        started = monotonic()
        first_token_at: float | None = None
        first_visible_token_at: float | None = None
        function_call_requests: list[FunctionCallRequest] = []
        function_call_results: list[FunctionCallResult] = []

        enabled_function_names = [ref.name for ref in enabled_function_refs]
        if enabled_function_names:
            completion_result, function_call_requests, function_call_results = self._run_function_calling_loop(
                llm_messages=llm_messages,
                enabled_function_names=enabled_function_names,
            )
            first_token_at = monotonic()
            first_visible_token_at = first_token_at
            split = split_thinking_blocks(completion_result.text)
            visible_text = split.visible_text.strip()
            thinking_text = split.thinking_text.strip()
            if on_visible_token is not None and visible_text:
                on_visible_token(visible_text)
        else:
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
            function_calls_json=[
                json.dumps(
                    {"name": call.name, "arguments_json": call.arguments_json},
                    sort_keys=True,
                    ensure_ascii=True,
                )
                for call in function_call_requests
            ],
            function_results_json=[
                json.dumps(
                    {"name": result.name, "result_json": result.result_json},
                    sort_keys=True,
                    ensure_ascii=True,
                )
                for result in function_call_results
            ],
            thinking_text=thinking_text,
        )
        response_context_id = context_hash(
            content_hash=response_content_id,
            model_name=self._model_name,
            function_schema_hashes=self._function_schema_hashes(enabled_function_refs),
            rag_provenance=[],
        )
        response_message = ResponseMessage(
            context_id=response_context_id,
            content_id=response_content_id,
            model=ModelName(value=self._model_name),
            in_response_to=prompt_context_id,
            text=visible_text,
            function_calls=function_call_requests,
            function_results=function_call_results,
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

    def append_history_prompt(self, text: str) -> PromptMessage:
        included_context_ids = [cid.md5 for cid in self._session.message_ids]
        augmented_prompt = self._macro_expander.expand(text)
        prompt_content_id = content_hash_for_prompt(
            original_prompt=text,
            augmented_prompt=augmented_prompt,
            enabled_functions=[],
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
            original_prompt=text,
            augmented_prompt=augmented_prompt,
            included_context=[ContextHashID(md5=cid) for cid in included_context_ids],
            enabled_functions=[],
        )
        self._message_store.upsert(prompt_message)
        self._session = self._session_store.append_prompt(self._session, prompt_context_id)
        return prompt_message

    def expand_prompt_macros(self, text: str) -> str:
        return self._macro_expander.expand(text)

    def delete_last_message(self) -> bool:
        if not self._session.turns:
            return False

        turns = list(self._session.turns)
        last_turn = turns[-1]
        if last_turn.response_id is not None:
            self._message_store.delete(last_turn.response_id)
            turns[-1] = last_turn.model_copy(update={"response_id": None})
            message_ids = [cid for cid in self._session.message_ids if cid.md5 != last_turn.response_id.md5]
            self._session = self._session.model_copy(update={"turns": turns, "message_ids": message_ids})
            self._session_store.save(self._session)
            return True

        self._message_store.delete(last_turn.prompt_id)
        self._session, _ = self._session_store.remove_last_turn(self._session)
        return True

    def generate_again(
        self, on_visible_token: Callable[[str], None] | None = None
    ) -> RuntimeResponse | None:
        last_prompt = self.last_prompt_message()
        if last_prompt is None:
            return None
        if not self.delete_last_message():
            return None
        # If the last message was a response, remove the paired prompt too before regenerating.
        current_last_prompt = self.last_prompt_message()
        if current_last_prompt is not None and current_last_prompt.context_id.md5 == last_prompt.context_id.md5:
            if not self.delete_last_message():
                return None
        return self.send_prompt(
            last_prompt.original_prompt,
            enabled_functions=[function.name for function in last_prompt.enabled_functions],
            on_visible_token=on_visible_token,
        )
