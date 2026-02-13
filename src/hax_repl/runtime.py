from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Callable, Literal, Sequence

from hax_repl.agents import AgentPlugin, AgentRunState, DefaultInteractiveAgent
from hax_repl.functions import (BuiltinFunctionProvider, FunctionProvider,
                                FunctionRegistry)
from hax_repl.hashing import (content_hash_for_prompt,
                              content_hash_for_response, context_hash,
                              split_thinking_blocks)
from hax_repl.llm_client import (ChatCompletionResult, ChatMessage,
                                 OpenRouterClient, ToolCall)
from hax_repl.macro import MacroExpander, MacroExpansionResult
from hax_repl.mcp import (DescriptorMcpLoader, LocalClassMcpClientAdapter,
                          McpClient, RegisteredMcpClient)
from hax_repl.message_store import MessageStore
from hax_repl.models import (AnyMessage, ContextHashID, EnabledFunction,
                             FunctionCallRequest, FunctionCallResult,
                             ModelName, PluginAgentMeta, PluginFunctionMeta,
                             PluginMCPMeta, PluginRagMeta, PromptMessage,
                             ResponseMessage, RoleName, SessionFile,
                             SessionName)
from hax_repl.plugin_system import (LoadedPlugin,
                                    load_plugins_from_config_or_fail)
from hax_repl.rag import RagRegistry, RagResult
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


@dataclass(frozen=True)
class FunctionCallDecision:
    action: Literal["approve", "reject", "manual"] = "approve"
    manual_result_json: str | None = None
    rejection_reason: str = ""


class AppRuntime:

    def __init__(
        self,
        session_name: str | None,
        model_name: str = "anthropic/claude-sonnet-4.5",
        plugins_config_path: str | None = None,
    ) -> None:
        app_dir = Path.home() / ".local" / "share" / "haxllm"
        sessions_dir = app_dir / "sessions"
        db_path = app_dir / "messages.sqlite3"
        repo_root = Path(__file__).resolve().parents[2]
        config_path = (Path(plugins_config_path).expanduser().resolve()
                       if plugins_config_path is not None else
                       (repo_root / "hax_repl.plugins.json"))

        self._session_store = SessionStore(sessions_dir)
        self._message_store = MessageStore(db_path)
        self._model_name = model_name
        self._client = OpenRouterClient(model_name=model_name)
        self._loaded_plugins = load_plugins_from_config_or_fail(
            config_path=config_path,
            interpolation_vars={"repo": str(repo_root)},
        )
        self._macro_expander = MacroExpander()
        self._rag_registry = RagRegistry()
        self._register_rag_plugins()
        self._function_registry = self._build_function_registry()
        self._mcp_loader = DescriptorMcpLoader()
        self._mcp_clients: dict[str, RegisteredMcpClient] = {}
        self._register_mcp_plugins()
        self._agent_plugins = self._build_agent_registry()
        self._active_agent_run: AgentRunState | None = None

        resolved_name: SessionName = self._session_store.resolve_session_name(
            session_name)
        self._session: SessionFile = self._session_store.load_or_create(
            resolved_name)
        self._default_role = RoleName(value="user")
        self._default_agent = "default-agent"

    def _loaded_plugins_for(self, kind: str) -> list[LoadedPlugin]:
        return [
            plugin for plugin in self._loaded_plugins if plugin.kind == kind
        ]

    def _build_function_registry(self) -> FunctionRegistry:
        registry = FunctionRegistry()
        builtin_provider: FunctionProvider = BuiltinFunctionProvider()
        for function_spec in builtin_provider.functions():
            registry.register(function_spec)

        for loaded in self._loaded_plugins_for("function_provider"):
            if not isinstance(loaded.metadata, PluginFunctionMeta):
                raise RuntimeError(
                    f"Invalid function plugin metadata in {loaded.path}")
            provider_candidate = loaded.description.get_plugin()
            if not hasattr(provider_candidate, "functions"):
                raise RuntimeError(
                    f"Invalid function provider plugin: {loaded.name}")
            provider = provider_candidate
            functions = provider.functions()
            for function_spec in functions:
                registry.register(function_spec)
        return registry

    def _register_rag_plugins(self) -> None:
        for loaded in self._loaded_plugins_for("rag_provider"):
            if not isinstance(loaded.metadata, PluginRagMeta):
                raise RuntimeError(
                    f"Invalid RAG plugin metadata in {loaded.path}")
            provider_candidate = loaded.description.get_plugin()
            provider_obj = provider_candidate
            if not hasattr(provider_obj, "list_indices") or not hasattr(
                    provider_obj, "query"):
                raise RuntimeError(
                    f"Invalid RAG provider plugin: {loaded.name}")
            self._rag_registry.register(loaded.name, provider_obj)

    def _build_agent_registry(self) -> dict[str, AgentPlugin]:
        agents: dict[str, AgentPlugin] = {}
        default_agent = DefaultInteractiveAgent()
        agents[default_agent.agent_name()] = default_agent
        for loaded in self._loaded_plugins_for("agent"):
            if not isinstance(loaded.metadata, PluginAgentMeta):
                raise RuntimeError(
                    f"Invalid agent plugin metadata in {loaded.path}")
            candidate = loaded.description.get_plugin()
            plugin_obj = candidate
            if not hasattr(plugin_obj, "agent_name") or not hasattr(
                    plugin_obj, "build_step_prompt"):
                raise RuntimeError(f"Invalid agent plugin: {loaded.name}")
            name = plugin_obj.agent_name()
            agents[name] = plugin_obj
        return agents

    def _register_mcp_plugins(self) -> None:
        for loaded in self._loaded_plugins_for("mcp_client"):
            if not isinstance(loaded.metadata, PluginMCPMeta):
                raise RuntimeError(
                    f"Invalid MCP plugin metadata in {loaded.path}")
            candidate = loaded.description.get_plugin()
            client_obj = candidate
            if hasattr(client_obj, "list_tools") and hasattr(
                    client_obj, "invoke") and hasattr(client_obj,
                                                      "client_name"):
                self._register_mcp_client(client_obj)
                continue
            if isinstance(client_obj, LocalClassMcpClientAdapter):
                self._register_mcp_client(client_obj)
                continue
            raise RuntimeError(
                f"Invalid MCP plugin: {loaded.name}, plugin is expected to have methods list_tools() and invoke(), but {type(client_obj)} got methods {', '.join(dir(client_obj))}"
            )

    def _register_mcp_client(self, client: McpClient) -> RegisteredMcpClient:
        function_names: list[str] = []
        for tool_spec in client.list_tools():
            self._function_registry.register(tool_spec)
            function_names.append(tool_spec.name)
        registered = RegisteredMcpClient(name=client.client_name(),
                                         client=client,
                                         function_names=function_names)
        self._mcp_clients[registered.name] = registered
        return registered

    @property
    def session(self) -> SessionFile:
        return self._session

    @property
    def prompt_state_label(self) -> str:
        active_agent = self._active_agent_run.agent_name if self._active_agent_run is not None else self._default_agent
        return f"{self._session.session.value}/{active_agent}|{self._default_role.value})"

    @property
    def model_name(self) -> str:
        return self._model_name

    def next_query_index(self) -> int:
        return len(self._session.turns) + 1

    def _message_for_context_id(
            self, context_id: ContextHashID) -> AnyMessage | None:
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
                messages.append(
                    ChatMessage(role=prompt.role.value,
                                content=prompt.augmented_prompt))
            if turn.response_id is not None:
                response = self._message_store.get(turn.response_id)
                if isinstance(response, ResponseMessage):
                    messages.append(
                        ChatMessage(role="assistant", content=response.text))
        return messages

    def _enabled_function_refs(
            self, enabled_function_names: Sequence[str] | None
    ) -> list[EnabledFunction]:
        specs = self._function_registry.enabled_specs(
            list(enabled_function_names) if enabled_function_names else None)
        return [
            EnabledFunction(
                name=spec.name,
                schema_hash=self._function_registry.schema_hash_for(spec))
            for spec in specs
        ]

    def _function_schema_hashes(self,
                                refs: Sequence[EnabledFunction]) -> list[str]:
        return [ref.schema_hash for ref in refs]

    def list_functions(self) -> list[EnabledFunction]:
        return [
            EnabledFunction(name=name, schema_hash=schema_hash) for name,
            schema_hash in self._function_registry.list_with_schema_hashes()
        ]

    def invoke_function(self, function_name: str, arguments_json: str) -> str:
        return self._function_registry.invoke_json(function_name,
                                                   arguments_json)

    def _expand_prompt_with_macros(
            self, original_prompt: str) -> MacroExpansionResult:
        return self._macro_expander.expand(
            original_prompt,
            rag_query=lambda provider, index, query: self.rag_query(
                provider, index, query),
        )

    def list_mcp_clients(self) -> list[RegisteredMcpClient]:
        return [
            self._mcp_clients[name]
            for name in sorted(self._mcp_clients.keys())
        ]

    def list_rag_providers(self) -> list[str]:
        return self._rag_registry.list_providers()

    def list_rag_indices(self, provider_name: str) -> Sequence[str]:
        return self._rag_registry.list_indices(provider_name)

    def rag_query(self,
                  provider_name: str,
                  index_name: str,
                  query_text: str,
                  options_json: str = "{}") -> RagResult:
        return self._rag_registry.query(
            provider_name=provider_name,
            index_name=index_name,
            query_text=query_text,
            options_json=options_json,
        )

    def rag_update(
        self,
        provider_name: str,
        index_name: str,
        sources: Sequence[str],
        options_json: str = "{}",
    ) -> None:
        self._rag_registry.update_index(
            provider_name=provider_name,
            index_name=index_name,
            sources=sources,
            options_json=options_json,
        )

    def load_mcp_descriptor(self, descriptor_path: str) -> RegisteredMcpClient:
        adapter = self._mcp_loader.load(Path(descriptor_path).expanduser())
        return self._register_mcp_client(adapter)

    def invoke_mcp_tool(self, client_name: str, tool_name: str,
                        arguments_json: str) -> str:
        if client_name not in self._mcp_clients:
            raise KeyError(f"Unknown MCP client: {client_name}")
        return self._mcp_clients[client_name].client.invoke(
            tool_name, arguments_json)

    def list_agents(self) -> list[str]:
        return sorted(self._agent_plugins.keys())

    def start_agent_run(self,
                        *,
                        agent_name: str,
                        goal: str,
                        max_steps: int = 8) -> AgentRunState:
        if agent_name not in self._agent_plugins:
            raise KeyError(f"Unknown agent: {agent_name}")
        state = AgentRunState(agent_name=agent_name,
                              goal=goal,
                              max_steps=max_steps)
        self._active_agent_run = state
        return state

    def agent_status(self) -> AgentRunState | None:
        return self._active_agent_run

    def pause_agent(self) -> bool:
        if self._active_agent_run is None:
            return False
        self._active_agent_run.paused = True
        return True

    def resume_agent(self) -> bool:
        if self._active_agent_run is None:
            return False
        self._active_agent_run.paused = False
        return True

    def clear_agent_run(self) -> None:
        self._active_agent_run = None

    def run_agent_step(
        self,
        *,
        on_visible_token: Callable[[str], None] | None = None,
        on_function_call_decision: Callable[[FunctionCallRequest],
                                            FunctionCallDecision]
        | None = None,
    ) -> RuntimeResponse | None:
        state = self._active_agent_run
        if state is None or state.done or state.paused:
            return None
        plugin = self._agent_plugins[state.agent_name]
        prompt = plugin.build_step_prompt(
            goal=state.goal,
            step_index=state.step_index,
            step_history=state.step_history,
        )
        response = self.send_prompt(
            prompt,
            enabled_functions=None,
            on_visible_token=on_visible_token,
            on_function_call_decision=on_function_call_decision,
        )
        state.step_history.append(response.response_message.text)
        state.last_response = response.response_message.text
        state.done = plugin.should_stop(
            response_text=response.response_message.text,
            step_index=state.step_index,
            max_steps=state.max_steps,
        )
        state.step_index += 1
        return response

    def _run_function_calling_loop(
        self,
        *,
        llm_messages: list[ChatMessage],
        enabled_function_names: list[str],
        on_function_call_decision: Callable[[FunctionCallRequest],
                                            FunctionCallDecision]
        | None = None,
    ) -> tuple[ChatCompletionResult, list[FunctionCallRequest],
               list[FunctionCallResult]]:
        messages = list(llm_messages)
        tool_specs = self._function_registry.to_tool_specs(
            enabled_function_names)
        all_requests: list[FunctionCallRequest] = []
        all_results: list[FunctionCallResult] = []
        max_rounds = 6
        last_result = ChatCompletionResult(text="", tool_calls=[])

        for _ in range(max_rounds):
            completion = self._client.complete_chat_with_tools(
                messages=messages, tool_specs=tool_specs)
            last_result = completion
            if not completion.tool_calls:
                return completion, all_requests, all_results

            messages.append(
                ChatMessage(
                    role="assistant",
                    content=completion.text,
                    tool_calls=completion.tool_calls,
                ))
            for tool_call in completion.tool_calls:
                request = FunctionCallRequest(
                    name=tool_call.name,
                    arguments_json=tool_call.arguments_json)
                all_requests.append(request)
                decision = (on_function_call_decision(request)
                            if on_function_call_decision is not None else
                            FunctionCallDecision(action="approve"))
                result_json = self._resolve_tool_call_result(
                    tool_call=tool_call, decision=decision)
                all_results.append(
                    FunctionCallResult(name=tool_call.name,
                                       result_json=result_json))
                messages.append(
                    ChatMessage(
                        role="tool",
                        content=result_json,
                        name=tool_call.name,
                        tool_call_id=tool_call.id,
                    ))

        return last_result, all_requests, all_results

    def _resolve_tool_call_result(self, *, tool_call: ToolCall,
                                  decision: FunctionCallDecision) -> str:
        if decision.action == "manual":
            return self._normalize_manual_result(decision.manual_result_json)
        if decision.action == "reject":
            return json.dumps(
                {
                    "ok":
                    False,
                    "rejected":
                    True,
                    "tool_name":
                    tool_call.name,
                    "reason":
                    decision.rejection_reason.strip() or "Rejected by user.",
                },
                ensure_ascii=True,
            )
        return self._invoke_tool_with_failure_payload(tool_call)

    def _normalize_manual_result(self, manual_result_json: str | None) -> str:
        text = (manual_result_json or "").strip()
        if not text:
            return json.dumps({
                "ok": True,
                "manual": True,
                "result": None
            },
                              ensure_ascii=True)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = {"ok": True, "manual": True, "result": text}
        return json.dumps(payload, ensure_ascii=True)

    def _invoke_tool_with_failure_payload(self, tool_call: ToolCall) -> str:
        try:
            return self._function_registry.invoke_json(
                tool_call.name, tool_call.arguments_json)
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
        on_function_call_decision: Callable[[FunctionCallRequest],
                                            FunctionCallDecision]
        | None = None,
    ) -> RuntimeResponse:
        enabled_function_refs = self._enabled_function_refs(enabled_functions)
        included_context_ids = [cid.md5 for cid in self._session.message_ids]
        macro_expansion = self._expand_prompt_with_macros(original_prompt)
        augmented_prompt = macro_expansion.expanded_text
        rag_provenance = [
            f"{record.provider}:{record.index}:{len(record.chunks)}"
            for record in macro_expansion.rag_records
        ]

        prompt_content_id = content_hash_for_prompt(
            original_prompt=original_prompt,
            augmented_prompt=augmented_prompt,
            enabled_functions=enabled_function_refs,
            included_context_ids=included_context_ids,
        )
        prompt_context_id = context_hash(
            content_hash=prompt_content_id,
            model_name=self._model_name,
            function_schema_hashes=self._function_schema_hashes(
                enabled_function_refs),
            rag_provenance=rag_provenance,
        )
        prompt_message = PromptMessage(
            context_id=prompt_context_id,
            content_id=prompt_content_id,
            role=self._default_role,
            original_prompt=original_prompt,
            augmented_prompt=augmented_prompt,
            included_context=[
                ContextHashID(md5=cid) for cid in included_context_ids
            ],
            enabled_functions=enabled_function_refs,
        )

        llm_messages = self._conversation_messages()
        llm_messages.append(ChatMessage(role="user", content=augmented_prompt))
        query_chars = sum(len(message.content) for message in llm_messages)

        self._message_store.upsert(prompt_message)
        self._session = self._session_store.append_prompt(
            self._session, prompt_context_id)

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
                on_function_call_decision=on_function_call_decision,
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
                visible_delta = split.visible_text[len(visible_so_far):]
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
                    {
                        "name": call.name,
                        "arguments_json": call.arguments_json
                    },
                    sort_keys=True,
                    ensure_ascii=True,
                ) for call in function_call_requests
            ],
            function_results_json=[
                json.dumps(
                    {
                        "name": result.name,
                        "result_json": result.result_json
                    },
                    sort_keys=True,
                    ensure_ascii=True,
                ) for result in function_call_results
            ],
            thinking_text=thinking_text,
        )
        response_context_id = context_hash(
            content_hash=response_content_id,
            model_name=self._model_name,
            function_schema_hashes=self._function_schema_hashes(
                enabled_function_refs),
            rag_provenance=rag_provenance,
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
        self._session = self._session_store.attach_response_to_last_turn(
            self._session, response_context_id)

        completed_at = monotonic()
        elapsed_ms = int((completed_at - started) * 1000)
        time_until_first_token_ms = (int(
            (first_token_at - started) *
            1000) if first_token_at is not None else elapsed_ms)
        if first_token_at is None:
            model_thinking_ms = 0
        elif first_visible_token_at is not None:
            model_thinking_ms = max(
                int((first_visible_token_at - first_token_at) * 1000), 0)
        else:
            model_thinking_ms = max(
                int((completed_at - first_token_at) * 1000), 0)
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
        return RuntimeResponse(prompt_message=prompt_message,
                               response_message=response_message,
                               stats=stats)

    def append_history_prompt(self, text: str) -> PromptMessage:
        included_context_ids = [cid.md5 for cid in self._session.message_ids]
        macro_expansion = self._expand_prompt_with_macros(text)
        augmented_prompt = macro_expansion.expanded_text
        rag_provenance = [
            f"{record.provider}:{record.index}:{len(record.chunks)}"
            for record in macro_expansion.rag_records
        ]
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
            rag_provenance=rag_provenance,
        )
        prompt_message = PromptMessage(
            context_id=prompt_context_id,
            content_id=prompt_content_id,
            role=self._default_role,
            original_prompt=text,
            augmented_prompt=augmented_prompt,
            included_context=[
                ContextHashID(md5=cid) for cid in included_context_ids
            ],
            enabled_functions=[],
        )
        self._message_store.upsert(prompt_message)
        self._session = self._session_store.append_prompt(
            self._session, prompt_context_id)
        return prompt_message

    def expand_prompt_macros(self, text: str) -> str:
        return self._expand_prompt_with_macros(text).expanded_text

    def delete_last_message(self) -> bool:
        if not self._session.turns:
            return False

        turns = list(self._session.turns)
        last_turn = turns[-1]
        if last_turn.response_id is not None:
            self._message_store.delete(last_turn.response_id)
            turns[-1] = last_turn.model_copy(update={"response_id": None})
            message_ids = [
                cid for cid in self._session.message_ids
                if cid.md5 != last_turn.response_id.md5
            ]
            self._session = self._session.model_copy(update={
                "turns": turns,
                "message_ids": message_ids
            })
            self._session_store.save(self._session)
            return True

        self._message_store.delete(last_turn.prompt_id)
        self._session, _ = self._session_store.remove_last_turn(self._session)
        return True

    def generate_again(
        self,
        on_visible_token: Callable[[str], None] | None = None,
        on_function_call_decision: Callable[[FunctionCallRequest],
                                            FunctionCallDecision]
        | None = None,
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
            enabled_functions=[
                function.name for function in last_prompt.enabled_functions
            ],
            on_visible_token=on_visible_token,
            on_function_call_decision=on_function_call_decision,
        )
