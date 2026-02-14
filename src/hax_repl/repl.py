from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
import textwrap
from typing import cast

import click
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.syntax import Syntax

from hax_repl.models import FunctionCallRequest
from hax_repl.runtime import AppRuntime, FunctionCallDecision, RuntimeResponse


@dataclass
class ReplCommandContext:
    runtime: AppRuntime
    console: Console
    pending_includes: list[str]
    keep_running: bool = True


def _build_prompt_session(command_phrases: list[str]) -> PromptSession[str]:
    bindings = KeyBindings()

    @bindings.add("c-j")
    def _submit_with_ctrl_j(event) -> None:  # type: ignore[no-untyped-def]
        event.current_buffer.validate_and_handle()

    @bindings.add("escape", "enter")
    def _submit_with_alt_enter(event) -> None:  # type: ignore[no-untyped-def]
        event.current_buffer.validate_and_handle()

    # Common CSI-u / modified enter variants emitted by some terminals.
    @bindings.add("escape", "[", "1", "3", ";", "5", "u")
    def _submit_with_csi_u_ctrl_enter(event) -> None:  # type: ignore[no-untyped-def]
        event.current_buffer.validate_and_handle()

    @bindings.add("escape", "[", "2", "7", ";", "5", ";", "1", "3", "~")
    def _submit_with_legacy_ctrl_enter(event) -> None:  # type: ignore[no-untyped-def]
        event.current_buffer.validate_and_handle()

    @bindings.add("c-o")
    def _edit_in_editor(event) -> None:
        buffer = event.current_buffer
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(buffer.text)
            temp_path = f.name

        editor = os.environ.get("EDITOR", "vim")
        subprocess.call([editor, temp_path])

        with open(temp_path, "r") as f:
            buffer.text = f.read()

        os.unlink(temp_path)
        buffer.cursor_position = len(buffer.text)

    completer = WordCompleter(command_phrases, ignore_case=True, sentence=True)
    return PromptSession(
        completer=completer,
        complete_while_typing=False,
        key_bindings=bindings,
    )


def _parse_command(raw: str) -> list[str]:
    try:
        return shlex.split(raw)
    except ValueError:
        return raw.split()


def _build_repl_click_group() -> click.Group:

    @click.group(help="Interactive REPL command namespace.")
    def repl_commands() -> None:
        """Root command group for REPL dot-commands."""

    @repl_commands.command(name="help", help="Show help for commands and sub-commands.")
    @click.argument("topic", nargs=-1)
    @click.pass_context
    def help_command(ctx: click.Context, topic: tuple[str, ...]) -> None:
        """Render Click-generated help text for the full command tree or a specific topic."""
        state = cast(ReplCommandContext, ctx.obj)
        root = cast(click.Group, ctx.find_root().command)
        target: click.Command = root
        target_info_name = "."
        target_parent: click.Context | None = None
        for name in topic:
            if not isinstance(target, click.Group):
                raise click.UsageError(f"Command {' '.join(topic)} has no sub-commands.",
                                       ctx=ctx)
            next_ctx = click.Context(target,
                                     info_name=target_info_name,
                                     parent=target_parent)
            resolved = target.get_command(next_ctx, name)
            if resolved is None:
                raise click.UsageError(f"No such command: {name}", ctx=ctx)
            target_parent = next_ctx
            target = resolved
            if target_info_name == ".":
                target_info_name = f".{name}"
            else:
                target_info_name = f"{target_info_name} {name}"
        help_ctx = click.Context(target, info_name=target_info_name, parent=target_parent)
        state.console.print(help_ctx.get_help(), markup=False)

    @repl_commands.command(name="exit", help="Exit the REPL.")
    @click.pass_obj
    def exit_command(state: ReplCommandContext) -> None:
        """Stop the current REPL session."""
        state.keep_running = False

    @repl_commands.command(name="quit", help="Exit the REPL (alias for exit).")
    @click.pass_obj
    def quit_command(state: ReplCommandContext) -> None:
        """Stop the current REPL session."""
        state.keep_running = False

    @repl_commands.group(help="Conversation history management commands.")
    def conversation() -> None:
        """Inspect and mutate conversation state."""

    @conversation.command(name="delete-last-message",
                          help="Delete the most recent message in session history.")
    @click.pass_obj
    def conversation_delete_last_message(state: ReplCommandContext) -> None:
        """Remove the latest persisted prompt/response from the active session."""
        if state.runtime.delete_last_message():
            state.console.print("[green]Removed last message from session.[/green]")
        else:
            state.console.print("[yellow]No messages to delete.[/yellow]")

    @conversation.command(
        name="generate-again",
        help="Re-run the previous prompt and stream a fresh response.",
    )
    @click.pass_obj
    def conversation_generate_again(state: ReplCommandContext) -> None:
        """Regenerate the previous turn using the same runtime/tool-calling flow."""
        query_index = state.runtime.next_query_index()
        first_visible_token_seen = False
        try:
            with state.console.status("[yellow]thinking...[/yellow]",
                                      spinner="dots") as status:

                def _on_visible_token(token: str) -> None:
                    nonlocal first_visible_token_seen
                    if not first_visible_token_seen:
                        first_visible_token_seen = True
                        status.stop()
                        state.console.print(f"[red]RESULT [{query_index}]:[/red]")

                runtime_response = state.runtime.generate_again(
                    on_visible_token=_on_visible_token,
                    on_function_call_decision=lambda request:
                    _interactive_function_call_decision(
                        console=state.console,
                        status=status,
                        request=request,
                    ),
                )

        except Exception as exc:
            state.console.print(f"[bold red]generate-again failed:[/bold red] {exc}")
            return

        if runtime_response is None:
            state.console.print("[yellow]No previous prompt to regenerate.[/yellow]")

            return
        if not first_visible_token_seen:
            state.console.print(f"[red]RESULT [{query_index}]:[/red]")

        _render_runtime_response(
            runtime=state.runtime,
            console=state.console,
            query_index=query_index,
            prompt_text=runtime_response.prompt_message.original_prompt,
            runtime_response=runtime_response,
        )

    @repl_commands.group(help="Session history manipulation commands.")
    def history() -> None:
        """Modify session history without sending a model request."""

    @history.command(name="append",
                     help="Append text to session history as a user prompt.")
    @click.argument("text_tokens", nargs=-1)
    @click.pass_context
    def history_append(ctx: click.Context, text_tokens: tuple[str, ...]) -> None:
        """Append a synthetic user prompt to persisted history."""
        if not text_tokens:
            raise click.UsageError("Usage: .history append <text>", ctx=ctx)
        state = cast(ReplCommandContext, ctx.obj)
        text = " ".join(text_tokens).strip()
        if not text:
            raise click.UsageError("Usage: .history append <text>", ctx=ctx)
        state.runtime.append_history_prompt(text)
        state.console.print(
            "[green]Appended to history as user prompt (not sent).[/green]")

    @repl_commands.group(help="Display helper commands.")
    def show() -> None:
        """Show hidden or derived information from prior responses."""

    @show.command(name="last-thinking",
                  help="Show the hidden thinking block from the last response.")
    @click.pass_obj
    def show_last_thinking(state: ReplCommandContext) -> None:
        """Display the parsed think-tag content from the latest response, if present."""
        last_response = state.runtime.last_response_message()
        if last_response is None:
            state.console.print("[yellow]No response found.[/yellow]")
            return
        thinking = last_response.thinking_text.strip()
        if not thinking:
            state.console.print("[dim](no thinking content in last response)[/dim]")
            return
        state.console.print("[bold]Last thinking:[/bold]")
        state.console.print(thinking)

    @repl_commands.group(help="Copy response snippets to the system clipboard.")
    def copy() -> None:
        """Clipboard convenience commands for the last model response."""

    @copy.command(name="last-code",
                  help="Copy the last fenced code block from the latest response.")
    @click.pass_obj
    def copy_last_code(state: ReplCommandContext) -> None:
        """Extract the most recent fenced markdown code block and copy it."""
        last_response = state.runtime.last_response_message()
        if last_response is None:
            state.console.print("[yellow]No response found.[/yellow]")
            return
        last_code = _extract_last_code_block(last_response.text)
        if not last_code:
            state.console.print(
                "[yellow]No fenced code block found in last response.[/yellow]")
            return
        ok, message = _copy_to_clipboard(last_code)
        state.console.print(
            f"[green]{message}[/green]" if ok else f"[yellow]{message}[/yellow]")

    @copy.command(name="last-response", help="Copy the full text of the latest response.")
    @click.pass_obj
    def copy_last_response(state: ReplCommandContext) -> None:
        """Copy the complete latest assistant response body."""
        last_response = state.runtime.last_response_message()
        if last_response is None:
            state.console.print("[yellow]No response found.[/yellow]")
            return
        ok, message = _copy_to_clipboard(last_response.text)
        state.console.print(
            f"[green]{message}[/green]" if ok else f"[yellow]{message}[/yellow]")

    @repl_commands.command(name="file",
                           help="Queue a file payload for inclusion in the next prompt.")
    @click.argument("path_text")
    @click.pass_obj
    def file_command(state: ReplCommandContext, path_text: str) -> None:
        """Read a local file and inject it into the next prompt as an XML-like payload."""
        candidate = Path(path_text).expanduser()
        file_path = candidate if candidate.is_absolute() else (Path.cwd() / candidate)
        if not file_path.exists() or not file_path.is_file():
            state.console.print(f"[yellow]File not found: {file_path}[/yellow]")
            return
        file_text = file_path.read_text(encoding="utf-8", errors="replace")
        payload = f'<file path="{file_path}">\n{file_text}\n</file>'
        state.pending_includes.append(payload)
        state.console.print(f"[green]Queued file for next query:[/green] {file_path}")

    @repl_commands.command(name="macro",
                           help="Expand macro text and queue it for the next prompt.")
    @click.argument("text_tokens", nargs=-1)
    @click.pass_context
    def macro_command(ctx: click.Context, text_tokens: tuple[str, ...]) -> None:
        """Expand registered prompt macros and stage the result for the next query."""
        if not text_tokens:
            raise click.UsageError("Usage: .macro <text-with-macros>", ctx=ctx)
        state = cast(ReplCommandContext, ctx.obj)
        raw_body = " ".join(text_tokens).strip()
        expanded = state.runtime.expand_prompt_macros(raw_body)
        state.pending_includes.append(expanded)
        state.console.print("[green]Expanded macro text queued for next query.[/green]")

    @repl_commands.group(
        help="Inspect loaded functions and invoke tools manually.",
        invoke_without_command=True,
    )
    @click.pass_context
    def functions(ctx: click.Context) -> None:
        """Function provider commands."""

    @functions.command(name="loaded", help="List loaded functions with schema hashes.")
    @click.pass_obj
    def functions_loaded(state: ReplCommandContext) -> None:
        """Display all currently loaded function tools."""
        loaded = state.runtime.list_enabled_functions()
        if not loaded:
            state.console.print("[yellow]No functions loaded.[/yellow]")
            return
        state.console.print("[bold]Loaded functions:[/bold]")
        for function in loaded:
            state.console.print(f"  - {function.name} {function}")

    @functions.command(name="list", help="List loaded functions with schema hashes.")
    @click.pass_obj
    def functions_list(state: ReplCommandContext) -> None:
        """Display all currently loaded function tools."""
        loaded = state.runtime._function_registry.all_specs()
        if not loaded:
            state.console.print("[yellow]No functions registered.[/yellow]")
            return

        state.console.print("[bold]Registered functions:[/bold]")
        for function in loaded:
            state.console.print(f"  - {function.name} {function.description}")

    @functions.command(name="call", help="Invoke a loaded function with JSON arguments.")
    @click.argument("name")
    @click.argument("json_tokens", nargs=-1)
    @click.pass_context
    def functions_call(ctx: click.Context, name: str, json_tokens: tuple[str,
                                                                         ...]) -> None:
        """Call a named function directly from the REPL."""
        if not json_tokens:
            raise click.UsageError("Usage: .functions call <name> <json-args>", ctx=ctx)
        state = cast(ReplCommandContext, ctx.obj)
        arguments_json = " ".join(json_tokens).strip()
        try:
            result_json = state.runtime.invoke_function(name, arguments_json)
        except Exception as exc:
            state.console.print(f"[bold red]function call failed:[/bold red] {exc}")
            return
        pretty = result_json
        if len(pretty) > 4000:
            pretty = textwrap.shorten(pretty, width=4000, placeholder=" ...[truncated]")
        state.console.print("[green]Function result:[/green]")
        state.console.print(pretty)

    @repl_commands.group(help="Manage loaded MCP clients and call MCP tools.")
    def mcp() -> None:
        """MCP client commands."""

    @mcp.command(name="list", help="List loaded MCP clients and their exported tools.")
    @click.pass_obj
    def mcp_list(state: ReplCommandContext) -> None:
        """Display registered MCP clients."""
        clients = state.runtime.list_mcp_clients()
        if not clients:
            state.console.print("[yellow]No MCP clients loaded.[/yellow]")
            return
        state.console.print("[bold]Loaded MCP clients:[/bold]")
        for client in clients:
            tools = ", ".join(
                client.function_names) if client.function_names else "(no tools)"
            state.console.print(f"  - {client.name}: {tools}")

    @mcp.command(name="load", help="Load an MCP descriptor JSON file at runtime.")
    @click.argument("descriptor_path")
    @click.pass_obj
    def mcp_load(state: ReplCommandContext, descriptor_path: str) -> None:
        """Load and register one MCP client from a descriptor file."""
        try:
            loaded = state.runtime.load_mcp_descriptor(descriptor_path)
        except Exception as exc:
            state.console.print(
                f"[bold red]Failed to load MCP descriptor:[/bold red] {exc}")
            return
        state.console.print(f"[green]Loaded MCP client:[/green] {loaded.name}")

    @mcp.command(name="call", help="Call an MCP tool with JSON arguments.")
    @click.argument("client_name")
    @click.argument("tool_name")
    @click.argument("json_tokens", nargs=-1)
    @click.pass_context
    def mcp_call(
        ctx: click.Context,
        client_name: str,
        tool_name: str,
        json_tokens: tuple[str, ...],
    ) -> None:
        """Invoke one MCP client tool directly from the REPL."""
        if not json_tokens:
            raise click.UsageError(
                "Usage: .mcp call <client-name> <tool-name> <json-args>", ctx=ctx)
        state = cast(ReplCommandContext, ctx.obj)
        arguments_json = " ".join(json_tokens).strip()
        try:
            result_json = state.runtime.invoke_mcp_tool(client_name, tool_name,
                                                        arguments_json)
        except Exception as exc:
            state.console.print(f"[bold red]MCP call failed:[/bold red] {exc}")
            return
        state.console.print("[green]MCP result:[/green]")
        state.console.print(result_json)

    @repl_commands.group(help="RAG provider/index commands.")
    def rag() -> None:
        """Query and update retrieval indexes."""

    @rag.command(name="providers", help="List loaded RAG providers.")
    @click.pass_obj
    def rag_providers(state: ReplCommandContext) -> None:
        """Print all registered RAG provider names."""
        providers = state.runtime.list_rag_providers()
        if not providers:
            state.console.print("[yellow]No RAG providers loaded.[/yellow]")
            return
        state.console.print("[bold]RAG providers:[/bold]")
        for provider in providers:
            state.console.print(f"  - {provider}")

    @rag.command(name="indices", help="List indices for a provider.")
    @click.argument("provider")
    @click.pass_obj
    def rag_indices(state: ReplCommandContext, provider: str) -> None:
        """List known index names for one provider."""
        try:
            indices = state.runtime.list_rag_indices(provider)
        except Exception as exc:
            state.console.print(f"[bold red]Failed to list indices:[/bold red] {exc}")
            return
        state.console.print(f"[bold]Indices for {provider}:[/bold]")
        if not indices:
            state.console.print("  (none)")
            return
        for index in indices:
            state.console.print(f"  - {index}")

    @rag.command(name="update",
                 help="Update an index from one or more source files/directories.")
    @click.argument("provider")
    @click.argument("index_name")
    @click.argument("sources", nargs=-1)
    @click.pass_context
    def rag_update(
        ctx: click.Context,
        provider: str,
        index_name: str,
        sources: tuple[str, ...],
    ) -> None:
        """Rebuild or incrementally update one RAG index from source paths."""
        if not sources:
            raise click.UsageError(
                "Usage: .rag update <provider> <index> <path1> [path2 ...]", ctx=ctx)
        state = cast(ReplCommandContext, ctx.obj)
        try:
            state.runtime.rag_update(provider, index_name, list(sources))
        except Exception as exc:
            state.console.print(f"[bold red]RAG update failed:[/bold red] {exc}")
            return
        state.console.print(
            f"[green]Updated RAG index {index_name} on {provider} with {len(sources)} source(s).[/green]"
        )

    @rag.command(name="query", help="Query an index and print scored chunks.")
    @click.argument("provider")
    @click.argument("index_name")
    @click.argument("query_tokens", nargs=-1)
    @click.pass_context
    def rag_query(
        ctx: click.Context,
        provider: str,
        index_name: str,
        query_tokens: tuple[str, ...],
    ) -> None:
        """Run a retrieval query against a provider/index pair."""
        if not query_tokens:
            raise click.UsageError("Usage: .rag query <provider> <index> <query>",
                                   ctx=ctx)
        state = cast(ReplCommandContext, ctx.obj)
        query_text = " ".join(query_tokens).strip()
        try:
            result = state.runtime.rag_query(provider, index_name, query_text)
        except Exception as exc:
            state.console.print(f"[bold red]RAG query failed:[/bold red] {exc}")
            return
        state.console.print(f"[bold]RAG {provider}/{index_name} results:[/bold]")
        if not result.chunks:
            state.console.print("  (no results)")
            return
        for chunk in result.chunks:
            state.console.print(f"- [{chunk.source_id}] score={chunk.score:.4f}")
            state.console.print(
                textwrap.shorten(chunk.text.replace("\n", " "),
                                 width=220,
                                 placeholder=" ..."))

    @repl_commands.group(help="Agent orchestration and execution controls.")
    def agent() -> None:
        """Manage one active agent run and its execution state."""

    @agent.command(name="list", help="List available agent plugin names.")
    @click.pass_obj
    def agent_list(state: ReplCommandContext) -> None:
        """Show all registered agents."""
        agents = state.runtime.list_agents()
        state.console.print("[bold]Available agents:[/bold]")
        for name in agents:
            state.console.print(f"  - {name}")

    @agent.command(name="start", help="Start an agent run with a goal.")
    @click.argument("agent_name")
    @click.argument("goal_tokens", nargs=-1)
    @click.pass_context
    def agent_start(ctx: click.Context, agent_name: str, goal_tokens: tuple[str,
                                                                            ...]) -> None:
        """Initialize a new agent run and print current status."""
        if not goal_tokens:
            raise click.UsageError("Usage: .agent start <agent-name> <goal>", ctx=ctx)
        state = cast(ReplCommandContext, ctx.obj)
        goal = " ".join(goal_tokens).strip()
        try:
            state.runtime.start_agent_run(agent_name=agent_name, goal=goal, max_steps=8)
        except Exception as exc:
            state.console.print(f"[bold red]Failed to start agent:[/bold red] {exc}")
            return
        state.console.print(f"[green]Started agent run:[/green] {agent_name}")
        _print_agent_status(state.runtime, state.console)

    @agent.command(name="status", help="Show current active agent state.")
    @click.pass_obj
    def agent_status(state: ReplCommandContext) -> None:
        """Print status for the current agent run."""
        _print_agent_status(state.runtime, state.console)

    @agent.command(name="pause", help="Pause the active agent run.")
    @click.pass_obj
    def agent_pause(state: ReplCommandContext) -> None:
        """Pause the active run, if any."""
        if state.runtime.pause_agent():
            state.console.print("[green]Agent paused.[/green]")
        else:
            state.console.print("[yellow]No active agent run.[/yellow]")

    @agent.command(name="resume", help="Resume a paused active agent run.")
    @click.pass_obj
    def agent_resume(state: ReplCommandContext) -> None:
        """Resume the current paused agent run."""
        if state.runtime.resume_agent():
            state.console.print("[green]Agent resumed.[/green]")
        else:
            state.console.print("[yellow]No active agent run.[/yellow]")

    @agent.command(name="step", help="Execute one step of the active agent run.")
    @click.pass_obj
    def agent_step(state: ReplCommandContext) -> None:
        """Run exactly one agent step through the streaming/tool loop."""
        state.keep_running = _run_agent_step(state.runtime, state.console)

    @agent.command(name="run", help="Execute multiple agent steps (default: 3).")
    @click.argument("steps", required=False, type=int)
    @click.pass_context
    def agent_run(ctx: click.Context, steps: int | None) -> None:
        """Run the agent for N steps or until paused/done."""
        state = cast(ReplCommandContext, ctx.obj)
        max_steps = 3 if steps is None else max(1, steps)
        for _ in range(max_steps):
            current = state.runtime.agent_status()
            if current is None or current.paused or current.done:
                break
            if not _run_agent_step(state.runtime, state.console):
                state.keep_running = False
                return
        _print_agent_status(state.runtime, state.console)

    @agent.command(name="stop", help="Clear the active agent run state.")
    @click.pass_obj
    def agent_stop(state: ReplCommandContext) -> None:
        """Clear active agent run metadata and progress."""
        state.runtime.clear_agent_run()
        state.console.print("[green]Cleared active agent run.[/green]")

    return repl_commands


def _build_click_command_phrases(root: click.Group) -> list[str]:
    phrases: set[str] = set()

    def _walk(group: click.Group, prefix: tuple[str, ...],
              parent_ctx: click.Context) -> None:
        for name in group.list_commands(parent_ctx):
            command = group.get_command(parent_ctx, name)
            if command is None:
                continue
            parts = prefix + (name,)
            phrases.add("." + " ".join(parts))
            if isinstance(command, click.Group):
                child_ctx = click.Context(command,
                                          info_name=" ".join(parts),
                                          parent=parent_ctx)
                _walk(command, parts, child_ctx)

    root_ctx = click.Context(root, info_name=".")
    _walk(root, tuple(), root_ctx)
    return sorted(phrases)


REPL_CLICK_GROUP = _build_repl_click_group()
COMMAND_PHRASES = _build_click_command_phrases(REPL_CLICK_GROUP)


def _print_click_exception(exc: click.ClickException, console: Console) -> None:
    console.print(f"[yellow]{exc.format_message()}[/yellow]")
    if exc.ctx is not None:
        console.print(exc.ctx.get_help(), markup=False)


def _extract_last_code_block(markdown_text: str) -> str | None:
    matches = re.findall(r"```[^\n]*\n(.*?)```", markdown_text, flags=re.DOTALL)
    if not matches:
        return None
    return matches[-1].strip()


def _copy_to_clipboard(text: str) -> tuple[bool, str]:
    clipboard_candidates: list[list[str]] = []
    if shutil.which("wl-copy"):
        clipboard_candidates.append(["wl-copy"])
    if shutil.which("xclip"):
        clipboard_candidates.append(["xclip", "-selection", "clipboard"])
    if shutil.which("xsel"):
        clipboard_candidates.append(["xsel", "--clipboard", "--input"])

    if not clipboard_candidates:
        return False, "No clipboard tool found. Install one of: wl-copy, xclip, xsel."

    for cmd in clipboard_candidates:
        result = subprocess.run(cmd,
                                input=text,
                                text=True,
                                capture_output=True,
                                check=False)
        if result.returncode == 0:
            return True, f"Copied to clipboard via {' '.join(cmd)}."
    return False, "Failed to copy to clipboard with available tools."


def _interactive_function_call_decision(
    *,
    console: Console,
    status,
    request: FunctionCallRequest,
) -> FunctionCallDecision:
    status.stop()
    console.print(f"[bold yellow]Function call requested:[/bold yellow] {request.name}")
    try:
        parsed_args = json.loads(request.arguments_json)
        pretty_args = json.dumps(parsed_args, indent=2, sort_keys=True, ensure_ascii=True)
    except json.JSONDecodeError:
        pretty_args = request.arguments_json
    console.print("[dim]arguments:[/dim]")
    console.print(pretty_args)

    answer = console.input(
        "[cyan]Approve call?[/cyan] \\[y]es / \\[n]o / \\[m]anual-result: ").strip(
        ).lower()
    if answer.startswith("n"):
        reason = console.input("[cyan]Reject reason (optional):[/cyan] ").strip()
        status.start()
        return FunctionCallDecision(action="reject", rejection_reason=reason)
    if answer.startswith("m"):
        manual_result = console.input(
            "[cyan]Manual result JSON (or any text):[/cyan] ").strip()
        status.start()
        return FunctionCallDecision(action="manual", manual_result_json=manual_result)
    status.start()
    return FunctionCallDecision(action="approve")


def _render_runtime_response(
    *,
    runtime: AppRuntime,
    console: Console,
    query_index: int,
    prompt_text: str,
    runtime_response: RuntimeResponse,
) -> None:
    "Format statistics about the model and query processing time."
    console.print("[dim]" + " | ".join([
        f"stats: model={runtime.model_name}",
        f"session={runtime.session.session.value}",
        f"prompt_chars={len(prompt_text)}",
        f"total={runtime_response.stats.elapsed_ms} ms",
        f"first_token={runtime_response.stats.time_until_first_token_ms} ms",
        f"thinking={runtime_response.stats.model_thinking_ms} ms",
        f"query_chars={runtime_response.stats.query_chars}",
        f"response_chars={runtime_response.stats.response_chars}",
    ]) + "[/dim]")

    visible_text = runtime_response.response_message.text

    if visible_text:
        console.print(
            Syntax(
                visible_text,
                "markdown",
                word_wrap=False,
                background_color="default",
            ))

    else:
        console.print("[dim](empty response)[/dim]")


def _run_streaming_query(runtime: AppRuntime, console: Console, query_index: int,
                         prompt_text: str) -> bool:
    first_visible_token_seen = False
    try:
        with console.status("[yellow]thinking...[/yellow]", spinner="dots") as status:

            def _on_visible_token(token: str) -> None:
                nonlocal first_visible_token_seen
                if not first_visible_token_seen:
                    first_visible_token_seen = True
                    status.stop()
                    console.print(f"[red]RESULT [{query_index}]:[/red]")

            runtime_response = runtime.send_prompt(
                prompt_text,
                on_visible_token=_on_visible_token,
                on_function_call_decision=lambda request:
                _interactive_function_call_decision(
                    console=console,
                    status=status,
                    request=request,
                ),
            )

    except KeyboardInterrupt:
        console.print("[yellow]Interrupted. Exiting.[/yellow]")
        return False

    if not first_visible_token_seen:
        console.print(f"[red]RESULT [{query_index}]:[/red]")

    _render_runtime_response(
        runtime=runtime,
        console=console,
        query_index=query_index,
        prompt_text=prompt_text,
        runtime_response=runtime_response,
    )

    return True


def _print_agent_status(runtime: AppRuntime, console: Console) -> None:
    state = runtime.agent_status()
    if state is None:
        console.print("[yellow]No active agent run.[/yellow]")
        return
    console.print(f"[bold]Agent[/bold]: {state.agent_name} | "
                  f"step={state.step_index}/{state.max_steps} | "
                  f"paused={state.paused} | done={state.done}")
    console.print(f"[dim]goal:[/dim] {state.goal}")


def _run_agent_step(runtime: AppRuntime, console: Console) -> bool:
    state = runtime.agent_status()
    if state is None:
        console.print("[yellow]No active agent run. Use .agent start ...[/yellow]")
        return True
    if state.paused:
        console.print("[yellow]Agent is paused. Use .agent resume first.[/yellow]")
        return True
    if state.done:
        console.print("[yellow]Agent run is already done.[/yellow]")
        return True

    query_index = runtime.next_query_index()
    first_visible_token_seen = False
    try:
        with console.status("[yellow]agent thinking...[/yellow]",
                            spinner="dots") as status:

            def _on_visible_token(token: str) -> None:
                nonlocal first_visible_token_seen
                if not first_visible_token_seen:
                    first_visible_token_seen = True
                    status.stop()
                    console.print(f"[red]RESULT [{query_index}]:[/red]")

            runtime_response = runtime.run_agent_step(
                on_visible_token=_on_visible_token,
                on_function_call_decision=lambda request:
                _interactive_function_call_decision(
                    console=console,
                    status=status,
                    request=request,
                ),
            )
    except Exception as exc:
        console.print(f"[bold red]agent step failed:[/bold red] {exc}")
        return True

    if runtime_response is None:
        console.print("[yellow]Agent did not run a step.[/yellow]")
        return True
    if not first_visible_token_seen:
        console.print(f"[red]RESULT [{query_index}]:[/red]")
    _render_runtime_response(
        runtime=runtime,
        console=console,
        query_index=query_index,
        prompt_text=runtime_response.prompt_message.original_prompt,
        runtime_response=runtime_response,
    )
    _print_agent_status(runtime, console)
    return True


def _handle_command(
    *,
    runtime: AppRuntime,
    console: Console,
    command_text: str,
    pending_includes: list[str],
) -> bool:
    command_body = command_text.strip()[1:].strip()
    args = _parse_command(command_body)
    if not args:
        return True
    state = ReplCommandContext(runtime=runtime,
                               console=console,
                               pending_includes=pending_includes)
    try:
        REPL_CLICK_GROUP.main(args=args, prog_name=".", standalone_mode=False, obj=state)
    except click.ClickException as exc:
        _print_click_exception(exc, console)
    except Exception as exc:
        console.print(f"[bold red]command failed:[/bold red] {exc}")
    return state.keep_running


def _end_repl(runtime: AppRuntime, console: Console) -> None:
    console.print("EXITING REPL")


def run_repl(runtime: AppRuntime) -> None:
    console = Console()
    prompt_session = _build_prompt_session(COMMAND_PHRASES)
    pending_includes: list[str] = []

    console.print("[bold]HAX LLM REPL[/bold]  (type .exit to quit)")
    console.print(
        "[dim]Submit: Ctrl+J, Esc+Enter, or Ctrl+Enter (CSI-u terminals). New line: Enter. Complete: Tab.[/dim]"
    )

    # console.print("Registered functions")
    # for f in runtime._function_registry.all_specs():
    #     console.print(f"  {f.name} {f.description}")

    while True:
        query_index = runtime.next_query_index()
        console.print(f"[cyan]{runtime.prompt_state_label}[/cyan]")
        try:
            default_text = ""
            if pending_includes:
                default_text = "\n\n".join(pending_includes) + "\n\n"
            prompt_text = prompt_session.prompt(
                ANSI(f"\x1b[32mQUERY [{query_index}]: \x1b[0m"),
                multiline=True,
                default=default_text,
            )

        except KeyboardInterrupt:
            console.print("[dim]Interrupted[/dim]")
            continue

        except EOFError:
            _end_repl(runtime, console)
            return

        prompt_text = prompt_text.strip()
        if not prompt_text:
            continue

        if prompt_text.startswith("."):
            try:
                keep_running = _handle_command(
                    runtime=runtime,
                    console=console,
                    command_text=prompt_text,
                    pending_includes=pending_includes,
                )
            except KeyboardInterrupt:
                console.print("[dim]Command interrupted[/dim]")
                continue

            if not keep_running:
                _end_repl(runtime, console)
                return

            continue

        pending_includes.clear()
        try:
            keep_running = _run_streaming_query(
                runtime=runtime,
                console=console,
                query_index=query_index,
                prompt_text=prompt_text,
            )
        except KeyboardInterrupt:
            console.print("[dim]Query interrupted[/dim]")
            continue

        if not keep_running:
            return
