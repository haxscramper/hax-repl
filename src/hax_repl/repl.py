from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import textwrap
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.markdown import Markdown

from hax_repl.models import FunctionCallRequest
from hax_repl.runtime import AppRuntime, FunctionCallDecision

COMMANDS = [
    ".help",
    ".exit",
    ".quit",
    ".conversation delete-last-message",
    ".conversation generate-again",
    ".history append",
    ".show last-thinking",
    ".copy last-code",
    ".copy last-response",
    ".file",
    ".macro",
    ".functions",
    ".functions call",
    ".mcp list",
    ".mcp load",
    ".mcp call",
    ".agent list",
    ".agent start",
    ".agent status",
    ".agent pause",
    ".agent resume",
    ".agent step",
    ".agent run",
    ".agent stop",
]


def _build_prompt_session() -> PromptSession[str]:
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

    completer = WordCompleter(COMMANDS, ignore_case=True, sentence=True)
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
        result = subprocess.run(cmd, input=text, text=True, capture_output=True, check=False)
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

    answer = console.input("[cyan]Approve call?[/cyan] [y]es / [n]o / [m]anual-result: ").strip().lower()
    if answer.startswith("n"):
        reason = console.input("[cyan]Reject reason (optional):[/cyan] ").strip()
        status.start()
        return FunctionCallDecision(action="reject", rejection_reason=reason)
    if answer.startswith("m"):
        manual_result = console.input("[cyan]Manual result JSON (or any text):[/cyan] ").strip()
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
    runtime_response,
) -> None:
    console.print(
        f"[dim]stats: model={runtime.model_name} | session={runtime.session.session.value} | "
        f"prompt_chars={len(prompt_text)}[/dim]"
    )
    visible_text = runtime_response.response_message.text
    if visible_text:
        console.print(Markdown(visible_text))
    else:
        console.print("[dim](empty response)[/dim]")
    console.print(
        f"[dim]done: total={runtime_response.stats.elapsed_ms} ms | "
        f"first_token={runtime_response.stats.time_until_first_token_ms} ms | "
        f"thinking={runtime_response.stats.model_thinking_ms} ms | "
        f"query_chars={runtime_response.stats.query_chars} | "
        f"response_chars={runtime_response.stats.response_chars}[/dim]"
    )


def _run_streaming_query(runtime: AppRuntime, console: Console, query_index: int, prompt_text: str) -> bool:
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
                enabled_functions=None,
                on_visible_token=_on_visible_token,
                on_function_call_decision=lambda request: _interactive_function_call_decision(
                    console=console,
                    status=status,
                    request=request,
                ),
            )
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted. Exiting.[/yellow]")
        return False
    except Exception as exc:
        console.print(f"[bold red]request failed:[/bold red] {exc}")
        return True

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
    console.print(
        f"[bold]Agent[/bold]: {state.agent_name} | "
        f"step={state.step_index}/{state.max_steps} | "
        f"paused={state.paused} | done={state.done}"
    )
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
        with console.status("[yellow]agent thinking...[/yellow]", spinner="dots") as status:

            def _on_visible_token(token: str) -> None:
                nonlocal first_visible_token_seen
                if not first_visible_token_seen:
                    first_visible_token_seen = True
                    status.stop()
                    console.print(f"[red]RESULT [{query_index}]:[/red]")

            runtime_response = runtime.run_agent_step(
                on_visible_token=_on_visible_token,
                on_function_call_decision=lambda request: _interactive_function_call_decision(
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
    args = _parse_command(command_text)
    if not args:
        return True

    if args[0] in {".exit", ".quit"}:
        console.print("Bye.")
        return False

    if args[0] == ".help":
        console.print("Commands:")
        console.print("  .help")
        console.print("  .exit | .quit")
        console.print("  .conversation delete-last-message")
        console.print("  .conversation generate-again")
        console.print("  .history append <text>")
        console.print("  .show last-thinking")
        console.print("  .copy last-code")
        console.print("  .copy last-response")
        console.print("  .file <relative-or-absolute-path>")
        console.print("  .macro <text-with-macros>")
        console.print("  .functions")
        console.print("  .functions call <name> <json-args>")
        console.print("  .mcp list")
        console.print("  .mcp load <descriptor.json>")
        console.print("  .mcp call <client-name> <tool-name> <json-args>")
        console.print("  .agent list")
        console.print("  .agent start <agent-name> <goal>")
        console.print("  .agent status")
        console.print("  .agent pause | .agent resume")
        console.print("  .agent step | .agent run [steps]")
        console.print("  .agent stop")
        return True

    if args[0] == ".functions":
        if len(args) == 1:
            functions = runtime.list_functions()
            if not functions:
                console.print("[yellow]No functions loaded.[/yellow]")
                return True
            console.print("[bold]Loaded functions:[/bold]")
            for function in functions:
                console.print(f"  - {function.name} (schema_hash={function.schema_hash})")
            return True
        if len(args) >= 4 and args[1] == "call":
            name = args[2]
            arguments_json = command_text.split(name, 1)[1].strip()
            if not arguments_json:
                console.print("[yellow]Usage: .functions call <name> <json-args>[/yellow]")
                return True
            try:
                result_json = runtime.invoke_function(name, arguments_json)
            except Exception as exc:
                console.print(f"[bold red]function call failed:[/bold red] {exc}")
                return True
            pretty = result_json
            if len(pretty) > 4000:
                pretty = textwrap.shorten(pretty, width=4000, placeholder=" ...[truncated]")
            console.print("[green]Function result:[/green]")
            console.print(pretty)
            return True
        console.print("[yellow]Usage: .functions OR .functions call <name> <json-args>[/yellow]")
        return True

    if args[0] == ".mcp":
        if len(args) == 2 and args[1] == "list":
            clients = runtime.list_mcp_clients()
            if not clients:
                console.print("[yellow]No MCP clients loaded.[/yellow]")
                return True
            console.print("[bold]Loaded MCP clients:[/bold]")
            for client in clients:
                tools = ", ".join(client.function_names) if client.function_names else "(no tools)"
                console.print(f"  - {client.name}: {tools}")
            return True
        if len(args) >= 3 and args[1] == "load":
            path_text = command_text.split("load", 1)[1].strip()
            if not path_text:
                console.print("[yellow]Usage: .mcp load <descriptor.json>[/yellow]")
                return True
            try:
                loaded = runtime.load_mcp_descriptor(path_text)
            except Exception as exc:
                console.print(f"[bold red]Failed to load MCP descriptor:[/bold red] {exc}")
                return True
            console.print(f"[green]Loaded MCP client:[/green] {loaded.name}")
            return True
        if len(args) >= 5 and args[1] == "call":
            client_name = args[2]
            tool_name = args[3]
            arguments_json = command_text.split(tool_name, 1)[1].strip()
            if not arguments_json:
                console.print("[yellow]Usage: .mcp call <client-name> <tool-name> <json-args>[/yellow]")
                return True
            try:
                result_json = runtime.invoke_mcp_tool(client_name, tool_name, arguments_json)
            except Exception as exc:
                console.print(f"[bold red]MCP call failed:[/bold red] {exc}")
                return True
            console.print("[green]MCP result:[/green]")
            console.print(result_json)
            return True
        console.print("[yellow]Usage: .mcp <list|load|call> ...[/yellow]")
        return True

    if args[0] == ".agent":
        if len(args) == 2 and args[1] == "list":
            agents = runtime.list_agents()
            console.print("[bold]Available agents:[/bold]")
            for name in agents:
                console.print(f"  - {name}")
            return True
        if len(args) >= 4 and args[1] == "start":
            agent_name = args[2]
            goal = command_text.split(agent_name, 1)[1].strip()
            if not goal:
                console.print("[yellow]Usage: .agent start <agent-name> <goal>[/yellow]")
                return True
            try:
                runtime.start_agent_run(agent_name=agent_name, goal=goal, max_steps=8)
            except Exception as exc:
                console.print(f"[bold red]Failed to start agent:[/bold red] {exc}")
                return True
            console.print(f"[green]Started agent run:[/green] {agent_name}")
            _print_agent_status(runtime, console)
            return True
        if len(args) == 2 and args[1] == "status":
            _print_agent_status(runtime, console)
            return True
        if len(args) == 2 and args[1] == "pause":
            if runtime.pause_agent():
                console.print("[green]Agent paused.[/green]")
            else:
                console.print("[yellow]No active agent run.[/yellow]")
            return True
        if len(args) == 2 and args[1] == "resume":
            if runtime.resume_agent():
                console.print("[green]Agent resumed.[/green]")
            else:
                console.print("[yellow]No active agent run.[/yellow]")
            return True
        if len(args) == 2 and args[1] == "step":
            return _run_agent_step(runtime, console)
        if len(args) >= 2 and args[1] == "run":
            max_steps = 3
            if len(args) >= 3:
                try:
                    max_steps = max(1, int(args[2]))
                except ValueError:
                    console.print("[yellow]Usage: .agent run [steps][/yellow]")
                    return True
            for _ in range(max_steps):
                state = runtime.agent_status()
                if state is None or state.paused or state.done:
                    break
                if not _run_agent_step(runtime, console):
                    return False
            _print_agent_status(runtime, console)
            return True
        if len(args) == 2 and args[1] == "stop":
            runtime.clear_agent_run()
            console.print("[green]Cleared active agent run.[/green]")
            return True
        console.print(
            "[yellow]Usage: .agent <list|start|status|pause|resume|step|run|stop> ...[/yellow]"
        )
        return True

    if args[0] == ".conversation":
        if len(args) < 2:
            console.print("[yellow]Usage: .conversation <delete-last-message|generate-again>[/yellow]")
            return True
        if args[1] == "delete-last-message":
            if runtime.delete_last_message():
                console.print("[green]Removed last message from session.[/green]")
            else:
                console.print("[yellow]No messages to delete.[/yellow]")
            return True
        if args[1] == "generate-again":
            query_index = runtime.next_query_index()
            first_visible_token_seen = False
            try:
                with console.status("[yellow]thinking...[/yellow]", spinner="dots") as status:

                    def _on_visible_token(token: str) -> None:
                        nonlocal first_visible_token_seen
                        if not first_visible_token_seen:
                            first_visible_token_seen = True
                            status.stop()
                            console.print(f"[red]RESULT [{query_index}]:[/red]")

                    runtime_response = runtime.generate_again(
                        on_visible_token=_on_visible_token,
                        on_function_call_decision=lambda request: _interactive_function_call_decision(
                            console=console,
                            status=status,
                            request=request,
                        ),
                    )
            except Exception as exc:
                console.print(f"[bold red]generate-again failed:[/bold red] {exc}")
                return True

            if runtime_response is None:
                console.print("[yellow]No previous prompt to regenerate.[/yellow]")
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
            return True
        console.print("[yellow]Unknown .conversation command.[/yellow]")
        return True

    if args[0] == ".history":
        if len(args) >= 3 and args[1] == "append":
            text = command_text.split("append", 1)[1].strip()
            if not text:
                console.print("[yellow]Usage: .history append <text>[/yellow]")
                return True
            runtime.append_history_prompt(text)
            console.print("[green]Appended to history as user prompt (not sent).[/green]")
            return True
        console.print("[yellow]Usage: .history append <text>[/yellow]")
        return True

    if args[0] == ".show":
        if len(args) == 2 and args[1] == "last-thinking":
            last_response = runtime.last_response_message()
            if last_response is None:
                console.print("[yellow]No response found.[/yellow]")
                return True
            thinking = last_response.thinking_text.strip()
            if not thinking:
                console.print("[dim](no thinking content in last response)[/dim]")
                return True
            console.print("[bold]Last thinking:[/bold]")
            console.print(thinking)
            return True
        console.print("[yellow]Usage: .show last-thinking[/yellow]")
        return True

    if args[0] == ".copy":
        if len(args) < 2:
            console.print("[yellow]Usage: .copy <last-code|last-response>[/yellow]")
            return True
        last_response = runtime.last_response_message()
        if last_response is None:
            console.print("[yellow]No response found.[/yellow]")
            return True
        if args[1] == "last-response":
            ok, message = _copy_to_clipboard(last_response.text)
            console.print(f"[green]{message}[/green]" if ok else f"[yellow]{message}[/yellow]")
            return True
        if args[1] == "last-code":
            last_code = _extract_last_code_block(last_response.text)
            if not last_code:
                console.print("[yellow]No fenced code block found in last response.[/yellow]")
                return True
            ok, message = _copy_to_clipboard(last_code)
            console.print(f"[green]{message}[/green]" if ok else f"[yellow]{message}[/yellow]")
            return True
        console.print("[yellow]Usage: .copy <last-code|last-response>[/yellow]")
        return True

    if args[0] == ".file":
        if len(args) < 2:
            console.print("[yellow]Usage: .file <relative-or-absolute-path>[/yellow]")
            return True
        candidate = Path(args[1]).expanduser()
        file_path = candidate if candidate.is_absolute() else (Path.cwd() / candidate)
        if not file_path.exists() or not file_path.is_file():
            console.print(f"[yellow]File not found: {file_path}[/yellow]")
            return True
        file_text = file_path.read_text(encoding="utf-8", errors="replace")
        payload = f'<file path="{file_path}">\n{file_text}\n</file>'
        pending_includes.append(payload)
        console.print(f"[green]Queued file for next query:[/green] {file_path}")
        return True

    if args[0] == ".macro":
        if len(args) < 2:
            console.print("[yellow]Usage: .macro <text-with-macros>[/yellow]")
            return True
        raw_body = command_text.split(".macro", 1)[1].strip()
        expanded = runtime.expand_prompt_macros(raw_body)
        pending_includes.append(expanded)
        console.print("[green]Expanded macro text queued for next query.[/green]")
        return True

    console.print("[yellow]Unknown command. Use .help[/yellow]")
    return True


def run_repl(runtime: AppRuntime) -> None:
    console = Console()
    prompt_session = _build_prompt_session()
    pending_includes: list[str] = []

    console.print("[bold]HAX LLM REPL[/bold]  (type .exit to quit)")
    console.print(
        "[dim]Submit: Ctrl+J, Esc+Enter, or Ctrl+Enter (CSI-u terminals). New line: Enter. Complete: Tab.[/dim]"
    )
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
        except (EOFError, KeyboardInterrupt):
            console.print("Bye.")
            return
        prompt_text = prompt_text.strip()
        if not prompt_text:
            continue
        if prompt_text.startswith("."):
            keep_running = _handle_command(
                runtime=runtime,
                console=console,
                command_text=prompt_text,
                pending_includes=pending_includes,
            )
            if not keep_running:
                return
            continue

        pending_includes.clear()
        keep_running = _run_streaming_query(
            runtime=runtime,
            console=console,
            query_index=query_index,
            prompt_text=prompt_text,
        )
        if not keep_running:
            return
