from __future__ import annotations

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.markdown import Markdown

from hax_repl.runtime import AppRuntime

COMMANDS = [
    ".help",
    ".exit",
    ".quit",
    ".conversation",
    ".history",
    ".show",
    ".copy",
    ".file",
    ".macro",
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

    completer = WordCompleter(COMMANDS, ignore_case=True, sentence=False)
    return PromptSession(
        completer=completer,
        complete_while_typing=False,
        key_bindings=bindings,
    )


def run_repl(runtime: AppRuntime) -> None:
    console = Console()
    prompt_session = _build_prompt_session()

    console.print("[bold]HAX LLM REPL[/bold]  (type .exit to quit)")
    console.print(
        "[dim]Submit: Ctrl+J, Esc+Enter, or Ctrl+Enter (CSI-u terminals). New line: Enter. Complete: Tab.[/dim]"
    )
    while True:
        query_index = runtime.next_query_index()
        console.print(f"[cyan]{runtime.prompt_state_label}[/cyan]")
        try:
            prompt_text = prompt_session.prompt(
                ANSI(f"\x1b[32mQUERY [{query_index}]: \x1b[0m"),
                multiline=True,
            )
        except (EOFError, KeyboardInterrupt):
            console.print("Bye.")
            return
        prompt_text = prompt_text.strip()
        if not prompt_text:
            continue
        if prompt_text in {".exit", ".quit"}:
            console.print("Bye.")
            return
        if prompt_text == ".help":
            console.print("Commands: .help, .exit, .quit")
            continue

        console.print(
            f"[dim]stats: model={runtime.model_name} | session={runtime.session.session.value} | "
            f"prompt_chars={len(prompt_text)}[/dim]"
        )

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
                    enabled_functions=[],
                    on_visible_token=_on_visible_token,
                )
        except KeyboardInterrupt:
            console.print("[yellow]Interrupted. Exiting.[/yellow]")
            return
        except Exception as exc:
            console.print(f"[bold red]request failed:[/bold red] {exc}")
            continue

        if not first_visible_token_seen:
            console.print(f"[red]RESULT [{query_index}]:[/red]")

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
