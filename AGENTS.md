# HAX REPL Agent Notes

This file describes the current architecture and conventions for contributors and coding agents.

## Project State

- This repository currently implements phases 0-3 of the POC:
  - package skeleton and CLI entrypoint
  - session/message domain models
  - YAML + SQLite persistence
  - interactive REPL loop with streaming responses
  - command layer with conversation/history/show/copy/file/macro commands
- Phase 4 is partially implemented:
  - plugin-loaded function registry and tool-calling runtime loop
- Advanced features (plugin-wired function calling, concrete RAG providers, MCP integration, agent loop control) are scaffolded but not complete yet.

## Implementation Phases and Status

Legend:

- `DONE`: implemented in codebase
- `PARTIAL`: started, but not complete
- `PLANNED`: not implemented yet

### Phase 0 - Project Skeleton (`DONE`)

- Package layout under `src/hax_repl/`
- CLI entrypoint (`hax_repl.cli:main`)
- `pyproject.toml` dependencies and script wiring
- Initial scaffolding for plugin namespaces

### Phase 1 - Domain Models and Persistence (`DONE`)

- Strict Pydantic models for sessions/messages/hash IDs in `src/hax_repl/models.py`
- Deterministic content/context hashing in `src/hax_repl/hashing.py`
- Session YAML persistence in `src/hax_repl/session_store.py`
- SQLite message persistence in `src/hax_repl/message_store.py`

### Phase 2 - Interactive REPL and LLM Streaming (`DONE`)

- Prompt/render loop in `src/hax_repl/repl.py`
- OpenRouter streaming client in `src/hax_repl/llm_client.py`
- Runtime orchestration in `src/hax_repl/runtime.py`
- Thinking tag parsing and hidden-thinking behavior
- REPL stats, submit key bindings, command completion, graceful Ctrl+C/Ctrl+D exit
- File-based logging configured in `src/hax_repl/cli.py`

### Phase 3 - Command Layer (`DONE`)

- Implemented:
  - `.conversation delete-last-message`
  - `.conversation generate-again`
  - `.history append`
  - `.show last-thinking`
  - `.copy last-code`
  - `.copy last-response`
  - `.file <path>` queue for next prompt
  - `.macro <text>` expansion queue for next prompt
  - command parsing/dispatch + multi-word autocomplete
  - clipboard support via `wl-copy` / `xclip` / `xsel`
  - runtime command helpers for history inspection/mutation

### Phase 4 - Plugin Runtime and Function Calling (`PARTIAL`)

- Implemented:
  - function registry and provider protocol in `src/hax_repl/functions.py`
  - schema hashing for function signatures (`name + description + args schema`)
  - prompt persistence now stores enabled functions as `{name, schema_hash}`
  - runtime startup loads function providers from `hax_repl.function_providers` entry points
  - fail-fast plugin validation during runtime initialization
  - built-in default function provider with `python_eval` tool
  - OpenRouter tool-calling loop (assistant tool call -> local execution -> tool result message)
  - persistence of function call requests/results in `ResponseMessage`
- Missing:
  - richer function/provider configuration and selective enable/disable by agent profile
  - dedicated "agent configuration + function set" model
  - explicit agent runner abstraction beyond direct function tools

### Phase 5 - RAG and Macro Expansion (`PARTIAL`)

- Implemented:
  - RAG protocol/data scaffolding (`src/hax_repl/rag.py`)
  - macro parser placeholder (`src/hax_repl/macro.py`)
- Missing:
  - concrete RAG provider implementation(s)
  - runtime RAG invocation and provenance persistence
  - deterministic RAG macro handling (for example `$(rag:...)`)
  - index management/update command flows

### Phase 6 - MCP and Agent Pause/Resume (`PLANNED`)

- Planned:
  - MCP client descriptor loading from JSON
  - local class-to-MCP adapter and tool mapping
  - runtime MCP registration and invocation path
  - pause/resume human-in-the-loop agent control

### Phase 7 - Testing and Hardening (`PARTIAL`)

- Implemented:
  - ad hoc lint/syntax checks during development
- Missing:
  - automated unit tests for hashing/model/store behavior
  - integration tests for REPL command flows
  - plugin contract tests (RAG/MCP/function providers)
  - golden/snapshot tests for terminal UX rendering

## Runtime Overview

- CLI entrypoint: `hax_repl.cli:main`
- REPL loop: `src/hax_repl/repl.py`
- Orchestration runtime: `src/hax_repl/runtime.py`
- OpenRouter client (streaming SSE): `src/hax_repl/llm_client.py`
- Hashing and think-tag parsing: `src/hax_repl/hashing.py`

High-level flow per query:

1. Read prompt from multiline `prompt_toolkit` input
2. Build and persist `PromptMessage` with content/context hashes
3. Build remote query payload from stored conversation + current prompt
4. Stream assistant output from OpenRouter
5. Split `<think>...</think>` from visible text
6. Persist `ResponseMessage` and attach it to current turn
7. Print response + timing/size stats in REPL

Runtime command helper APIs currently available:

- `last_prompt_message()`
- `last_response_message()`
- `append_history_prompt()`
- `delete_last_message()`
- `generate_again()`
- `expand_prompt_macros()`

Function-calling APIs currently available:

- Runtime-internal tool loop in `send_prompt()` using OpenRouter `tools`
- `FunctionRegistry.to_tool_specs()` for OpenAI-compatible tool schemas
- `FunctionRegistry.invoke_json()` for typed invocation from JSON arguments

## Data Model and Storage

- Pydantic models are in `src/hax_repl/models.py`
- Session YAML files:
  - `~/.local/share/haxllm/sessions/{session-name}.yaml`
- Message store SQLite:
  - `~/.local/share/haxllm/messages.sqlite3`
- Logging file:
  - `~/.local/share/haxllm/logs/hax-repl.log`
- Prompt message stores enabled functions with schema hashes (`EnabledFunction`)
- Response message stores function calls and function results

## Current REPL UX

- Prompt state line: `<session>/<agent|role>)`
- Prompt line: `QUERY [N]:` (green)
- Spinner shown while waiting for visible response
- `RESULT [N]:` (red) appears on first visible token
- Basic command completion is enabled for dot-prefixed commands
- Supported commands:
  - `.help`
  - `.exit`, `.quit`
  - `.conversation delete-last-message`
  - `.conversation generate-again`
  - `.history append <text>`
  - `.show last-thinking`
  - `.copy last-code`
  - `.copy last-response`
  - `.file <path>`
  - `.macro <text>`
- Submit shortcuts:
  - `Ctrl+J`
  - `Esc+Enter`
  - CSI-u style `Ctrl+Enter` sequences (terminal-dependent)

## Coding Conventions

- Python 3.13+
- Use type annotations
- Use double quotes in code
- Prefer explicit Pydantic models over free-form dictionaries
- Keep fail-fast behavior for startup/import-level failures

## Environment and Defaults

- API key env var: `HAXSCRAMPER_LLM_REPL_KEY`
- Default model: `anthropic/claude-sonnet-4.5`

## Extension Points (Scaffolded)

- Plugin groups and namespaces are present:
  - `hax_repl.rag_providers`
  - `hax_repl.function_providers`
  - `hax_repl.mcp_clients`
  - `hax_repl.agents`
- Helper modules:
  - `src/hax_repl/plugin_system.py`
  - `src/hax_repl/rag.py`
  - `src/hax_repl/functions.py`
  - `src/hax_repl/macro.py`
- Function provider entry point group:
  - `hax_repl.function_providers`

## Known Gaps

- No dedicated RAG provider implementation yet
- Function calling exists but lacks provider/agent-level configuration UX
- No MCP client wiring into runtime yet
- No pause/resume control for autonomous agent execution yet
