# HAX REPL Agent Notes

This file describes the current architecture and conventions for contributors and coding agents.

## Project State

- This repository currently implements phases 0-2 of the POC:
  - package skeleton and CLI entrypoint
  - session/message domain models
  - YAML + SQLite persistence
  - interactive REPL loop with streaming responses
- Advanced features (full command suite, RAG providers, MCP, function calling, agent loop control) are scaffolded but not complete yet.

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

## Data Model and Storage

- Pydantic models are in `src/hax_repl/models.py`
- Session YAML files:
  - `~/.local/share/haxllm/sessions/{session-name}.yaml`
- Message store SQLite:
  - `~/.local/share/haxllm/messages.sqlite3`
- Logging file:
  - `~/.local/share/haxllm/logs/hax-repl.log`

## Current REPL UX

- Prompt state line: `<session>/<agent|role>)`
- Prompt line: `QUERY [N]:` (green)
- Spinner shown while waiting for visible response
- `RESULT [N]:` (red) appears on first visible token
- Basic command completion is enabled for dot-prefixed commands
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

## Known Gaps

- Dot commands beyond `.help/.exit/.quit` are not implemented yet
- No dedicated RAG provider implementation yet
- No complete function-calling loop yet
- No MCP client wiring into runtime yet
- No pause/resume control for autonomous agent execution yet
