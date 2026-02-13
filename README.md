## HAX LLM REPL (POC)

Initial implementation of phases 0-2:

- package skeleton and runtime boundary
- strict pydantic domain models for sessions/messages
- YAML session persistence and SQLite message storage
- interactive REPL with prompt state header, query numbering, thinking spinner, streaming result rendering

### Run

1. Set key:
   - `export HAXSCRAMPER_LLM_REPL_KEY=...`
2. Install:
   - `uv sync`
3. Start:
   - `uv run hax-repl --session my-test`

### Function plugins quick start

Example function-provider plugins are included and auto-loaded via entry points:

- `example_functions` provider:
  - `echo_text`
  - `sum_numbers`
  - `list_directory`
  - `read_text_file`
- `json_functions` provider:
  - `pretty_json`
- built-in provider:
  - `python_eval`

In REPL:

- List loaded functions:
  - `.functions`
- Call function directly:
  - `.functions call sum_numbers {"numbers":[1,2,3.5]}`
  - `.functions call echo_text {"text":"hello","uppercase":true}`
- Let model call tools automatically:
  - `Use available tools to list files in the current directory and summarize them.`

### MCP and agent loop quick start

- List loaded MCP clients:
  - `.mcp list`
- Load MCP client from descriptor JSON:
  - `.mcp load examples/mcp_time_descriptor.json`
- Invoke MCP method directly:
  - `.mcp call example-time now_utc {}`
- List available agents:
  - `.agent list`
- Start and run an agent:
  - `.agent start code-exec "Inspect the project and summarize next refactor steps"`
  - `.agent run 3`
- Pause/resume/step/status:
  - `.agent pause`
  - `.agent resume`
  - `.agent step`
  - `.agent status`

See `docs/plugins.md` for full plugin authoring and usage instructions.
