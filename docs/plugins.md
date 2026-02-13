# Plugins and Function Calling

This document explains how plugin loading works and how to use function plugins from the REPL.

## Plugin groups

The project defines these entry point groups:

- `hax_repl.function_providers`
- `hax_repl.rag_providers` (scaffolded)
- `hax_repl.mcp_clients`
- `hax_repl.agents`

At this stage, function providers are fully wired into runtime tool calling, and MCP/agent plugins are integrated into the runtime loop.

RAG providers are also wired into runtime and macro expansion.

## Included example function providers

Configured in `pyproject.toml`:

- `example_functions = "hax_repl.plugins.functions.example_functions:ExampleFunctionProvider"`
- `json_functions = "hax_repl.plugins.functions.json_functions:JsonFunctionProvider"`

Built-in functions are also registered by runtime.

### Example functions

- `echo_text`
- `sum_numbers`
- `list_directory`
- `read_text_file`
- `pretty_json`
- `python_eval` (built-in, intentionally simple/insecure for POC)

## Included example MCP plugins

Configured in `pyproject.toml`:

- `example_fs = "hax_repl.plugins.mcp.example_fs:example_fs_mcp_client"`
- `example_time = "hax_repl.plugins.mcp.example_time:example_time_mcp_client"`

The MCP tools are registered into the same function-calling registry and exposed to the model as tools.

### MCP REPL commands

- `.mcp list` - list loaded MCP clients and their function names
- `.mcp load <descriptor.json>` - load a local-class MCP client from JSON descriptor
- `.mcp call <client-name> <tool-name> <json-args>` - invoke MCP tool directly

Example descriptor file is included:

- `examples/mcp_time_descriptor.json`

Descriptor schema:

```json
{
  "name": "descriptor-time",
  "module": "hax_repl.plugins.mcp.example_time",
  "class": "ExampleTimeTools"
}
```

## Included example agent plugins

Configured in `pyproject.toml`:

- `code_exec = "hax_repl.plugins.agents.code_exec:CodeExecAgentPlugin"`
- `research = "hax_repl.plugins.agents.research:ResearchAgentPlugin"`

### Agent REPL commands

- `.agent list`
- `.agent start <agent-name> <goal>`
- `.agent status`
- `.agent pause`
- `.agent resume`
- `.agent step`
- `.agent run [steps]`
- `.agent stop`

Agent execution is step-wise and uses the same model/tool loop with interactive per-tool approval.

## Included example RAG providers

Configured in `pyproject.toml`:

- `chroma = "hax_repl.plugins.rag.chroma_provider:ChromaVectorRagProvider"`
- `tantivy = "hax_repl.plugins.rag.tantivy_provider:TantivyFullTextRagProvider"`

The Chroma provider uses OpenRouter embeddings with model `openai/text-embedding-3-small`:

- `HAXSCRAMPER_LLM_REPL_KEY` is required
- optional: `HAX_REPL_EMBEDDING_MODEL` (defaults to `openai/text-embedding-3-small`; `text-embedding-3-small` alias is accepted)

### RAG REPL commands

- `.rag providers`
- `.rag indices <provider>`
- `.rag update <provider> <index> <path1> [path2 ...]`
- `.rag query <provider> <index> <query>`

### Macro syntax

- `$(get-os)` expands to host OS description
- `$(rag:provider/index "query text")` performs RAG query and inlines result chunks into prompt

Example:

- `Explain this code: $(rag:chroma/mydocs "how repl commands are parsed")`

## How to use functions in the REPL

### 1) Inspect loaded functions

Run:

` .functions `

This prints function names and schema hashes.

### 2) Call a function directly

Run:

- `.functions call sum_numbers {"numbers":[1,2,3.5]}`
- `.functions call read_text_file {"path":"README.md","max_chars":500}`
- `.functions call pretty_json {"value_json":"{\"a\":2,\"b\":1}"}`

### 3) Let the model call functions automatically

Function tools are sent to the model automatically by default.

Prompt examples:

- `Calculate 5 * 21 by using python_eval and report only the final value.`
- `Read README.md with a tool and summarize it in 3 bullet points.`
- `List files in current directory and tell me which look like Python modules.`

## How function calling works internally

1. Runtime loads providers from `hax_repl.function_providers`.
2. Each `FunctionSpec` is converted into OpenAI-compatible `tools` schema.
3. Model response may contain `tool_calls`.
4. Runtime invokes local function implementation with Pydantic argument validation.
5. Tool results are appended back to model context as `tool` messages.
6. Loop continues until final assistant text response arrives.
7. Function call requests/results are persisted in `ResponseMessage`.
8. During REPL-driven runs, each tool call is presented for user decision:
   - approve
   - reject with reason
   - manual result payload
9. Macro expansion executes before prompt send; RAG invocations contribute provenance into context hashing.

## Writing your own function provider

Implement a provider with:

- `provider_name() -> str`
- `functions() -> list[FunctionSpec]`

Function spec fields:

- `name`
- `description`
- `args_model` (`pydantic.BaseModel`)
- `result_model` (`BaseModel | None`)
- `impl(args_model_instance) -> result`

Register provider in `pyproject.toml`:

```toml
[project.entry-points."hax_repl.function_providers"]
my_tools = "my_package.my_module:MyFunctionProvider"
```

Then run:

- `uv sync`
- `uv run hax-repl --session test-tools`

## Writing your own MCP plugin

Return an object that exposes:

- `client_name() -> str`
- `list_tools() -> list[FunctionSpec]`
- `invoke(tool_name: str, arguments_json: str) -> str`

Or use `LocalClassMcpClientAdapter` with a regular Python class.

## Writing your own agent plugin

Implement:

- `agent_name() -> str`
- `build_step_prompt(goal, step_index, step_history) -> str`
- `should_stop(response_text, step_index, max_steps) -> bool`

Register under `[project.entry-points."hax_repl.agents"]`.

## Notes

- Plugin import/init is fail-fast during runtime startup.
- Function execution failure is captured and returned to the model as structured error JSON.
- The POC intentionally prioritizes working behavior over strict sandboxing or advanced safeguards.
