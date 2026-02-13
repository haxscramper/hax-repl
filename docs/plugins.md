# Plugins and Function Calling

This document explains how plugin loading works and how to use function plugins from the REPL.

## Plugin groups

The project defines these entry point groups:

- `hax_repl.function_providers`
- `hax_repl.rag_providers` (scaffolded)
- `hax_repl.mcp_clients` (scaffolded)
- `hax_repl.agents` (scaffolded)

At this stage, function providers are fully wired into runtime tool calling.

## Included example function providers

Configured in `pyproject.toml`:

- `example_functions = "hax_repl.plugins.functions.examples:ExampleFunctionProvider"`
- `json_functions = "hax_repl.plugins.functions.examples:JsonFunctionProvider"`

Built-in functions are also registered by runtime.

### Example functions

- `echo_text`
- `sum_numbers`
- `list_directory`
- `read_text_file`
- `pretty_json`
- `python_eval` (built-in, intentionally simple/insecure for POC)

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

## Notes

- Plugin import/init is fail-fast during runtime startup.
- Function execution failure is captured and returned to the model as structured error JSON.
- The POC intentionally prioritizes working behavior over strict sandboxing or advanced safeguards.
