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
