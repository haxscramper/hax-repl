Implementation of the LLM repl, mostly inspired by https://github.com/sigoden/aichat, but designed to overcome a few issues with the UX/UI that I could not resolve, or that are treated as "won't fix" in the aichat repo. 

The result is a much more personalized tool that made several design decisions in a different manner. It suits me, it might suit you, it might not. 

- [llm-functions is suboptimal #1355](https://github.com/sigoden/aichat/issues/1355)
  - I spent about a day trying to understand how to add custom functions and if there is maybe a simpler way to do this than juggling half a dozen separate scripts and build components. Maybe I just missed something, but I did not see anything
  - For hax-repl I'm explicitly focusing on making it easier to integrate **python** functions, agents and MCP implementations in the REPL, which fits my workflow better, even though it will mean the other extension function customization methods would be harder to use. 
- [HIde thinking token #1215](https://github.com/sigoden/aichat/issues/1215)
  - Some models, like `kimi2.5` can generate thousands of thinking tokens, and I don't care about them 99.9% of the time. Allegedly there is some way to hide the thinking tokens in the `aichat` -- according to the reply in the issue, but I haven't found any. 
  - For hax-repl I hide the thinking tokens by default. 
- https://github.com/sigoden/aichat/issues/88
  - https://github.com/sigoden/aichat/pull/162 marked the feature request as resolved, but did not implement the actually useful, instead writing "Editing the session files directly is a faster and more efficient way to edit sessions. So aichat don't provide commands like `.session delete` or `.session edit`." which I also don't understand as the yaml files are not saved by default, and when they are saved and I want to delete the last request and response I have to (1) close the session, (2) open the yaml file, (3) delete the incorrect part, (4) re-start the session again and continue the conversation. 
  - for hax-repl the set of commands for editing and interacting with messages is more expansive. 

Some features can be resolved/configured in the aichat proper, but are not defaults. For the hax-repl I choose a different set of defaults:

- I add the aichat configuration to dotbot, so the default placement for the session location is `~/.local/state`. AIchat session state is stored in the `~/.config`
- Sessions are not named and saved by default -- fixable with `--save-session` and `--session (date -Is)`, but it should be a default behavior IMO

Some features that are explicitly left out and are more limited than the aichat 

- Only supports openrouter for LLM provider -- that's what I use, and I'm not planning on adding a more general provider support. 

# Run

1. Set key:
   - `export HAXSCRAMPER_LLM_REPL_KEY=...`
2. Install:
   - `uv sync`
3. Start:
   - `uv run hax-repl --session my-test`

# Note

The code in this repo *was* initially written using Cursor and ChatGPT Codex-5.3 for the initial POC, but in the future I will be documenting, cleaning it up and expanding. 
