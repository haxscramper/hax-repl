from __future__ import annotations

from hax_repl.agents import AgentPlugin


class CodeExecAgentPlugin(AgentPlugin):
    def agent_name(self) -> str:
        return "code-exec"

    def build_step_prompt(self, *, goal: str, step_index: int, step_history: list[str]) -> str:
        history = "\n\n".join(step_history[-2:]) if step_history else "(none)"
        return (
            "You are a coding assistant with access to local tools.\n"
            f"Goal: {goal}\n"
            f"Current step: {step_index + 1}\n"
            f"Recent history:\n{history}\n\n"
            "Use tools when useful. Keep output concise and actionable. "
            "When done, include <agent_done>."
        )

    def should_stop(self, *, response_text: str, step_index: int, max_steps: int) -> bool:
        return "<agent_done>" in response_text or step_index + 1 >= max_steps
