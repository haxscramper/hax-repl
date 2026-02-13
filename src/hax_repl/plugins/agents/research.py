from __future__ import annotations

from hax_repl.agents import AgentPlugin
from hax_repl.models import PluginAgentMeta
from hax_repl.plugin_system import PluginDescriptor


class ResearchAgentPlugin(AgentPlugin):

    def agent_name(self) -> str:
        return "research"

    def build_step_prompt(self, *, goal: str, step_index: int,
                          step_history: list[str]) -> str:
        return (
            "You are a step-wise research assistant.\n"
            f"Research goal: {goal}\n"
            f"Step: {step_index + 1}\n"
            "Return key facts and next action. Include <agent_done> when complete."
        )

    def should_stop(self, *, response_text: str, step_index: int,
                    max_steps: int) -> bool:
        return "<agent_done>" in response_text or step_index + 1 >= max_steps


def register() -> PluginDescriptor:
    return PluginDescriptor(
        metadata=PluginAgentMeta(
            name="research",
            description="Step-wise research agent plugin.",
        ),
        plugin_factory=ResearchAgentPlugin,
    )
