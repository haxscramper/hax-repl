from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class AgentPlugin(Protocol):

    def agent_name(self) -> str:
        ...

    def build_step_prompt(self, *, goal: str, step_index: int,
                          step_history: list[str]) -> str:
        ...

    def should_stop(self, *, response_text: str, step_index: int,
                    max_steps: int) -> bool:
        ...


@dataclass
class AgentRunState:
    agent_name: str
    goal: str
    max_steps: int
    step_index: int = 0
    paused: bool = False
    done: bool = False
    step_history: list[str] = field(default_factory=list)
    last_response: str = ""


class DefaultInteractiveAgent:

    def agent_name(self) -> str:
        return "default-agent"

    def build_step_prompt(self, *, goal: str, step_index: int,
                          step_history: list[str]) -> str:
        prior = "\n\n".join(step_history[-3:]) if step_history else "(none)"
        return ("You are running as a step-wise coding agent.\n"
                f"Goal: {goal}\n"
                f"Step: {step_index + 1}\n"
                "Prior step outputs (latest up to 3):\n"
                f"{prior}\n\n"
                "Produce the next concise action/result. "
                "If the goal is complete, include the marker <agent_done>.")

    def should_stop(self, *, response_text: str, step_index: int,
                    max_steps: int) -> bool:
        if "<agent_done>" in response_text:
            return True
        return step_index + 1 >= max_steps
