from typing import TYPE_CHECKING, Dict, List

from .aligned import Aligned
from .react import ReAct


if TYPE_CHECKING:
    from .types import Agent


_agents: Dict[str, "Agent"] = {}


def register_agent(agent: "Agent") -> None:
    _agents[agent.type] = agent


def list_agents() -> List[str]:
    return list(_agents.keys())


def get_agent(agent_type: str) -> "Agent":
    agent = _agents.get(agent_type, None)
    if agent is None:
        raise ValueError("Agent not found.")

    return agent


register_agent(Aligned())
register_agent(ReAct())
