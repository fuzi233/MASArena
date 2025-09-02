import threading
import pandas as pd
from dataclasses import dataclass

from mas_arena.agents import AgentSystem


@dataclass
class Cost:
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float


class CostManager:

    def __init__(self):

        self.total_input_tokens = {}
        self.total_output_tokens = {}
        self.total_tokens = {}

        self.total_input_cost = {}
        self.total_output_cost = {}
        self.total_cost = {}

        self._lock = threading.Lock()

    def get_total_cost(self):

        total_cost = 0.0
        for model in self.total_cost.keys():
            total_cost += self.total_cost[model]
        return total_cost


cost_manager = CostManager()

def get_agent_by_name(agent_name: str) -> AgentSystem:
    from mas_arena.agents import AgentSystemRegistry
    config = {}
    agent_system = AgentSystemRegistry.get(agent_name, config)
    if agent_system is None:
        raise ValueError(f"Agent system '{agent_name}' not found in registry.")
    return agent_system
