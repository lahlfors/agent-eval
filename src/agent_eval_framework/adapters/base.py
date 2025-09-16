# src/agent_eval_framework/adapters/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseAgentAdapter(ABC):
    """Abstract base class for agent adapters."""

    def __init__(self, **kwargs):
        self.agent_config = kwargs
        self.agent = self.load_agent(**kwargs)

    @abstractmethod
    def load_agent(self, **kwargs):
        """Loads the underlying agent instance."""
        pass

    @abstractmethod
    def get_response(self, prompt: str) -> Dict[str, Any]:
        """Gets a response from the agent."""
        pass
