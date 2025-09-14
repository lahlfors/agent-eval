# Copyright 2025 Google LLC
# ... (license headers) ...

from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseAgentAdapter(ABC):
    """Abstract base class for agent adapters."""

    def __init__(self, **kwargs):
        self.agent = self.load_agent(**kwargs)

    @abstractmethod
    def load_agent(self, **kwargs):
        """Loads and returns the agent instance."""
        pass

    @abstractmethod
    def get_response(self, prompt: str) -> Dict[str, Any]:
        """Gets a response from the agent for a single prompt.

        Should return a dictionary, minimally including:
        {'actual_response': str}
        Optionally, can include other keys like 'actual_trajectory'.
        """
        pass

    def batch_get_response(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """Optional: Efficiently gets responses for multiple prompts."""
        return [self.get_response(prompt) for prompt in prompts]
