from abc import ABC, abstractmethod

class BaseAgentAdapter(ABC):
    @abstractmethod
    def get_response(self, prompt: str) -> dict:
        """Takes a prompt and returns the agent's response and trajectory."""
        pass
