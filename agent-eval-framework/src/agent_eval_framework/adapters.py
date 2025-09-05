from abc import ABC, abstractmethod

class BaseAgentAdapter(ABC):
    @abstractmethod
    def get_response(self, prompt: str) -> dict[str, any]:
        """
        Takes a prompt and returns the agent's response.

        Returns:
            A dictionary containing at least the following key:
            - "actual_response" (str): The final text response from the agent.
            - "actual_trajectory" (list): The sequence of tool calls made by the agent.
        """
        pass
