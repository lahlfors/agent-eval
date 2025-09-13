"""Provides the abstract base class for agent adapters."""

from abc import ABC, abstractmethod

class BaseAgentAdapter(ABC):
    """Abstract base class for an agent adapter.

    This class defines the standard interface for an agent to be used within
    the evaluation framework. A concrete implementation of this class must

    be created for each agent to be evaluated. This "adapts" the specific
    agent's API to the common interface expected by the evaluation runner.
    """

    @abstractmethod
    def get_response(self, prompt: str) -> dict[str, any]:
        """Invokes the agent with a given prompt and returns its structured response.

        This method should encapsulate the logic for calling the agent, whether
        it's a direct function call, an API request, or another mechanism. It
        must return a dictionary containing the agent's final answer and the
        trajectory of tool calls it made to get there.

        Args:
            prompt: The input string to be sent to the agent.

        Returns:
            A dictionary with the following keys:
            - "actual_response" (str): The final text response from the agent.
            - "actual_trajectory" (list): A list representing the sequence of
              tool calls made by the agent. The structure of the items in this
              list can be specific to the agent but should be consistent.
        """
        pass
