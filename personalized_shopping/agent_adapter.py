"""Provides an adapter for the deployed Vertex AI Agent Engine.

This module contains the `VertexAgentEngineAdapter`, a concrete implementation
of the `BaseAgentAdapter` from the `agent-eval-framework`. It handles the
specific logic required to communicate with a live, deployed agent on
Vertex AI Agent Engine, making it possible to evaluate the production agent
using the generic evaluation framework.
"""

from vertexai import agent_engines
from agent_eval_framework.adapters import BaseAgentAdapter

class VertexAgentEngineAdapter(BaseAgentAdapter):
    """An adapter to interface with a deployed Vertex AI Agent Engine.

    This class connects to a specific agent deployed on Vertex AI Agent Engine
    and handles the communication for the evaluation framework. It initializes
    a session with the agent and uses it to send prompts and receive responses.

    Attributes:
        agent_engine_id: The unique identifier of the deployed agent.
        agent_engine: The Vertex AI Agent Engine client object.
        session: A single, persistent session created for the evaluation run.
    """
    def __init__(self, agent_engine_id: str):
        """Initializes the VertexAgentEngineAdapter.

        Args:
            agent_engine_id: The unique identifier for the deployed agent in
                             Vertex AI Agent Engine.

        Raises:
            ValueError: If `agent_engine_id` is not provided.
        """
        if not agent_engine_id:
            raise ValueError("agent_engine_id must be provided.")
        self.agent_engine_id = agent_engine_id
        self.agent_engine = agent_engines.get(self.agent_engine_id)
        # Create a single session for the lifetime of this adapter instance
        self.session = self.agent_engine.create_session(user_id="evaluation_user")

    def get_response(self, prompt: str) -> dict:
        """Calls the deployed agent and returns its response and trajectory.

        This method sends the user's prompt to the configured Vertex AI agent
        and processes the response. It extracts the text part of the response
        for evaluation.

        Note:
            The current version of the SDK does not easily expose the detailed
            tool call trajectory from the response object. Therefore, the
            "actual_trajectory" is returned as an empty list.

        Args:
            prompt: The input prompt to send to the agent.

        Returns:
            A dictionary containing the agent's final text response under the
            key "actual_response" and an empty list for "actual_trajectory".
        """
        response = self.agent_engine.query(
            user_id=self.session["user_id"], session_id=self.session["id"], message=prompt
        )

        response_text = "".join(part["text"] for part in response.parts if "text" in part)
        trajectory = []

        return {"actual_response": response_text, "actual_trajectory": trajectory}
