import os
from vertexai import agent_engines

# The BaseAgentAdapter will be imported from the framework once it's installed.
# For now, we define a placeholder to allow the code to be written.
class BaseAgentAdapter:
    def get_response(self, prompt: str) -> dict:
        raise NotImplementedError

class VertexAgentEngineAdapter(BaseAgentAdapter):
    def __init__(self, agent_engine_id: str):
        if not agent_engine_id:
            raise ValueError("agent_engine_id must be provided.")
        self.agent_engine_id = agent_engine_id
        self.agent_engine = agent_engines.get(self.agent_engine_id)
        # Create a single session for the lifetime of this adapter instance
        self.session = self.agent_engine.create_session(user_id="evaluation_user")

    def get_response(self, prompt: str) -> dict:
        """Calls the deployed agent and returns the response and trajectory."""
        response = self.agent_engine.query(
            user_id=self.session["user_id"], session_id=self.session["id"], message=prompt
        )

        response_text = "".join(part["text"] for part in response.parts if "text" in part)

        # Note: The current SDK's response object doesn't easily expose the full tool call trajectory.
        # For now, we return an empty list for trajectory. This can be enhanced in a future sprint.
        trajectory = []

        return {"actual_response": response_text, "actual_trajectory": trajectory}
