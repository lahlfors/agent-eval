# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provides an adapter for the local, in-process agent.

This module contains the `LocalAgentAdapter`, a concrete implementation
of the `BaseAgentAdapter` from the `agent-eval-framework`. It handles the
specific logic required to communicate with the `personalized_shopping`
agent when it is running in the same Python process as the evaluation
framework. This allows for a "local deployment" scenario where the agent
is not hosted on a remote server.
"""

from agent_eval_framework.adapters import BaseAgentAdapter
from personalized_shopping.agent import root_agent as local_shopping_agent

class LocalAgentAdapter(BaseAgentAdapter):
    """An adapter to interface with a local, in-process agent instance.

    This class directly imports and uses the `personalized_shopping` agent
    object. It is used for evaluation scenarios where the agent does not need
    to be deployed to a separate server (like Vertex AI Agent Engine).

    Attributes:
        agent: An instance of the local agent.
    """
    def __init__(self):
        """Initializes the LocalAgentAdapter.

        This adapter requires no configuration as it directly instantiates
        the agent from the imported source code.
        """
        self.agent = local_shopping_agent

    def get_response(self, prompt: str) -> dict:
        """Calls the local agent and returns its response and trajectory.

        This method invokes the `query` method of the local agent instance
        and processes the response.

        Args:
            prompt: The input prompt to send to the agent.

        Returns:
            A dictionary containing the agent's final text response under the
            key "actual_response" and the tool call trajectory under the key
            "actual_trajectory".
        """
        # The local agent's query method returns a dictionary with 'output' and 'intermediate_steps'
        response_dict = self.agent.query(prompt)

        response_text = response_dict.get("output", "")

        # The trajectory is a list of (AgentAction, observation) tuples.
        # We need to format this into a list of dictionaries for the framework.
        raw_trajectory = response_dict.get("intermediate_steps", [])
        formatted_trajectory = []
        for action, observation in raw_trajectory:
            formatted_trajectory.append(
                {
                    "tool_name": action.tool,
                    "tool_input": action.tool_input,
                    "tool_output": observation,
                }
            )

        return {
            "actual_response": response_text,
            "actual_trajectory": formatted_trajectory,
        }
