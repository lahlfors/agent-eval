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
