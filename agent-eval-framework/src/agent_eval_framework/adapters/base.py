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

"""Base abstract classes for agent test harness adapters.

Provides a unified interface for evaluating both mono-agent (single agent)
and multi-agent systems (hierarchies, swarms, sequential pipelines).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union


class BaseAgentAdapter(ABC):
    """Abstract base class for all agent test harness adapters."""

    def __init__(self, **kwargs):
        self.config = kwargs
        self.agent = self.load_agent(**kwargs)

    @abstractmethod
    def load_agent(self, **kwargs) -> Any:
        """Loads and returns the agent or multi-agent system instance."""
        pass

    @abstractmethod
    def get_response(self, prompt: str) -> Dict[str, Any]:
        """Executes the agent / multi-agent system with a prompt.

        Returns:
            Dictionary formatted for Vertex AI EvalTask:
            {
                "response": str,               # Final text answer
                "actual_response": str,        # Standard alias
                "predicted_trajectory": list,  # Sequence of tool calls / sub-agent actions
                "actual_trajectory": list,     # Standard alias
                "multi_agent_events": list,    # (Optional) Inter-agent messages and delegations
                "metadata": dict,              # (Optional) Token counts, latency, routing
            }
        """
        pass

    def __call__(self, prompt: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Callable interface allowing direct execution by Vertex AI EvalTask."""
        if isinstance(prompt, dict):
            prompt_text = prompt.get("prompt") or prompt.get("query") or str(prompt)
        else:
            prompt_text = str(prompt)
        return self.get_response(prompt_text)

    def batch_get_response(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """Executes responses for a batch of prompts."""
        return [self.get_response(p) for p in prompts]


def normalize_adapter_output(
    response_text: str,
    trajectory: Optional[List[Dict[str, Any]]] = None,
    multi_agent_events: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Standardizes agent execution output into Vertex AI EvalTask schema."""
    traj = trajectory or []
    events = multi_agent_events or []
    meta = metadata or {}

    out = {
        "response": response_text,
        "actual_response": response_text,
        "predicted_trajectory": traj,
        "actual_trajectory": traj,
        "multi_agent_events": events,
        "metadata": meta,
    }
    if error:
        out["error"] = error
    return out
