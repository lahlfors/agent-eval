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

"""Multi-Agent System (MAS) Test Harness Adapter.

Provides structured evaluation for complex multi-agent architectures:
- Hierarchical Coordinator / Supervisor with Specialist Sub-Agents
- Sequential Pipeline Workflows (e.g., Planner -> Executor -> Reviewer)
- Collaborative Multi-Agent Swarms
"""

import importlib
import inspect
from typing import Any, Callable, Dict, List, Optional, Union
from agent_eval_framework.adapters.base import BaseAgentAdapter, normalize_adapter_output
from agent_eval_framework.utils.logger import get_logger

log = get_logger(__name__)


class MultiAgentSystemAdapter(BaseAgentAdapter):
    """Adapter specifically designed to benchmark Multi-Agent Systems (MAS)."""

    def __init__(
        self,
        coordinator_entrypoint: Optional[str] = None,
        sub_agents: Optional[List[str]] = None,
        topology: str = "hierarchical",  # "hierarchical", "sequential", "collaborative"
        **kwargs,
    ):
        """Initializes the Multi-Agent System Adapter.

        Args:
            coordinator_entrypoint: Dot-path to coordinator/orchestrator agent (e.g. "my_mas.coordinator:run_mas").
            sub_agents: List of registered sub-agent names or dot-paths.
            topology: Architecture pattern ("hierarchical", "sequential", "collaborative").
        """
        self.coordinator_entrypoint = coordinator_entrypoint
        self.sub_agents = sub_agents or []
        self.topology = topology
        self.extra_kwargs = kwargs
        super().__init__(**kwargs)

    def load_agent(self, **kwargs) -> Any:
        if not self.coordinator_entrypoint:
            return None
        if ":" in self.coordinator_entrypoint:
            module_path, attr_name = self.coordinator_entrypoint.split(":", 1)
        elif "." in self.coordinator_entrypoint:
            module_path, attr_name = self.coordinator_entrypoint.rsplit(".", 1)
        else:
            raise ValueError(f"Coordinator entrypoint '{self.coordinator_entrypoint}' must be 'module:func' or 'module.func'.")

        module = importlib.import_module(module_path)
        obj = getattr(module, attr_name)
        if inspect.isclass(obj):
            return obj(**kwargs)
        return obj

    def get_response(self, prompt: str) -> Dict[str, Any]:
        """Runs the multi-agent system and captures inter-agent traces."""
        if not self.agent:
            return normalize_adapter_output(
                response_text="Error: Coordinator entrypoint not loaded.",
                error="Coordinator not loaded",
            )

        try:
            # Execute coordinator
            if hasattr(self.agent, "run_mas") and callable(self.agent.run_mas):
                result = self.agent.run_mas(prompt, **self.extra_kwargs)
            elif hasattr(self.agent, "query") and callable(self.agent.query):
                result = self.agent.query(prompt)
            elif callable(self.agent):
                result = self.agent(prompt)
            else:
                raise TypeError(f"Coordinator {type(self.agent)} is not callable.")

            # Process structured multi-agent output
            if isinstance(result, str):
                return normalize_adapter_output(response_text=result)

            if isinstance(result, dict):
                response_text = result.get("response") or result.get("final_response") or result.get("output", "")
                
                # Consolidated trajectory (across all subagents)
                tool_trajectory = result.get("trajectory") or result.get("tool_calls", [])
                
                # Multi-agent specific event traces (sub-agent routing & delegation steps)
                multi_agent_events = result.get("multi_agent_events") or result.get("delegations", [])
                
                # Metadata including topology and subagent participation
                metadata = {
                    "topology": self.topology,
                    "registered_sub_agents": self.sub_agents,
                    "participating_agents": result.get("participating_agents", []),
                    "delegation_count": len(multi_agent_events),
                    "total_turns": result.get("turns", 1),
                }
                if "token_usage" in result:
                    metadata["token_usage"] = result["token_usage"]

                return normalize_adapter_output(
                    response_text=str(response_text),
                    trajectory=tool_trajectory,
                    multi_agent_events=multi_agent_events,
                    metadata=metadata,
                )

            return normalize_adapter_output(response_text=str(result))

        except Exception as e:
            log.error(f"Error executing MultiAgentSystemAdapter: {e}", exc_info=True)
            return normalize_adapter_output(
                response_text=f"Multi-Agent Execution Error: {e}",
                error=str(e),
            )
