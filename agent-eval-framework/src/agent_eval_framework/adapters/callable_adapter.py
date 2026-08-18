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

"""Callable / Generic Python Agent Adapter.

Interfaces any arbitrary Python function, class instance, LangChain / LangGraph runnable,
CrewAI crew, or AutoGen group chat for evaluation.
"""

import importlib
import inspect
from typing import Any, Callable, Dict, List, Optional, Union
from agent_eval_framework.adapters.base import BaseAgentAdapter, normalize_adapter_output
from agent_eval_framework.utils.logger import get_logger

log = get_logger(__name__)


def load_callable(target: Union[str, Callable]) -> Callable:
    """Loads a callable from a string path (e.g. 'module.sub:func' or 'module.sub.func') or callable."""
    if callable(target):
        return target
    if isinstance(target, str):
        if ":" in target:
            module_path, attr_name = target.split(":", 1)
        elif "." in target:
            module_path, attr_name = target.rsplit(".", 1)
        else:
            raise ValueError(f"Target '{target}' must be in 'module:callable' or 'module.callable' format.")
        
        mod = importlib.import_module(module_path)
        obj = getattr(mod, attr_name)
        return obj
    raise TypeError(f"Target must be a callable or string path, got {type(target)}")


class CallableAgentAdapter(BaseAgentAdapter):
    """Adapter to evaluate any Python callable, LangGraph, or custom agent class."""

    def __init__(
        self,
        target: Optional[Union[str, Callable]] = None,
        agent_module: Optional[str] = None,
        agent_name: Optional[str] = None,
        response_key: str = "response",
        trajectory_key: str = "trajectory",
        **kwargs,
    ):
        """Initializes the Callable Agent Adapter.

        Args:
            target: Fully qualified path (e.g. "my_app.agent.invoke_agent") or callable.
            agent_module: (Optional) module containing the agent.
            agent_name: (Optional) variable/function name in the module.
            response_key: Key in dict output containing the response string (default: "response").
            trajectory_key: Key in dict output containing tool calls (default: "trajectory").
        """
        self.target_ref = target or (f"{agent_module}.{agent_name}" if agent_module and agent_name else None)
        self.response_key = response_key
        self.trajectory_key = trajectory_key
        super().__init__(target=self.target_ref, **kwargs)

    def load_agent(self, **kwargs) -> Any:
        if not self.target_ref:
            raise ValueError("CallableAgentAdapter requires 'target' or ('agent_module' + 'agent_name').")
        obj = load_callable(self.target_ref)
        if inspect.isclass(obj):
            # Instantiate class if class passed
            return obj(**kwargs)
        return obj

    def get_response(self, prompt: str) -> Dict[str, Any]:
        """Executes the callable agent and formats output."""
        try:
            # Check if agent has invoke/query/run or is callable
            if hasattr(self.agent, "invoke") and callable(self.agent.invoke):
                raw_out = self.agent.invoke({"input": prompt} if isinstance(self.agent.invoke, Callable) else prompt)
            elif hasattr(self.agent, "query") and callable(self.agent.query):
                raw_out = self.agent.query(prompt)
            elif hasattr(self.agent, "run") and callable(self.agent.run):
                raw_out = self.agent.run(prompt)
            elif callable(self.agent):
                raw_out = self.agent(prompt)
            else:
                raise TypeError(f"Agent object {type(self.agent)} has no callable interface (invoke, query, run, __call__).")

            # Parse output
            if isinstance(raw_out, str):
                return normalize_adapter_output(response_text=raw_out)

            if isinstance(raw_out, dict):
                response_text = (
                    raw_out.get(self.response_key)
                    or raw_out.get("output")
                    or raw_out.get("actual_response")
                    or raw_out.get("content")
                    or str(raw_out)
                )
                trajectory = (
                    raw_out.get(self.trajectory_key)
                    or raw_out.get("tool_calls")
                    or raw_out.get("intermediate_steps")
                    or []
                )
                events = raw_out.get("multi_agent_events") or raw_out.get("agent_steps") or []
                meta = raw_out.get("metadata") or {}
                return normalize_adapter_output(
                    response_text=str(response_text),
                    trajectory=trajectory,
                    multi_agent_events=events,
                    metadata=meta,
                )

            return normalize_adapter_output(response_text=str(raw_out))

        except Exception as e:
            log.error(f"Error in CallableAgentAdapter: {e}", exc_info=True)
            return normalize_adapter_output(
                response_text=f"Error executing agent: {e}",
                error=str(e),
            )
