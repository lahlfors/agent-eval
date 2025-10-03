# Copyright 2024 Google LLC
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
#

"""Adapter for evaluating agents built with the Google Agent Development Kit (ADK)."""

import asyncio
import importlib
import json
import uuid
from typing import Any, Dict, List, Optional
import importlib
from abc import ABC, abstractmethod # Import ABC

from google.adk.runners import Runner
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.genai import types as genai_types
from opentelemetry import trace

# --- Corrected Import ---
from agent_eval_framework.adapters.base import BaseAgentAdapter # Import BaseAgentAdapter
from agent_eval_framework.utils.logger import get_logger

log = get_logger(__name__)
tracer = trace.get_tracer(__name__)

class ADKAgentAdapter(BaseAgentAdapter): # Inherit from BaseAgentAdapter
    """Adapter for interacting with agents built using the Google ADK."""

    def __init__(
        self,
        agent_module: str,
        agent_name: str,
        user_id: str = "agent-eval-user",
        artifact_service=None,
        session_service=None,
        memory_service=None,
        **kwargs,
    ):
        """Initializes the ADKAgentAdapter.

        Args:
            agent_module: The fully qualified name of the Python module
                containing the ADK agent (e.g., "personalized_shopping.agent").
            agent_name: The name of the variable in the module that holds the
                ADK agent instance (e.g., "root_agent").
            user_id: The user ID to use for agent sessions.
            artifact_service: An instance of ADK BaseArtifactService.
            session_service: An instance of ADK BaseSessionService.
            memory_service: An instance of ADK BaseMemoryService.
            **kwargs: Additional keyword arguments.
        """
        self.agent_module = agent_module
        self.agent_name = agent_name
        self.user_id = user_id
        self._kwargs = kwargs
        self.agent = None
        super().__init__() # Call parent init after setting basic attributes

        # Load the agent FIRST
        self.load_agent()

        if not self.agent:
            raise RuntimeError(f"Agent '{self.agent_name}' could not be loaded from '{self.agent_module}'.")

        # Use provided services or create default InMemory services
        self._artifact_service = artifact_service if artifact_service else InMemoryArtifactService()
        self._session_service = session_service if session_service else InMemorySessionService()
        self._memory_service = memory_service if memory_service else InMemoryMemoryService()

        # Initialize the Runner AFTER the agent is loaded
        self.runner = Runner(
            app_name="agent_eval_framework_adk_runner",
            agent=self.agent, # Now self.agent is guaranteed to be set
            artifact_service=self._artifact_service,
            session_service=self._session_service,
            memory_service=self._memory_service,
        )
        log.debug("ADK Runner initialized within Adapter.")

    def load_agent(self):
        """Loads the agent instance from the specified module."""
        try:
            module = importlib.import_module(self.agent_module)
            self.agent = getattr(module, self.agent_name)
            log.info(f"{self.agent_module} module loaded and {self.agent_name} defined.")
        except Exception as e:
            log.error(f"Failed to load agent '{self.agent_name}' from module '{self.agent_module}': {e}", exc_info=True)

    async def _run_agent_async(self, prompt: str) -> Dict[str, Any]:
        """Runs the ADK agent asynchronously and captures its output."""
        session_id = str(uuid.uuid4())
        content = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=prompt)],
        )

        final_response = ""
        tool_calls = []
        full_conversation = []
        usage_metadata = None

        try:
            # --- Create the session in the session service ---
            await self._session_service.create_session(
                app_name=self.runner.app_name,
                user_id=self.user_id,
                session_id=session_id
            )
            log.debug(f"ADK session created: {session_id}")

            async for event in self.runner.run_async(user_id=self.user_id, session_id=session_id, new_message=content):
                full_conversation.append(event.model_dump_json())
                if hasattr(event, "usage_metadata") and event.usage_metadata:
                    usage_metadata = event.usage_metadata
                if event.content:
                    text_content = " ".join(part.text for part in event.content.parts if part.text)
                    if event.author == self.agent.name: # Agent's response
                        final_response = text_content

                    # Log tool calls made by the agent
                    for part in event.content.parts:
                        if part.function_call:
                            tool_calls.append({
                                "tool_name": part.function_call.name,
                                "tool_input": part.function_call.args,
                            })
        except Exception as e:
            log.error(f"Error during runner.run_async: {e}", exc_info=True)
            return {
                "response": f"Error during agent execution: {e}",
                "predicted_trajectory": tool_calls,
                "full_conversation": json.dumps(full_conversation),
                "error": str(e),
            }

        result = {
            "response": final_response,
            "predicted_trajectory": tool_calls,
            "full_conversation": json.dumps(full_conversation),
        }
        if usage_metadata:
            result["usage_metadata"] = usage_metadata
        return result

    def get_response(self, prompt: str) -> Dict[str, Any]:
        """Gets the agent's response to a prompt."""
        if not hasattr(self, 'agent_name') or not self.agent_name:
            log.error("CRITICAL: self.agent_name is missing or empty in get_response!")
            return {"response": "Error: agent_name attribute missing", "error": "AttributeError: agent_name"}

        with tracer.start_as_current_span("ADKAgentAdapter.get_response") as span:
            span.set_attribute("agent.name", self.agent_name)
            span.set_attribute("input.prompt", prompt)
            # Add GenAI attributes
            span.set_attribute("gen_ai.system", "VertexAI")
            if self.agent and hasattr(self.agent, "model") and hasattr(self.agent.model, "model_name"):
                span.set_attribute("gen_ai.request.model", self.agent.model.model_name)
            span.set_attribute("gen_ai.prompt", prompt)
            try:
                result = asyncio.run(self._run_agent_async(prompt))
                if result.get("error"):
                    span.set_attribute("error", True)
                    span.set_attribute("error.message", result["error"])
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(result["error"])))
                else:
                    span.set_attribute("gen_ai.response.text", result.get("response"))
                    usage_metadata = result.get("usage_metadata")
                    if usage_metadata:
                        span.set_attribute("gen_ai.usage.input_tokens", usage_metadata.prompt_token_count)
                        span.set_attribute("gen_ai.usage.output_tokens", usage_metadata.candidates_token_count)
                        span.set_attribute("gen_ai.usage.total_tokens", usage_metadata.total_token_count)
                    span.set_status(trace.Status(trace.StatusCode.OK))
                return result
            except Exception as e:
                log.error(f"Error in ADKAgentAdapter.get_response: {e}", exc_info=True)
                span.record_exception(e)
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                return {
                    "response": f"Adapter call failed: {e}",
                    "predicted_trajectory": [],
                    "error": str(e),
                }

    def __call__(self, prompt: str) -> Dict[str, Any]:
        """Makes the adapter callable, running the agent for a given prompt.

        Args:
            prompt: The input prompt to send to the agent.

        Returns:
            A dictionary containing the agent's response.
        """
        return self.get_response(prompt)