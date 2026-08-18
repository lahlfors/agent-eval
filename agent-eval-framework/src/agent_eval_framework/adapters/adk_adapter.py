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

"""Adapter for evaluating single and multi-agent systems built with Google Agent Development Kit (ADK)."""

import asyncio
import importlib
import json
import uuid
from typing import Any, Dict, List, Optional
from google.adk.runners import Runner
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.genai import types as genai_types
from opentelemetry import trace

from agent_eval_framework.adapters.base import BaseAgentAdapter, normalize_adapter_output
from agent_eval_framework.utils.logger import get_logger

log = get_logger(__name__)
tracer = trace.get_tracer(__name__)


class ADKAgentAdapter(BaseAgentAdapter):
    """Adapter for interacting with ADK agents (both mono-agent and multi-agent hierarchies)."""

    def __init__(
        self,
        agent_module: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_instance: Optional[Any] = None,
        user_id: str = "agent-eval-user",
        model_name: Optional[str] = None,
        artifact_service=None,
        session_service=None,
        memory_service=None,
        **kwargs,
    ):
        """Initializes the ADKAgentAdapter.

        Args:
            agent_module: Fully qualified module name (e.g. "my_app.agent").
            agent_name: Attribute name of the ADK agent (e.g. "root_agent").
            agent_instance: Pre-instantiated ADK Agent object.
            user_id: User ID for session management.
            model_name: Optional foundation model override (e.g. "gemini-3.7-flash").
            artifact_service: ADK ArtifactService instance.
            session_service: ADK SessionService instance.
            memory_service: ADK MemoryService instance.
        """
        self.agent_module = agent_module
        self.agent_name = agent_name
        self.user_id = user_id
        self.model_name = model_name
        self._kwargs = kwargs
        self._provided_instance = agent_instance
        self.agent = None

        super().__init__(**kwargs)

        if not self.agent:
            raise RuntimeError(f"ADK Agent could not be loaded from {agent_module}:{agent_name}")

        # Override model if requested
        if self.model_name and hasattr(self.agent, "model"):
            self.agent.model = self.model_name
            log.info(f"Dynamically set ADK Agent model to: {self.model_name}")

        self._artifact_service = artifact_service if artifact_service else InMemoryArtifactService()
        self._session_service = session_service if session_service else InMemorySessionService()
        self._memory_service = memory_service if memory_service else InMemoryMemoryService()

        self.runner = Runner(
            app_name="agent_eval_framework_adk_runner",
            agent=self.agent,
            artifact_service=self._artifact_service,
            session_service=self._session_service,
            memory_service=self._memory_service,
        )
        log.debug("ADK Runner initialized within Adapter.")

    def load_agent(self, **kwargs) -> Any:
        if self._provided_instance:
            return self._provided_instance
        if self.agent_module and self.agent_name:
            module = importlib.import_module(self.agent_module)
            agent_obj = getattr(module, self.agent_name)
            return agent_obj
        return None

    async def _run_agent_async(self, prompt: str) -> Dict[str, Any]:
        """Runs the ADK agent asynchronously and captures multi-agent and tool traces."""
        session_id = str(uuid.uuid4())
        content = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=prompt)],
        )

        final_response = ""
        tool_calls = []
        multi_agent_events = []
        full_conversation = []
        usage_metadata = None

        try:
            await self._session_service.create_session(
                app_name=self.runner.app_name,
                user_id=self.user_id,
                session_id=session_id,
            )

            async for event in self.runner.run_async(user_id=self.user_id, session_id=session_id, new_message=content):
                full_conversation.append(event.model_dump_json() if hasattr(event, "model_dump_json") else str(event))
                
                if hasattr(event, "usage_metadata") and event.usage_metadata:
                    usage_metadata = event.usage_metadata

                # Capture subagent author/delegation events in multi-agent systems
                if hasattr(event, "author") and event.author:
                    multi_agent_events.append({
                        "author": event.author,
                        "content": " ".join(part.text for part in event.content.parts if part.text) if event.content else "",
                    })

                if event.content:
                    text_content = " ".join(part.text for part in event.content.parts if part.text)
                    if hasattr(self.agent, "name") and event.author == self.agent.name:
                        final_response = text_content
                    elif not final_response:
                        final_response = text_content

                    for part in event.content.parts:
                        if part.function_call:
                            tool_calls.append({
                                "tool_name": part.function_call.name,
                                "tool_input": part.function_call.args,
                                "agent_author": getattr(event, "author", "root_agent"),
                            })

        except Exception as e:
            log.error(f"Error during ADK runner execution: {e}", exc_info=True)
            return normalize_adapter_output(
                response_text=f"ADK Execution Error: {e}",
                trajectory=tool_calls,
                error=str(e),
            )

        metadata = {
            "session_id": session_id,
            "subagent_count": len(set(e["author"] for e in multi_agent_events if "author" in e)),
        }
        if usage_metadata:
            metadata["usage_metadata"] = usage_metadata

        return normalize_adapter_output(
            response_text=final_response,
            trajectory=tool_calls,
            multi_agent_events=multi_agent_events,
            metadata=metadata,
        )

    def get_response(self, prompt: str) -> Dict[str, Any]:
        """Synchronously executes the ADK agent."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(self._run_agent_async(prompt))
            else:
                return loop.run_until_complete(self._run_agent_async(prompt))
        except RuntimeError:
            return asyncio.run(self._run_agent_async(prompt))
