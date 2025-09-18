import asyncio
import json
import importlib
from typing import Any, Dict, List
import types as python_types
import uuid

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from google.adk.agents import Agent
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

# Assuming get_logger is in agent_eval_framework.utils.logger
# from ..utils.logger import get_logger
# Placeholder logger if the above import fails
import logging
log = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

class ADKAgentAdapter:
    def __init__(self, agent_module: str, agent_name: str = "root_agent", app_name: str = "eval_app", user_id: str = "eval_user", **kwargs):
        self.agent_module_str = agent_module
        self.agent_name = agent_name
        self.agent_config = kwargs
        self.app_name = app_name
        self.user_id = user_id
        self._load_agent_class()
        log.info(f"ADKAgentAdapter initialized for agent '{self.agent_name}'")

    def _load_agent_class(self):
        try:
            module = importlib.import_module(self.agent_module_str)
            log.debug(f"{self.agent_module_str} module loaded.")
            # Standardize on accessing the agent class directly
            self.agent_class = getattr(module, self.agent_name)
            log.debug(f"Successfully loaded agent class: {self.agent_name}")
        except ImportError as e:
            log.error(f"ImportError for {self.agent_name} from {self.agent_module_str}", exc_info=True)
            raise ImportError(f"Could not import module {self.agent_module_str}: {e}") from e
        except AttributeError as e:
            log.error(f"AttributeError: Class {self.agent_name} not found in {self.agent_module_str}", exc_info=True)
            raise AttributeError(f"Class {self.agent_name} not found in {self.agent_module_str}: {e}") from e

    def _parse_adk_output_to_dictionary(self, events: List[Event]) -> Dict[str, Any]:
        final_response = ""
        trajectory = []
        for event in events:
            if not getattr(event, "content", None) or not getattr(event.content, "parts", None):
                continue
            for part in event.content.parts:
                if getattr(part, "function_call", None):
                    info = {
                        "tool_name": part.function_call.name,
                        "tool_input": dict(part.function_call.args),
                    }
                    if info not in trajectory:
                        trajectory.append(info)
            if event.content.role == "model":
                for part in event.content.parts:
                    if getattr(part, "text", None) is not None: # Check for non-None text
                        final_response = part.text.strip()
        return {"response": final_response, "predicted_trajectory": trajectory}

    async def _run_agent_async(self, query: str) -> Dict[str, Any]:
        with tracer.start_as_current_span("ADKAgentAdapter._run_agent_async") as span:
            span.set_attribute("agent.name", self.agent_name)
            session_id = str(uuid.uuid4())
            span.set_attribute("session.id", session_id)

            try:
                agent = self.agent_class(**self.agent_config)
            except Exception as e:
                log.error(f"Error instantiating agent {self.agent_name}: {e}", exc_info=True)
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, f"Agent instantiation failed: {e}"))
                raise

            session_service = InMemorySessionService()
            await session_service.create_session(app_name=self.app_name, user_id=self.user_id, session_id=session_id)

            runner = Runner(agent=agent, app_name=self.app_name, session_service=session_service)
            content = genai_types.Content(role="user", parts=[genai_types.Part(text=query)])
            events = []

            with tracer.start_as_current_span("Runner.run_async") as runner_span:
                try:
                    async for event in runner.run_async(user_id=self.user_id, session_id=session_id, new_message=content):
                        events.append(event)
                except Exception as e:
                    log.error(f"Error during runner.run_async: {e}", exc_info=True)
                    runner_span.record_exception(e)
                    runner_span.set_status(Status(StatusCode.ERROR, f"runner.run_async failed: {e}"))
                    raise

            parsed_output = self._parse_adk_output_to_dictionary(events)
            span.set_attribute("output.response_length", len(parsed_output.get("response", "")))
            return parsed_output

    def __call__(self, prompt: str) -> Dict[str, Any]:
        """
        Makes the adapter instance callable, as expected by EvalTask.evaluate(runnable=...).
        """
        with tracer.start_as_current_span("ADKAgentAdapter.call") as span:
            span.set_attribute("agent.name", self.agent_name)
            span.set_attribute("input.prompt", prompt)
            try:
                log.debug(f"ADKAgentAdapter called with prompt: {prompt}")
                result = asyncio.run(self._run_agent_async(prompt))

                predicted_trajectory_list = result.get("predicted_trajectory", [])
                wrapped_trajectory = {"tool_calls": predicted_trajectory_list}

                # Format for evaluation
                return {
                    "actual_response": result.get("response"),
                    "predicted_trajectory": json.dumps(wrapped_trajectory)
                }
            except Exception as e:
                log.error(f"Error during ADK agent execution: {e}", exc_info=True)
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, f"ADKAgentAdapter.call failed: {e}"))
                return {
                    "actual_response": "AGENT_EXECUTION_ERROR",
                    "predicted_trajectory": json.dumps({"tool_calls": []}),
                    "error": str(e)
                }

    def get_response(self, prompt: str) -> Dict[str, Any]:
        # Alias for __call__ if needed, or adapt to the evaluator's expected method name
        return self.__call__(prompt)
