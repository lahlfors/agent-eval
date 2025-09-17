import asyncio
import json
from google.adk.agents import Agent
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import importlib
from ..utils.logger import get_logger

log = get_logger(__name__)

class ADKAgentAdapter:
    def __init__(self, agent_module: str, agent_name: str, **kwargs):
        self.agent_module_str = agent_module
        self.agent_name = agent_name
        self.agent_config = kwargs
        self._load_agent_class()
        log.info(f"ADKAgentAdapter initialized for agent '{agent_name}'")

    def _load_agent_class(self):
        try:
            module = importlib.import_module(self.agent_module_str)
            log.debug(f"{self.agent_module_str} module loaded.")
            self.agent_class = getattr(module, self.agent_name)
            log.debug(f"Successfully loaded agent class: {self.agent_name}")
        except (ImportError, AttributeError) as e:
            log.error(f"Could not load agent class {self.agent_name} from {self.agent_module_str}", exc_info=True)
            raise ImportError(f"Could not load agent class {self.agent_name} from {self.agent_module_str}: {e}")

    def _parse_adk_output_to_dictionary(self, events: list[Event]):
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
            # Capture the last text part from the model as the final response
            if event.content.role == "model":
                for part in event.content.parts:
                    if getattr(part, "text", None):
                        final_response = part.text.strip()

        return {"response": final_response, "predicted_trajectory": trajectory}

    async def _run_agent_async(self, query: str):
        app_name = "eval_app"
        user_id = "eval_user"
        session_id = str(hash(query)) # Session ID based on query

        # Instantiate the agent
        try:
            agent = self.agent_class(**self.agent_config)
        except Exception as e:
            log.error(f"Error instantiating agent {self.agent_name}: {e}", exc_info=True)
            raise

        session_service = InMemorySessionService()
        await session_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id)

        runner = Runner(agent=agent, app_name=app_name, session_service=session_service)
        content = types.Content(role="user", parts=[types.Part(text=query)])
        events = []
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
            events.append(event)

        return self._parse_adk_output_to_dictionary(events)

    def __call__(self, prompt: str) -> dict:
        """
        Makes the adapter instance callable, as expected by EvalTask.evaluate(runnable=...).
        """
        try:
            log.debug(f"ADKAgentAdapter called with prompt: {prompt}")
            result = asyncio.run(self._run_agent_async(prompt))

            # Wrap trajectory in {"tool_calls": [...]} and serialize to JSON
            predicted_trajectory_list = result.get("predicted_trajectory", [])
            wrapped_trajectory = {"tool_calls": predicted_trajectory_list}
            result["predicted_trajectory"] = json.dumps(wrapped_trajectory)

            log.debug(f"ADKAgentAdapter result: {result}")
            return result
        except Exception as e:
            log.error(f"Error during ADK agent execution: {e}", exc_info=True)
            return {
                "response": "AGENT_EXECUTION_ERROR",
                "predicted_trajectory": json.dumps({"tool_calls": []}),
                "error": str(e)
            }
