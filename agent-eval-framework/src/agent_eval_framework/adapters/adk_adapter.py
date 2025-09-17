# agent-eval-framework/src/agent_eval_framework/adapters/adk_adapter.py
import asyncio
import importlib
from typing import Any, Dict
from vertexai.preview.reasoning_engines import AdkApp
from google.genai import types as genai_types
import types as python_types

class ADKAgentAdapter:
    def __init__(self, agent_module: str, agent_name: str = "root_agent", **kwargs):
        try:
            module = importlib.import_module(agent_module)
            # Handle potential SimpleNamespace wrapper from previous fixes
            if hasattr(module, "agent") and hasattr(module.agent, agent_name):
                 agent = module.agent.root_agent
            else:
                 agent = getattr(module, agent_name)

            self.adk_app = AdkApp(agent=agent, enable_tracing=True)
            self.agent_name = agent.name
            print(f"ADKAgentAdapter initialized for agent '{self.agent_name}' with AdkApp tracing enabled.")
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not load agent {agent_name} from {agent_module}: {e}")

    async def _get_response_async(self, prompt: str) -> Dict[str, Any]:
        final_text = ""
        tool_calls = []
        user_id = "eval_user"
        try:
            async for event in self.adk_app.async_stream_query(user_id=user_id, message=prompt):
                content = event.get('content')
                if content and content.get('parts'):
                    for part in content['parts']:
                        if 'text' in part:
                            final_text += part['text']
                        # Capture the full tool call details as per Vertex AI eval docs
                        if 'tool_code' in part and part.get('tool_code'):
                            tool_call_data = {
                                "tool_name": part['tool_code'].get('name'),
                                "tool_input": part['tool_code'].get('args'),
                            }
                            if tool_call_data["tool_name"]:
                                tool_calls.append(tool_call_data)

            return {
                "response": final_text,
                "predicted_trajectory": tool_calls
            }
        except Exception as e:
            print(f"Error during ADK agent async_stream_query: {e}")
            return {"response": "ADK_APP_ERROR", "error": str(e), "predicted_trajectory": []}

    def get_response(self, prompt: str) -> Dict[str, Any]:
        # Run the async method in a new event loop
        return asyncio.run(self._get_response_async(prompt))
