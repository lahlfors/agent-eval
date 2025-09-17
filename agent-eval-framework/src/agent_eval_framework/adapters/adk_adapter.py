import asyncio
import importlib
from typing import Any, Dict, List
import json
import sys
from vertexai.preview.reasoning_engines import AdkApp
from opentelemetry import trace
from .. import otel_config

class ADKAgentAdapter:
    def __init__(self, agent_module: str, agent_name: str = "root_agent", **kwargs):
        self.agent_module_str = agent_module
        self.agent_name = agent_name
        self.agent_config = kwargs
        self.tracer = trace.get_tracer(__name__)
        sys.stdout.write(f"ADKAgentAdapter: Tracer type in __init__: {type(self.tracer)}\n")
        sys.stdout.flush()
        self._load_and_wrap_agent()
        sys.stdout.write(f"ADKAgentAdapter initialized for agent '{self.agent_name}' using AdkApp with tracing enabled.\n")
        sys.stdout.flush()

    def _load_and_wrap_agent(self):
        try:
            module = importlib.import_module(self.agent_module_str)
            sys.stdout.write(f"{self.agent_module_str} module loaded.\n")
            sys.stdout.flush()
            agent = getattr(module, self.agent_name)
            sys.stdout.write(f"Successfully loaded agent: {self.agent_name}\n")
            sys.stdout.flush()

            self.adk_app = AdkApp(agent=agent, enable_tracing=True)
            sys.stdout.write("Agent wrapped with vertexai.preview.reasoning_engines.AdkApp with enable_tracing=True\n")
            sys.stdout.flush()
        except (ImportError, AttributeError) as e:
            sys.stderr.write(f"Could not load agent {self.agent_name} from {self.agent_module_str}: {e}\n")
            sys.stderr.flush()
            raise ImportError(f"Could not load agent {self.agent_name} from {self.agent_module_str}: {e}")

    def _parse_adk_output_to_dictionary(self, events: list[dict]):
        final_response = ""
        trajectory = []
        for event in events:
            content = event.get('content')
            if not content or not content.get('parts'): continue
            for part in content['parts']:
                if 'functionCall' in part:
                    func_call = part['functionCall']
                    info = {"tool_name": func_call.get("name"), "tool_input": func_call.get("args", {})}
                    if info not in trajectory: trajectory.append(info)
                if content.get('role') == "model" and 'text' in part:
                    final_response = part['text'].strip()
        return {"response": final_response, "predicted_trajectory_list": trajectory}

    async def _run_agent_async(self, query: str) -> Dict[str, Any]:
        otel_config.log_otel_status("ADKAgentAdapter._run_agent_async START")
        user_id = "eval_user"
        events = []
        try:
            async for event in self.adk_app.async_stream_query(user_id=user_id, message=query):
                events.append(event)
            return self._parse_adk_output_to_dictionary(events)
        except Exception as e:
            sys.stderr.write(f"Error during ADK agent async_stream_query: {e}\n")
            sys.stderr.flush()
            return {"response": "ADK_APP_ERROR", "predicted_trajectory_list": [], "error": str(e)}

    def __call__(self, prompt: str) -> dict:
        sys.stdout.write(f"ADKAgentAdapter.__call__ called with prompt: {prompt}\n")
        sys.stdout.flush()
        otel_config.log_otel_status("ADKAgentAdapter.__call__ START")
        local_tracer = trace.get_tracer(__name__)
        sys.stdout.write(f"ADKAgentAdapter: Tracer type in __call__: {type(local_tracer)}\n")
        sys.stdout.flush()
        with local_tracer.start_as_current_span("ADKAgentAdapter.__call__") as span:
            sys.stdout.write(f"Span 'ADKAgentAdapter.__call__' created: {span.get_span_context().is_valid}, Recording: {span.is_recording()}\n")
            sys.stdout.flush()
            span.set_attribute("prompt", prompt)
            try:
                result = asyncio.run(self._run_agent_async(prompt))
                predicted_trajectory_list = result.get("predicted_trajectory_list", [])
                # Return the raw list for the mapping
                result["predicted_trajectory"] = predicted_trajectory_list
                result["response"] = result.get("response", "")
                span.set_attribute("response", result.get("response"))
                return result
            except Exception as e:
                sys.stderr.write(f"Error in ADKAgentAdapter __call__: {e}\n")
                sys.stderr.flush()
                if span.is_recording():
                    span.record_exception(e)
                return {
                    "response": "ADAPTER_ERROR",
                    "predicted_trajectory": [], # Raw list
                    "error": str(e)
                }
