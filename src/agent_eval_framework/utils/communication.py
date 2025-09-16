# src/agent_eval_framework/utils/communication.py
import json
from opentelemetry import trace
from opentelemetry.trace import Span
from contextlib import contextmanager
from typing import Any, Dict, Optional, Callable
import functools
from . import otel_config
from .logging_utils import log

otel_config.setup_opentelemetry()
TRACER_NAME = "agent_eval_framework.communication"
tracer = otel_config.get_tracer(TRACER_NAME)

def get_tracer():
    return tracer

@contextmanager
def trace_tool_invocation(tool_name: str, parameters: Optional[Dict[str, Any]] = None):
    with tracer.start_as_current_span(f"Tool.{tool_name}") as span:
        span.set_attribute("tool.name", tool_name)
        if parameters:
            try:
                param_str = json.dumps(parameters)
                if len(param_str) > 5000:
                    param_str = param_str[:5000] + "...(truncated)"
                span.set_attribute("tool.parameters", param_str)
            except TypeError as e:
                log.warning(f"Could not serialize parameters for tool {tool_name}: {e}")
                span.set_attribute("tool.parameters", str(parameters))
        try:
            yield span
            span.set_attribute("tool.status", "SUCCESS")
        except Exception as e:
            span.set_attribute("tool.status", "ERROR")
            span.record_exception(e)
            log.error(f"Error during tool invocation {tool_name}: {e}", exc_info=False)
            raise

def set_tool_output(span: Span, output: Any):
    if span and span.is_recording():
        try:
            output_str = json.dumps(output)
            if len(output_str) > 5000:
                output_str = output_str[:5000] + "...(truncated)"
            span.set_attribute("tool.return_value", output_str)
        except TypeError as e:
            log.warning(f"Could not serialize output for tool span: {e}")
            span.set_attribute("tool.return_value", str(output))

def trace_tool_function(tool_function: Callable) -> Callable:
    tool_name = tool_function.__name__ # Get function name
    @functools.wraps(tool_function)
    def wrapper(*args, **kwargs):
        parameters = {"args": args, "kwargs": kwargs}
        with tracer.start_as_current_span(f"Tool.{tool_name}") as span:
            span.set_attribute("tool.name", tool_name)
            try:
                # Attempt to serialize parameters, handling potential type errors
                param_str = json.dumps(parameters)
                if len(param_str) > 5000:
                    param_str = param_str[:5000] + "...(truncated)"
                span.set_attribute("tool.parameters", param_str)
            except TypeError as e:
                log.warning(f"Could not serialize parameters for tool {tool_name}: {e}")
                span.set_attribute("tool.parameters", str(parameters))
            try:
                result = tool_function(*args, **kwargs)
                span.set_attribute("tool.status", "SUCCESS")
                set_tool_output(span, result)
                return result
            except Exception as e:
                span.set_attribute("tool.status", "ERROR")
                span.record_exception(e)
                log.error(f"Error during tool execution {tool_name}: {e}", exc_info=False)
                raise
        return wrapper
    return wrapper
