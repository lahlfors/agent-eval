# Copyright 2025 Google LLC
# ... (license headers) ...

"""Provides the `click` tool for the Personalized Shopping Agent."""

from google.adk.tools import ToolContext
from google.genai import types

# Absolute import
from personalized_shopping.shared_libraries.init_env import get_webshop_env
import sys
import os
import pathlib

# Add agent-eval-framework to sys.path to find logger
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
FRAMEWORK_SRC = os.path.join(PROJECT_ROOT, "agent-eval-framework", "src")
if FRAMEWORK_SRC not in sys.path:
    sys.path.insert(0, FRAMEWORK_SRC)
try:
    from agent_eval_framework.utils.logger import get_logger
    log = get_logger(__name__)
except ImportError:
    import logging
    log = logging.getLogger(__name__)

async def click(button_name: str, tool_context: ToolContext) -> str:
    """Simulates clicking a button in the web environment."""
    webshop_env = get_webshop_env()
    status = {"reward": None, "done": False}
    action_string = f"click[{button_name}]"
    log.info(f"Performing click: {action_string}")
    _, status["reward"], status["done"], _ = webshop_env.step(action_string)

    ob = webshop_env.unwrapped.observation
    index = ob.find("Back to Search")
    if index >= 0:
        ob = ob[index:]

    log.info("Click complete", extra={"status": status})
    # print(f"observation: {ob}") # Optional: can be very long

    if button_name == "Back to Search":
        webshop_env.unwrapped.server.assigned_instruction_text = "Back to Search"

    try:
        await tool_context.save_artifact(
            artifact=types.ContentDict(
                parts=[{"text": webshop_env.unwrapped.state["html"]}]
            ),
            filename="click_artifact.html"  # Add this line
        )
    except Exception as e:
        log.warning(f"Error saving click artifact: {e}", exc_info=True)
    return ob
