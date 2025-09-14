# Copyright 2025 Google LLC
# ... (license headers) ...

"""Defines and configures the Personalized Shopping Agent."""

from google.adk.agents import Agent
from google.adk.tools import FunctionTool
import sys
import os
import pathlib

# Add project root to handle imports when run from different contexts
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Absolute imports for tools and prompt
from personalized_shopping.tools.search import search
from personalized_shopping.tools.click import click
from personalized_shopping.prompt import personalized_shopping_agent_instruction

# Assuming logger is in agent_eval_framework
try:
    from agent_eval_framework.utils.logger import get_logger
    log = get_logger(__name__)
except ImportError:
    import logging
    log = logging.getLogger(__name__)
    log.warning("agent_eval_framework.utils.logger not found, using standard logging.")

# The primary agent for the personalized shopping experience.
root_agent = Agent(
    model="gemini-1.5-flash", # Or your preferred model
    name="personalized_shopping_agent",
    instruction=personalized_shopping_agent_instruction,
    tools=[
        FunctionTool(func=search),
        FunctionTool(func=click),
    ],
)
log.info("personalized_shopping.agent.root_agent initialized")
