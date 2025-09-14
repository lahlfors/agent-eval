# Copyright 2025 Google LLC
# ... (license headers) ...

"""Defines and configures the Personalized Shopping Agent."""

from google.adk.agents import Agent
from google.adk.tools import FunctionTool

# Absolute imports from the personalized_shopping package
from personalized_shopping.tools.search import search
from personalized_shopping.tools.click import click
from personalized_shopping.prompt import personalized_shopping_agent_instruction

# The primary agent for the personalized shopping experience.
root_agent = Agent(
    model="gemini-2.5-flash", # Or your preferred model
    name="personalized_shopping_agent",
    instruction=personalized_shopping_agent_instruction,
    tools=[
        FunctionTool(func=search),
        FunctionTool(func=click),
    ],
)

print("personalized_shopping.agent module loaded and root_agent defined.")
