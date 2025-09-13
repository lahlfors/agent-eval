"""Defines and configures the Personalized Shopping Agent.

This module initializes the primary agent for the personalized shopping
experience. It configures the agent with a specific language model, a name,
a set of instructions from the `prompt` module, and the tools it can use
(search and click). This agent is the central component that processes user
requests and orchestrates tool calls to generate responses.
"""
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

import types
from google.adk.agents import Agent
from google.adk.tools import FunctionTool

from .tools.search import search
from .tools.click import click

from .prompt import personalized_shopping_agent_instruction

# The primary agent for the personalized shopping experience.
root_agent = Agent(
    model="gemini-2.5-flash",
    name="personalized_shopping_agent",
    instruction=personalized_shopping_agent_instruction,
    tools=[
        FunctionTool(
            func=search,
        ),
        FunctionTool(
            func=click,
        ),
    ],
)

# --- WORKAROUND START ---
# Create a namespace object called 'agent'
agent = types.SimpleNamespace(root_agent=root_agent)
# --- WORKAROUND END ---
