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

"""Defines and configures the Personalized Shopping Agent."""

from google.adk.agents import Agent
from google.adk.tools import FunctionTool
import sys
import os
import pathlib

# Absolute imports from the personalized_shopping package
from personalized_shopping.tools.search import search
from personalized_shopping.tools.click import click
from personalized_shopping.prompt import personalized_shopping_agent_instruction

# The primary agent for the personalized shopping experience.
root_agent = Agent(
    model="gemini-1.5-flash", # Consistent model name
    name="personalized_shopping_agent",
    instruction=personalized_shopping_agent_instruction,
    tools=[
        FunctionTool(func=search),
        FunctionTool(func=click),
    ],
)

print("personalized_shopping.agent module loaded and root_agent defined.")
