"""Provides the `click` tool for the Personalized Shopping Agent.

This module defines the `click` tool, which allows the agent to interact with
the simulated web environment by clicking on buttons.
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

from google.adk.tools import ToolContext
from google.genai import types

from ..shared_libraries.init_env import webshop_env

async def click(button_name: str, tool_context: ToolContext) -> str:
    """Simulates clicking a button in the web environment.

    This tool takes the name of a button, simulates a click action in the
    `webshop_env`, and returns the new state of the web page observation.
    It also logs the status and observation and attempts to save the resulting
    HTML as a tool artifact.

    Args:
        button_name: The name or label of the button to be clicked.
        tool_context: The context provided by the ADK, used here for saving
                      artifacts.

    Returns:
        The new web page observation (as a string) after the click action
        has been performed.
    """
    status = {"reward": None, "done": False}
    action_string = f"click[{button_name}]"
    _, status["reward"], status["done"], _ = webshop_env.step(action_string)

    ob = webshop_env.observation
    index = ob.find("Back to Search")
    if index >= 0:
        ob = ob[index:]

    print("#" * 50)
    print("Click result:")
    print(f"status: {status}")
    print(f"observation: {ob}")
    print("#" * 50)

    if button_name == "Back to Search":
        webshop_env.server.assigned_instruction_text = "Back to Search"

    # Show artifact in the UI.
    try:
        await tool_context.save_artifact(
            "html",
            types.Part.from_uri(
                file_uri=webshop_env.state["html"], mime_type="text/html"
            ),
        )
    except ValueError as e:
        print(f"Error saving artifact: {e}")
    return ob
