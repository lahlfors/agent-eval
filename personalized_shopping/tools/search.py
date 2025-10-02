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

"""Provides the `search` tool for the Personalized Shopping Agent."""

import os
import pathlib
import sys

from google.adk.tools import ToolContext
from vertexai.generative_models import Part

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

# Absolute import from within the personalized_shopping package
from personalized_shopping.shared_libraries.init_env import get_webshop_env


async def search(keywords: str, tool_context: ToolContext) -> str:
    """Performs a keyword search in the web shopping environment.

    This tool takes a string of keywords, executes a search action in the
    simulation, and returns the new observation (the text content of the

    search results page). It also saves the full HTML of the results page as a
    tool artifact for debugging.

    Args:
        keywords: The search terms to query.
        tool_context: The context object provided by the ADK, used for
            saving artifacts.

    Returns:
        A string representing the search results page's observation text.
    """
    webshop_env = get_webshop_env()
    status = {"reward": None, "done": False}
    action_string = f"search[{keywords}]"
    log.info(f"Performing search: {action_string}")
    webshop_env.server.assigned_instruction_text = f"Find me {keywords}."
    _, status["reward"], status["done"], _ = webshop_env.step(action_string)

    ob = webshop_env.observation
    index = ob.find("Back to Search")
    if index >= 0:
        ob = ob[index:]

    log.info("Search complete", extra={"status": status})

    try:
        # Create the artifact object from the HTML data and its MIME type
        html_artifact = Part.from_data(
            data=webshop_env.state["html"].encode("utf-8"),
            mime_type="text/html"
        )

        # Call the save_artifact method with the correct arguments
        await tool_context.save_artifact(
            filename=f"Search Results for {keywords}.html",
            artifact=html_artifact
        )
    except Exception as e:
        log.warning(f"Error saving search artifact: {e}", exc_info=True)
    return ob