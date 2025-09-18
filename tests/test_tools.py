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

"""Unit tests for the tools of the Personalized Shopping Agent."""

import os
import dotenv
import pytest
from google.adk.evaluation.agent_evaluator import AgentEvaluator

pytest_plugins = ("pytest_asyncio",)


@pytest.fixture(scope="session", autouse=True)
def load_env():
    """Loads environment variables from a .env file.

    This is a session-scoped autouse fixture, so it runs once before any
    tests in this file and ensures that the environment is configured.
    """
    dotenv.load_dotenv()


@pytest.mark.asyncio
async def test_tools():
    """Runs the ADK's built-in evaluator to test the agent's tools.

    This test uses the `AgentEvaluator` from the ADK to run a series of
    pre-defined test cases against the `personalized_shopping` agent. The
    test cases are defined in the `tests/tools` directory and are designed
    to validate the basic functionality of the agent's tools (`search` and
    `click`).
    """
    await AgentEvaluator.evaluate(
        "personalized_shopping",
        os.path.join(os.path.dirname(__file__), "tools"),
        num_runs=1,
    )
