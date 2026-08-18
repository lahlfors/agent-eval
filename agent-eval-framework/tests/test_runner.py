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

"""Unit tests for the agent evaluation runner."""

import pytest
import os
import sys
import pathlib
from collections import defaultdict
from dotenv import load_dotenv
import pandas as pd

# --- Add project root to sys.path ---
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
print(f"Adjusted sys.path for test: {sys.path}")

@pytest.fixture(scope="session", autouse=True)
def setup_env():
    """Loads environment variables from the .env file at the project root.

    This is a session-scoped autouse fixture, so it runs once before any
    tests in this file and ensures that the environment is configured.
    """
    dotenv_path = PROJECT_ROOT / ".env"
    if dotenv_path.exists():
        print(f"Loading environment variables from: {dotenv_path}")
        load_dotenv(dotenv_path=dotenv_path, override=True)
    else:
        print(f"Warning: .env file not found at {dotenv_path}")

def test_run_evaluation_mono_agent(mocker):
    """Tests the `run_evaluation` function for a mono-agent system with mocked Vertex AI."""
    import agent_eval_framework.runner

    # Mock GCP calls
    mocker.patch('agent_eval_framework.runner.vertexai.init')
    mocker.patch('agent_eval_framework.runner.aiplatform.init')
    mock_eval_result = mocker.Mock()
    mock_eval_result.summary_metrics = {"exact_match": 1.0, "token_cost_usd": 0.0001}
    mock_eval_result.metrics_table = pd.DataFrame()

    mock_eval_task_class = mocker.patch('agent_eval_framework.runner.EvalTask')
    mock_eval_task_instance = mock_eval_task_class.return_value
    mock_eval_task_instance.evaluate.return_value = mock_eval_result

    from agent_eval_framework.runner import run_evaluation
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "eval_config.yaml")
    assert os.path.exists(config_path), f"Config file not found: {config_path}"
    eval_result = run_evaluation(config_path)
    assert eval_result is not None
    assert eval_result.summary_metrics["exact_match"] == 1.0


def test_run_evaluation_multi_agent(mocker):
    """Tests the `run_evaluation` function for a multi-agent system."""
    import agent_eval_framework.runner

    mocker.patch('agent_eval_framework.runner.vertexai.init')
    mocker.patch('agent_eval_framework.runner.aiplatform.init')
    mock_eval_result = mocker.Mock()
    mock_eval_result.summary_metrics = {"trajectory_exact_match": 1.0, "cost_savings_multiplier": 18.5}
    mock_eval_result.metrics_table = pd.DataFrame()

    mock_eval_task_class = mocker.patch('agent_eval_framework.runner.EvalTask')
    mock_eval_task_instance = mock_eval_task_class.return_value
    mock_eval_task_instance.evaluate.return_value = mock_eval_result

    from agent_eval_framework.runner import run_evaluation
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "multi_agent_eval_config.yaml")
    assert os.path.exists(config_path), f"Config file not found: {config_path}"
    eval_result = run_evaluation(config_path)
    assert eval_result is not None
    assert eval_result.summary_metrics["cost_savings_multiplier"] == 18.5
