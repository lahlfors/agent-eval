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

def test_run_evaluation(mocker):
    """Tests the `run_evaluation` function with mocked external dependencies.

    This test verifies that the `run_evaluation` function can be executed
    without making actual calls to Google Cloud services or loading large
    datasets. It mocks the Vertex AI SDK, GCS client, and data loading
    functions to ensure the core orchestration logic works as expected.

    Args:
        mocker: The pytest-mock fixture for mocking objects.
    """
    # Mock GCP calls
    mocker.patch('vertexai.init')
    mocker.patch('google.cloud.aiplatform.init')
    mock_eval_result = mocker.Mock()
    mock_eval_result.summary_metrics = {"some_metric": 1.0}
    mock_eval_result.metrics_table = pd.DataFrame() # Mock metrics_table as an empty DataFrame

    mock_eval_task_class = mocker.patch('vertexai.evaluation.EvalTask')
    mock_eval_task_instance = mock_eval_task_class.return_value
    mock_eval_task_instance.evaluate.return_value = mock_eval_result

    # Mock data loading
    mocker.patch(
        'personalized_shopping.shared_libraries.web_agent_site.envs.web_agent_text_env.load_products',
        return_value=([], {}, {}, defaultdict(set))
    )
    from agent_eval_framework.runner import run_evaluation
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "eval_config.yaml")
    assert os.path.exists(config_path), f"Config file not found: {config_path}"
    print(f"Running evaluation with config: {config_path}")
    eval_result = run_evaluation(config_path)
    assert eval_result is not None
    print("Eval result summary:", eval_result.summary_metrics)
