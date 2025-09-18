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

"""Integration tests for the agent evaluation runner with Vertex AI.

These tests run the full evaluation pipeline, including making live calls to
the Vertex AI API and a deployed agent. They are designed to be run in an
environment where Google Cloud credentials are properly configured.
"""

import pytest
import os
from google.cloud import aiplatform
from google.api_core import exceptions
from agent_eval_framework.runner import run_evaluation
import uuid
from datetime import datetime
import yaml

def load_config(config_path):
    """Loads a YAML configuration file.

    Args:
        config_path: The path to the YAML file.

    Returns:
        A dictionary containing the loaded configuration.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

CONFIG_PATH = "agent-eval-framework/config/adk_eval_config.yaml"

@pytest.fixture(scope="session")
def eval_config():
    """Provides the evaluation configuration loaded from the YAML file.

    This is a session-scoped fixture, so the configuration is loaded only
    once per test session.

    Returns:
        A dictionary with the evaluation configuration.
    """
    return load_config(CONFIG_PATH)

@pytest.fixture(scope="function")
def unique_run_suffix():
    """Generates a unique suffix for a test run name.

    This ensures that each test execution creates a new, non-conflicting run
    in Vertex AI Experiments.

    Returns:
        A unique string combining a timestamp and a UUID.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"{timestamp}-{unique_id}"

@pytest.fixture(scope="function")
def vertex_ai_context_manager(eval_config, unique_run_suffix):
    """Initializes Vertex AI and yields a unique run name for a test.

    This fixture handles the setup of the Vertex AI environment and is
    intended to also handle the teardown (cleanup) of the created experiment
    run, although the cleanup is currently disabled.

    Args:
        eval_config: The evaluation configuration dictionary.
        unique_run_suffix: The unique suffix for the run name.

    Yields:
        The full, unique experiment run name for the test.
    """
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION")

    experiment_name = eval_config.get("experiment_name", "default-experiment")
    run_name_prefix = eval_config.get("run_name_prefix", "run")

    experiment_run_name = f"{run_name_prefix}-{unique_run_suffix}"
    actual_context_id = f"{experiment_name}-{experiment_run_name}"

    aiplatform.init(project=project_id, location=location)

    yield experiment_run_name

    print("Teardown disabled. Run will be preserved in Vertex AI.")

def test_shopping_agent_vertex_eval(vertex_ai_context_manager, eval_config):
    """Runs an end-to-end evaluation using the live Vertex AI service.

    This test executes the `run_evaluation` function with a specific configuration
    that points to a live, deployed agent. It verifies that the evaluation
    completes successfully and produces a result. This test is skipped if
    GCP environment variables are not set.

    Args:
        vertex_ai_context_manager: The fixture that provides the unique run name.
        eval_config: The fixture that provides the evaluation configuration.
    """
    if not os.getenv("GOOGLE_CLOUD_PROJECT") or not os.getenv("GOOGLE_CLOUD_LOCATION"):
        pytest.skip("GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION must be set in .env file to run Vertex AI evaluations.")

    experiment_run_name = vertex_ai_context_manager

    print(f"Running evaluation with config: {CONFIG_PATH}")
    print(f"Using experiment run name: {experiment_run_name}")
    actual_context_id = f"{eval_config.get('experiment_name')}-{experiment_run_name}"
    print(f"Expecting actual context ID to be: {actual_context_id}")

    try:
        eval_result = run_evaluation(config_path=CONFIG_PATH, experiment_run_name=experiment_run_name)

        assert eval_result is not None, "Evaluation failed to produce results."
        print("Evaluation completed successfully using agent-eval-framework.")
    except Exception as e:
        pytest.fail(f"run_evaluation failed: {e}")
