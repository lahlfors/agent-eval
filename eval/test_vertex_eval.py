"""Test script for running evaluation on the deployed Vertex AI agent.

This script uses `pytest` to trigger an evaluation run. It leverages the
`agent-eval-framework` by importing and calling the `run_evaluation` function,
pointing it to the local `config.yaml` file.

This demonstrates how a consumer project (the `personalized_shopping` agent)
can easily run a comprehensive, framework-driven evaluation with a simple
test script.
"""

import os
import pytest

# The run_evaluation function is now imported from the installed framework
from agent_eval_framework.runner import run_evaluation

def test_shopping_agent_evaluation():
    """Runs the full evaluation for the personalized shopping agent.

    This test function serves as the entry point for the evaluation. It calls
    the generic `run_evaluation` function from the framework, which then
    handles all the complex orchestration of loading data, calling the agent,
    and computing metrics.

    The test includes error handling to gracefully skip the test if the
    required GCP environment variables are not set, and it will fail on any
    other unexpected errors during the evaluation process.
    """
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    try:
        result = run_evaluation(config_path)
        # A simple assertion to ensure the evaluation ran and returned a result.
        assert result is not None, "The evaluation framework did not return a result."
    except EnvironmentError as e:
        # The framework raises an EnvironmentError if GCP vars are not set.
        # We use pytest.skip to gracefully skip the test in CI/CD environments
        # where credentials might not be configured.
        pytest.skip(f"Skipping evaluation due to missing environment configuration: {e}")
    except Exception as e:
        pytest.fail(f"Evaluation framework failed with an unexpected error: {e}")
