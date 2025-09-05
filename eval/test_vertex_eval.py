import os
import pytest

# The run_evaluation function is now imported from the installed framework
from agent_eval_framework.runner import run_evaluation

def test_shopping_agent_evaluation():
    """
    Runs the evaluation for the personalized shopping agent
    by calling the generic evaluation framework.
    """
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    try:
        # The framework's runner will handle loading .env, checking for GCP vars, etc.
        # We just need to call it.
        result = run_evaluation(config_path)

        # A simple assertion to ensure the evaluation ran and returned a result.
        assert result is not None
    except EnvironmentError as e:
        # The framework raises an EnvironmentError if GCP vars are not set.
        # We can use pytest.skip to gracefully skip the test in this case.
        pytest.skip(f"Skipping evaluation: {e}")
    except Exception as e:
        pytest.fail(f"Evaluation framework failed with an unexpected error: {e}")
