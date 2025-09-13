import pytest
import os
import pathlib
from agent_eval_framework.runner import run_evaluation
from dotenv import load_dotenv

@pytest.fixture(scope="session", autouse=True)
def setup_env():
    project_root = pathlib.Path(__file__).parent.parent
    dotenv_path = project_root / ".env"
    if dotenv_path.exists():
        print(f"Loading environment variables from: {dotenv_path}")
        load_dotenv(dotenv_path=dotenv_path)
    else:
        print(f"Warning: .env file not found at {dotenv_path}")

def test_run_evaluation():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "eval_config.yaml")
    assert os.path.exists(config_path), f"Config file not found: {config_path}"
    eval_result = run_evaluation(config_path)
    assert eval_result is not None
    print("Eval result summary:", eval_result.summary_metrics)
