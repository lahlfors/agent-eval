import os
import json
import pytest
import dotenv
import yaml
import importlib
from google.cloud import aiplatform

def load_class(import_str: str):
    """Dynamically loads a class from a string path."""
    module_path, class_name = import_str.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

@pytest.fixture(scope="session", autouse=True)
def load_env():
    """Loads environment variables from .env file."""
    dotenv.load_dotenv()

def test_vertex_evaluation_with_config():
    """
    Tests the agent using a config-driven, adapter-based approach.
    """
    # 1. Load Configuration
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Setup GCP and AI Platform
    project_id = os.getenv("GCP_PROJECT_ID")
    location = os.getenv("GCP_REGION")
    if not project_id or not location:
        pytest.skip("GCP_PROJECT_ID and GCP_REGION environment variables must be set.")
    aiplatform.init(project=project_id, location=location)

    # 3. Instantiate Agent Adapter
    # Override agent_engine_id from config with environment variable if it exists for security
    agent_engine_id_from_env = os.getenv("AGENT_ENGINE_ID")
    if agent_engine_id_from_env:
        config["agent_config"]["agent_engine_id"] = agent_engine_id_from_env

    adapter_class = load_class(config["agent_adapter_class"])
    adapter = adapter_class(**config["agent_config"])

    # 4. Load Golden Dataset (path is relative to project root)
    dataset_path = config["dataset_path"]
    with open(dataset_path, "r") as f:
        golden_dataset = [json.loads(line) for line in f]

    # 5. Generate Actual Responses
    for record in golden_dataset:
        prompt = record["prompt"]
        try:
            # Use the adapter to get the response
            response_data = adapter.get_response(prompt)
            record["actual_response"] = response_data["actual_response"]
        except Exception as e:
            pytest.fail(f"Adapter failed for prompt '{prompt}': {e}")

    # 6. Define and Run Evaluation Task
    eval_task = aiplatform.evaluate.EvalTask(
        dataset=golden_dataset,
        metrics=config["metrics"],
        response_column="actual_response",
        reference_column="reference_response",
    )
    result = eval_task.evaluate()

    # 7. Print and Assert
    print("Evaluation results:")
    print(result.metrics_table)
    assert result is not None
    assert config["metrics"][0] in result.metrics_table.columns
