import pytest
import os
from google.cloud import aiplatform
from google.api_core import exceptions
from agent_eval_framework.runner import run_evaluation
import uuid
from datetime import datetime
import yaml

# Helper to load config
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

CONFIG_PATH = "agent-eval-framework/config/adk_eval_config.yaml"

@pytest.fixture(scope="session")
def eval_config():
    return load_config(CONFIG_PATH)

@pytest.fixture(scope="function")
def unique_run_suffix():
    # Generate a unique suffix for each test run
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"{timestamp}-{unique_id}"

@pytest.fixture(scope="function")
def vertex_ai_context_manager(eval_config, unique_run_suffix):
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION")

    experiment_name = eval_config.get("experiment_name", "default-experiment")
    run_name_prefix = eval_config.get("run_name_prefix", "run")

    # This is the name passed to start_run
    experiment_run_name = f"{run_name_prefix}-{unique_run_suffix}"

    # THIS IS THE CRITICAL PART: Reconstruct the ID that Vertex AI seems to be creating
    actual_context_id = f"{experiment_name}-{experiment_run_name}"

    aiplatform.init(project=project_id, location=location)

    yield experiment_run_name # Pass the intended run name to the test

    # --- CLEANUP TEMPORARILY DISABLED ---
    print("Teardown disabled. Run will be preserved in Vertex AI.")
    # print(f"Tearing down: Attempting to delete context {actual_context_id}")
    # try:
    #     context_to_delete = aiplatform.Context.get(resource_id=actual_context_id)
    #     context_to_delete.delete()
    #     print(f"Successfully deleted context: {actual_context_id}")
    # except exceptions.NotFound:
    #     print(f"Context {actual_context_id} not found during teardown.")
    # except Exception as e:
    #     print(f"Error deleting context {actual_context_id} during teardown: {e}")

def test_shopping_agent_vertex_eval(vertex_ai_context_manager, eval_config): # Inject fixtures
    if not os.getenv("GOOGLE_CLOUD_PROJECT") or not os.getenv("GOOGLE_CLOUD_LOCATION"):
        pytest.skip("GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION must be set in .env file to run Vertex AI evaluations.")

    experiment_run_name = vertex_ai_context_manager # Get the unique run name

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
