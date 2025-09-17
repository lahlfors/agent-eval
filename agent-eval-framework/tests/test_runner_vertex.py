import pytest
import os
from google.cloud import aiplatform
from google.api_core import exceptions
from agent_eval_framework.runner import run_evaluation
import uuid
from datetime import datetime

# Optional: Load .env for local runs if not handled by shell
from dotenv import load_dotenv
load_dotenv()

CONFIG_PATH = "agent-eval-framework/config/adk_eval_config.yaml"

@pytest.fixture(scope="function")
def unique_experiment_run_name():
    # Generate a unique name for each test function invocation
    experiment_name = "personalized-shopping-adk-eval"  # Or load from config
    run_name_prefix = "eval-run-final-20"  # Or load from config
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"{experiment_name}-{run_name_prefix}-{timestamp}-{unique_id}"

@pytest.fixture(scope="function")
def vertex_ai_context_manager(unique_experiment_run_name):
    project_id = os.getenv("GCP_PROJECT_ID")
    location = os.getenv("GCP_REGION")
    context_id = unique_experiment_run_name  # Use the unique name for the context ID

    aiplatform.init(project=project_id, location=location)

    # Teardown logic to delete the context after the test
    yield context_id # Test runs here

    print(f"Tearing down: Attempting to delete context {context_id}")
    try:
        # The context seems to be created *inside* run_evaluation -> vertexai.preview.start_run
        # We need to fetch it by the ID that was used.
        context_to_delete = aiplatform.Context.get(resource_id=context_id)
        context_to_delete.delete()
        print(f"Successfully deleted context: {context_id}")
    except exceptions.NotFound:
        print(f"Context {context_id} not found during teardown.")
    except Exception as e:
        print(f"Error deleting context {context_id} during teardown: {e}")

def test_shopping_agent_vertex_eval(vertex_ai_context_manager): # Inject the fixture
    """
    Triggers the agent evaluation using the agent-eval-framework,
    which uses the Vertex AI GenAI Evaluation Service.
    """
    if not os.getenv("GCP_PROJECT_ID") or not os.getenv("GCP_REGION"):
        pytest.skip("GCP_PROJECT_ID and GCP_REGION must be set in .env file to run Vertex AI evaluations.")

    context_id = vertex_ai_context_manager # Get the unique context ID

    print(f"Running evaluation with config: {CONFIG_PATH}")
    print(f"Using unique context ID: {context_id}")

    try:
        # IMPORTANT: You need to modify run_evaluation to accept and use
        # the generated context_id as the experiment_run_name.
        # Example: eval_result = run_evaluation(config_path=CONFIG_PATH, experiment_run_name=context_id)

        # Placeholder for the actual call - adapt as needed!
        # Assuming run_evaluation uses the passed name for the Vertex AI run
        eval_result = run_evaluation(config_path=CONFIG_PATH, experiment_run_name=context_id)

        assert eval_result is not None, "Evaluation failed to produce results."
        print("Evaluation completed successfully using agent-eval-framework.")
    except Exception as e:
        pytest.fail(f"run_evaluation failed: {e}")
