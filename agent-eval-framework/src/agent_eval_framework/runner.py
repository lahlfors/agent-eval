import os
import json
import yaml
import importlib
import dotenv
from google.cloud import aiplatform

def load_class(import_str: str):
    """Dynamically loads a class from a string path."""
    module_path, class_name = import_str.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def run_evaluation(config_path: str):
    """
    Runs the full evaluation based on a configuration file.
    """
    # Load environment variables from a .env file if it exists
    dotenv.load_dotenv()

    # 1. Load Configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Setup GCP and AI Platform
    project_id = os.getenv("GCP_PROJECT_ID")
    location = os.getenv("GCP_REGION")
    if not project_id or not location:
        raise EnvironmentError("GCP_PROJECT_ID and GCP_REGION environment variables must be set as environment variables.")
    aiplatform.init(project=project_id, location=location)

    # 3. Instantiate Agent Adapter
    agent_engine_id_from_env = os.getenv("AGENT_ENGINE_ID")
    if agent_engine_id_from_env:
        # Allow environment variable to override the config for security
        config["agent_config"]["agent_engine_id"] = agent_engine_id_from_env

    adapter_class = load_class(config["agent_adapter_class"])
    adapter = adapter_class(**config["agent_config"])

    # 4. Load Golden Dataset
    dataset_path = config["dataset_path"]
    with open(dataset_path, "r") as f:
        golden_dataset = [json.loads(line) for line in f]

    # 5. Generate Actual Responses
    print("Generating agent responses...")
    for record in golden_dataset:
        prompt = record["prompt"]
        try:
            response_data = adapter.get_response(prompt)
            record["actual_response"] = response_data["actual_response"]
        except Exception as e:
            print(f"ERROR: Adapter failed for prompt '{prompt}': {e}")
            # Add a placeholder response on failure to allow evaluation to continue
            record["actual_response"] = "AGENT_EXECUTION_ERROR"

    # 6. Define and Run Evaluation Task
    print("Running evaluation...")
    eval_task = aiplatform.evaluate.EvalTask(
        dataset=golden_dataset,
        metrics=config["metrics"],
        response_column="actual_response",
        reference_column="reference_response",
    )
    result = eval_task.evaluate()

    # 7. Print results
    print("\\n--- Evaluation Results ---")
    print(result.metrics_table)

    return result
