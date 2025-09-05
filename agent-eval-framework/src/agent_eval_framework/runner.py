import os
import json
import yaml
import importlib
import dotenv
import tempfile
from google.cloud import aiplatform
from google.cloud import storage

def _download_gcs_file(gcs_uri: str) -> str:
    """Downloads a file from GCS to a temporary local path."""
    client = storage.Client()
    bucket_name, blob_name = gcs_uri.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as temp_file:
        blob.download_to_filename(temp_file.name)
        return temp_file.name

def load_class(import_str: str):
    """Dynamically loads a class from a string path."""
    module_path, class_name = import_str.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def run_evaluation(config_path: str):
    """
    Runs the full evaluation based on a configuration file.
    """
    dotenv.load_dotenv()

    # 1. Load Configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Setup GCP and AI Platform
    project_id = os.getenv("GCP_PROJECT_ID")
    location = os.getenv("GCP_REGION")
    if not project_id or not location:
        raise EnvironmentError("GCP_PROJECT_ID and GCP_REGION must be set.")
    aiplatform.init(project=project_id, location=location)

    # 3. Instantiate Agent Adapter
    agent_engine_id_from_env = os.getenv("AGENT_ENGINE_ID")
    if agent_engine_id_from_env:
        config["agent_config"]["agent_engine_id"] = agent_engine_id_from_env

    adapter_class = load_class(config["agent_adapter_class"])
    adapter = adapter_class(**config["agent_config"])

    # 4. Load and Prepare Golden Dataset
    dataset_path = config["dataset_path"]
    if dataset_path.startswith("gs://"):
        print(f"Downloading dataset from GCS: {dataset_path}")
        local_dataset_path = _download_gcs_file(dataset_path)
    else:
        local_dataset_path = dataset_path

    with open(local_dataset_path, "r") as f:
        golden_dataset = [json.loads(line) for line in f]

    if dataset_path.startswith("gs://"):
        os.remove(local_dataset_path)

    # 4a. Apply Column Mapping if provided
    if "column_mapping" in config:
        mapping = config["column_mapping"]
        reversed_mapping = {v: k for k, v in mapping.items()}

        new_dataset = []
        for record in golden_dataset:
            new_record = {}
            for internal_name, user_name in mapping.items():
                if user_name in record:
                    new_record[internal_name] = record[user_name]

            # Copy over any other columns that weren't mapped
            for key, value in record.items():
                if key not in mapping.values():
                    new_record[key] = value
            new_dataset.append(new_record)
        golden_dataset = new_dataset
        print(f"Applied column mapping: {mapping}")

    # 5. Generate Actual Responses and Trajectories
    print("Generating agent responses...")
    for record in golden_dataset:
        prompt = record.get("prompt")
        try:
            response_data = adapter.get_response(prompt)
            record["actual_response"] = response_data.get("actual_response", "")
            record["actual_trajectory"] = response_data.get("actual_trajectory", [])
        except Exception as e:
            print(f"ERROR: Adapter failed for prompt '{prompt}': {e}")
            record["actual_response"] = "AGENT_EXECUTION_ERROR"
            record["actual_trajectory"] = []

    # 6. Define and Run Evaluation Task
    print("Running evaluation...")

    # Process metrics from config
    metrics_list = []
    has_trajectory_metrics = False
    for metric in config["metrics"]:
        if isinstance(metric, str):
            metrics_list.append(metric)
            if "trajectory" in metric:
                has_trajectory_metrics = True
        elif isinstance(metric, dict) and "custom_function_path" in metric:
            custom_function = load_class(metric["custom_function_path"])
            metrics_list.append(
                aiplatform.evaluate.Metric(
                    name=metric["name"],
                    custom_function=custom_function
                )
            )
        else:
            print(f"Warning: Skipping invalid metric config: {metric}")

    eval_task_args = {
        "dataset": golden_dataset,
        "metrics": metrics_list,
        "response_column": "actual_response",
    }

    if "reference_response" in golden_dataset[0]:
        eval_task_args["reference_column"] = "reference_response"

    if has_trajectory_metrics:
        eval_task_args["trajectory_column"] = "actual_trajectory"
        if "reference_trajectory" in golden_dataset[0]:
            eval_task_args["reference_trajectory_column"] = "reference_trajectory"

    eval_task = aiplatform.evaluate.EvalTask(**eval_task_args)
    result = eval_task.evaluate()

    # 7. Print results
    print("\\n--- Evaluation Results ---")
    print(result.metrics_table)

    return result
