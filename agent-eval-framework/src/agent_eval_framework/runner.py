import os
import json
import yaml
import importlib
import dotenv
import tempfile
from google.cloud import aiplatform
from google.cloud import storage
from vertexai.preview.language_models import EvaluationClient
from google.cloud.aiplatform_v1beta1.types import (
    evaluation as evaluation_types,
    model_evaluation as model_evaluation_types,
)

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

def _build_metrics(metrics_config: list, golden_dataset: list):
    """Builds the list of metric objects for the EvaluationClient."""
    metrics = []
    for metric_spec in metrics_config:
        metric_type = metric_spec.get("type", "computation") # Default to computation
        metric_name = metric_spec["name"]

        if metric_type == "computation":
            metrics.append(evaluation_types.Metric(name=metric_name))

        elif metric_type == "rubric":
            # Check for trajectory-based rubric
            if "trajectory" in metric_name and "actual_trajectory" not in golden_dataset[0]:
                 raise ValueError(f"Metric '{metric_name}' requires 'actual_trajectory' column in dataset.")

            rubric = evaluation_types.RubricMetric(
                name=metric_name,
                predefined_spec_name=metric_spec.get("predefined_spec_name"),
                metric_spec_parameters=metric_spec.get("metric_spec_parameters"),
                version=metric_spec.get("version"),
            )
            metrics.append(rubric)

        elif metric_type == "custom_function":
            custom_function = load_class(metric_spec["custom_function_path"])
            # The EvaluationClient expects a specific structure for custom functions
            custom_metric = evaluation_types.CustomMetric(
                name=metric_name,
                evaluation_function=custom_function
            )
            metrics.append(custom_metric)

        else:
            raise ValueError(f"Unknown metric type: {metric_type}")
    return metrics


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

    # 6. Define and Run Evaluation
    print("Running evaluation...")
    eval_client = EvaluationClient(project_id=project_id, location=location)

    metrics = _build_metrics(config["metrics"], golden_dataset)

    evaluation_run = eval_client.evaluate(
        dataset=golden_dataset,
        metrics=metrics,
        response_column="actual_response",
        reference_column="reference_response" if "reference_response" in golden_dataset[0] else None,
        trajectory_column="actual_trajectory" if "actual_trajectory" in golden_dataset[0] else None,
        reference_trajectory_column="reference_trajectory" if "reference_trajectory" in golden_dataset[0] else None,
    )

    # 7. Print results
    print("\\n--- Evaluation Results ---")

    # The result object from the new client is different.
    # We attempt to print the results in a structured way.
    try:
        # The new client may return a result object with a .metrics_table attribute
        if hasattr(evaluation_run, 'metrics_table') and evaluation_run.metrics_table is not None:
            print(evaluation_run.metrics_table)
        # Or it may have a 'metrics' attribute that is a list of metric objects
        elif hasattr(evaluation_run, 'metrics') and evaluation_run.metrics is not None:
            for metric in evaluation_run.metrics:
                print(f"Metric: {metric.name}")
                # Assuming the metric object has a 'value' or similar attribute
                if hasattr(metric, 'value'):
                    print(f"  Value: {metric.value}")
                if hasattr(metric, 'result'):
                    print(f"  Result: {metric.result}")
        # As a fallback, print the dictionary representation of the object
        else:
            print(vars(evaluation_run))

    except Exception as e:
        print(f"Could not parse and display results automatically. Printing raw object.")
        print(evaluation_run)


    return evaluation_run
