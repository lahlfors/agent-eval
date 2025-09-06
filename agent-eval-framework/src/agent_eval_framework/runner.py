"""Core logic for orchestrating and executing agent evaluations.

This module provides the main `run_evaluation` function that drives the entire
evaluation process based on a user-provided configuration file. It handles
everything from loading the configuration and data, to invoking the agent via
an adapter, to running the evaluation against the Vertex AI Evaluation Service
and printing the final results.
"""

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
    """Downloads a file from Google Cloud Storage to a temporary local path.

    Args:
        gcs_uri: The GCS URI of the file to download (e.g., "gs://bucket/file.jsonl").

    Returns:
        The local filesystem path to the downloaded temporary file.
    """
    client = storage.Client()
    bucket_name, blob_name = gcs_uri.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as temp_file:
        blob.download_to_filename(temp_file.name)
        return temp_file.name

def load_class(import_str: str) -> type:
    """Dynamically loads a class from a fully qualified string path.

    This allows the framework to instantiate classes (like agent adapters)
    that are defined in user-provided code, without needing to import them
    directly.

    Args:
        import_str: The fully qualified import path for the class
                    (e.g., "my_project.my_module.MyClass").

    Returns:
        The imported class object.
    """
    module_path, class_name = import_str.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def _build_metrics(metrics_config: list, golden_dataset: list) -> list:
    """Builds the list of metric objects for the Vertex AI EvaluationClient.

    This function parses the metric configuration provided by the user and
    constructs the appropriate metric objects required by the Vertex AI service.
    It supports standard computation metrics, rubric-based metrics, and custom
    Python functions.

    Args:
        metrics_config: A list of metric configurations from the config file.
        golden_dataset: The loaded golden dataset, used to validate that
                        trajectory-based metrics have the required data.

    Returns:
        A list of metric objects compatible with the Vertex AI EvaluationClient.

    Raises:
        ValueError: If an unknown metric type is specified or if a
                    trajectory-based metric is requested without the necessary
                    data columns in the dataset.
    """
    metrics = []
    for metric_spec in metrics_config:
        metric_type = metric_spec.get("type", "computation")
        metric_name = metric_spec["name"]

        if metric_type == "computation":
            metrics.append(evaluation_types.Metric(name=metric_name))

        elif metric_type == "rubric":
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
            custom_metric = evaluation_types.CustomMetric(
                name=metric_name,
                evaluation_function=custom_function
            )
            metrics.append(custom_metric)

        else:
            raise ValueError(f"Unknown metric type: {metric_type}")
    return metrics

def run_evaluation(config_path: str) -> model_evaluation_types.ModelEvaluation:
    """Runs the full, configuration-driven evaluation pipeline.

    This is the main entry point for the evaluation framework. It orchestrates
    the entire process:
    1. Loads environment variables and the main YAML configuration file.
    2. Initializes the GCP environment.
    3. Dynamically loads and instantiates the specified agent adapter.
    4. Loads the golden dataset (from a local path or GCS).
    5. Applies column mapping if specified.
    6. Iterates through the dataset, calling the agent for each example.
    7. Constructs the metric objects.
    8. Calls the Vertex AI Evaluation Service.
    9. Prints the results to the console.

    Args:
        config_path: The path to the main YAML configuration file.

    Returns:
        The ModelEvaluation object returned by the Vertex AI EvaluationClient,
        which contains the detailed results of the evaluation run.

    Raises:
        EnvironmentError: If required GCP environment variables are not set.
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
    print("\n--- Evaluation Results ---")
    try:
        if hasattr(evaluation_run, 'metrics_table') and evaluation_run.metrics_table is not None:
            print(evaluation_run.metrics_table)
        elif hasattr(evaluation_run, 'metrics') and evaluation_run.metrics is not None:
            for metric in evaluation_run.metrics:
                print(f"Metric: {metric.name}")
                if hasattr(metric, 'value'):
                    print(f"  Value: {metric.value}")
                if hasattr(metric, 'result'):
                    print(f"  Result: {metric.result}")
        else:
            print(vars(evaluation_run))
    except Exception as e:
        print(f"Could not parse and display results automatically. Printing raw object: {e}")
        print(evaluation_run)

    return evaluation_run
