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
import pandas as pd
import vertexai
from vertexai import evaluation
from google.cloud import storage
from typing import List, Dict, Any, Union

# Properly load the class
def load_class(import_str: str) -> type:
    """Dynamically loads a class from a fully qualified string path."""
    module_path, class_name = import_str.rsplit('.', 1)
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not load class {import_str}: {e}")

def _download_gcs_file(gcs_uri: str) -> str:
    """Downloads a file from Google Cloud Storage to a temporary local path."""
    try:
        client = storage.Client()
        bucket_name, blob_name = gcs_uri.replace("gs://", "").split("/", 1)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as temp_file:
            blob.download_to_filename(temp_file.name)
            return temp_file.name
    except Exception as e:
        raise RuntimeError(f"Failed to download {gcs_uri}: {e}")

def _build_metrics(metrics_config: List[Dict[str, Any]]) -> List[Union[str, evaluation.CustomMetric, evaluation.PointwiseMetric, evaluation.PairwiseMetric]]:
    """Builds the list of metric objects for the vertexai.evaluation.EvalTask.

    Args:
        metrics_config: A list of metric configurations from the config file.

    Returns:
        A list of metric objects compatible with EvalTask.
    """
    metrics = []
    for metric_spec in metrics_config:
        metric_type = metric_spec.get("type", "computation")
        metric_name = metric_spec["name"]

        if metric_type == "computation":
            # Built-in computation metrics are passed as strings
            metrics.append(metric_name)
        elif metric_type == "pointwise":
            metric_prompt_template = metric_spec.get("metric_prompt_template")
            if not metric_prompt_template:
                 # Try to load from examples if template not provided
                 try:
                     metric_prompt_template = evaluation.MetricPromptTemplateExamples.get_prompt_template(metric_name)
                 except ValueError:
                     raise ValueError(f"Pointwise metric '{metric_name}' needs a 'metric_prompt_template' or be a valid example name.")

            metrics.append(evaluation.PointwiseMetric(
                metric=metric_name,
                metric_prompt_template=metric_prompt_template,
                rating_rubric=metric_spec.get("rating_rubric", {}),
                input_variables=metric_spec.get("input_variables"),
            ))
        elif metric_type == "pairwise":
             # PairwiseMetric requires more complex setup, potentially including a baseline model
             # This is a placeholder, actual implementation depends on config structure
             raise NotImplementedError("Pairwise metric configuration not fully implemented in this refactor.")
        elif metric_type == "custom_function":
            custom_function = load_class(metric_spec["custom_function_path"])
            metrics.append(evaluation.CustomMetric(
                name=metric_name,
                metric_function=custom_function
            ))
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")
    return metrics

def run_evaluation(config_path: str):
    """Runs the full, configuration-driven evaluation pipeline."""
    dotenv.load_dotenv()

    # 1. Load Configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Setup GCP and AI Platform
    project_id = os.getenv("GCP_PROJECT_ID")
    location = os.getenv("GCP_REGION")
    if not project_id or not location:
        raise EnvironmentError("GCP_PROJECT_ID and GCP_REGION must be set.")
    vertexai.init(project=project_id, location=location)
    print(f"Vertex AI initialized for project: {project_id}, location: {location}")

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

    df_dataset = pd.DataFrame(golden_dataset)

    # 4a. Apply Column Mapping if provided
    column_mapping = config.get("column_mapping", {})
    df_dataset.rename(columns=column_mapping, inplace=True)
    print(f"Applied column mapping: {column_mapping}")

    # 5. Generate Actual Responses and Trajectories
    print("Generating agent responses...")
    actual_responses = []
    # actual_trajectories = [] # If trajectories are produced

    for index, record in df_dataset.iterrows():
        prompt = record.get("prompt")
        try:
            # Assuming adapter.get_response returns a dict like {"actual_response": "...", "actual_trajectory": ...}
            response_data = adapter.get_response(prompt)
            actual_responses.append(response_data.get("actual_response", "NO_RESPONSE"))
            # if "actual_trajectory" in response_data:
            #     actual_trajectories.append(response_data["actual_trajectory"])
        except Exception as e:
            print(f"ERROR: Adapter failed for prompt '{prompt}': {e}")
            actual_responses.append("AGENT_EXECUTION_ERROR")
            # actual_trajectories.append([])

    df_dataset["actual_response"] = actual_responses
    # if actual_trajectories:
    #     df_dataset["actual_trajectory"] = actual_trajectories

    # 6. Define and Run Evaluation using EvalTask
    print("Running evaluation using vertexai.evaluation.EvalTask...")
    metrics = _build_metrics(config["metrics"])
    eval_task = evaluation.EvalTask(
        dataset=df_dataset,
        metrics=metrics,
        experiment=config.get("experiment_name", "agent-eval-framework-run")
    )

    # Column names for EvalTask.evaluate()
    response_column_name = "actual_response"
    reference_column_name = config.get("column_mapping", {}).get("reference_response", "reference_response")
    if reference_column_name not in df_dataset.columns:
        reference_column_name = None
        print(f"Warning: '{reference_column_name}' not found, not passing reference column.")

    # Add other column mappings as needed

    eval_result = eval_task.evaluate(
        response_column_name=response_column_name,
        # baseline_model_response_column_name= ...
        # experiment_run_name= ...
    )

    # 7. Print results
    print("\n--- Evaluation Results ---")
    print("Summary Metrics:")
    print(eval_result.summary_metrics)

    print("\nMetrics Table:")
    from IPython.display import display
    display(eval_result.metrics_table)

    return eval_result
