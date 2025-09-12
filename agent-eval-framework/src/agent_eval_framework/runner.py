# agent-eval-framework/src/agent_eval_framework/runner.py

"""Core logic for orchestrating and executing agent evaluations using the GenAI Client.

This module provides the main run_evaluation function that drives the entire
evaluation process based on a user-provided configuration file. It handles
everything from loading the configuration and data, to invoking the agent
via an adapter, to running the evaluation against the Vertex AI Evaluation
Service and printing the final results.
"""

import os
import json
import yaml
import importlib
import dotenv
import tempfile
import pandas as pd
import vertexai
from vertexai import types as vertex_types  # Using vertex_types alias
from google.cloud import storage
from typing import List, Dict, Any, Union
import pathlib

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

def _build_metrics(metrics_config: List[Dict[str, Any]]) -> List[Any]:
    """Builds the list of metric objects for the vertexai.Client.evals.evaluate.

    Args:
        metrics_config: A list of metric configurations from the config file.

    Returns:
        A list of metric objects compatible with client.evals.evaluate.
    """
    metrics = []
    for metric_spec in metrics_config:
        metric_type = metric_spec.get("type", "computation")
        metric_name = metric_spec["name"]

        if metric_type == "computation":
            # Built-in computation metrics
            metrics.append(vertex_types.Metric(name=metric_name))
        elif metric_type == "pointwise":
            # LLM-based rubric metrics
            try:
                # Try to get a predefined rubric metric
                rubric_metric = getattr(vertex_types.RubricMetric, metric_name.upper())
                metrics.append(rubric_metric)
            except AttributeError:
                # If not predefined, treat as custom LLM metric
                metric_prompt_template = metric_spec.get("metric_prompt_template")
                if not metric_prompt_template:
                    raise ValueError(f"Pointwise metric '{metric_name}' needs 'metric_prompt_template' or be a predefined RubricMetric.")

                # Build prompt template if needed
                if isinstance(metric_prompt_template, dict):
                    prompt_builder = vertex_types.MetricPromptBuilder(
                        instruction=metric_prompt_template.get("instruction", ""),
                        criteria=metric_prompt_template.get("criteria", {}),
                        rating_scores=metric_prompt_template.get("rating_scores", {})
                    )
                    metric_prompt_template = prompt_builder

                metrics.append(vertex_types.LLMMetric(
                    name=metric_name,
                    prompt_template=metric_prompt_template
                ))
        elif metric_type == "custom_function":
            custom_function = load_class(metric_spec["custom_function_path"])
            metrics.append(vertex_types.Metric(
                name=metric_name,
                custom_function=custom_function
            ))
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")
    return metrics

def run_evaluation(config_path: str):
    """Runs the full, configuration-driven evaluation pipeline."""
    # Construct path to the .env file in the project root
    project_root = pathlib.Path(__file__).parent.parent.parent.parent
    dotenv_path = project_root / ".env"

    if dotenv_path.exists():
        print(f"Loading environment variables from: {dotenv_path}")
        dotenv.load_dotenv(dotenv_path=dotenv_path)
    else:
        print(f"Warning: .env file not found at {dotenv_path}")

    # 1. Load Configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Setup GCP and AI Platform
    project_id = os.getenv("GCP_PROJECT_ID")
    location = os.getenv("GCP_REGION")
    if not project_id or not location:
        raise EnvironmentError("GCP_PROJECT_ID and GCP_REGION must be set.")

    client = vertexai.Client(project=project_id, location=location)
    print(f"Vertex AI Client initialized for project: {project_id}, location: {location}")

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

    if dataset_path.startswith("gs://") and os.path.exists(local_dataset_path):
        os.remove(local_dataset_path)

    df_dataset = pd.DataFrame(golden_dataset)

    # 4a. Apply Column Mapping if provided
    column_mapping = config.get("column_mapping", {})
    df_dataset.rename(columns=column_mapping, inplace=True)
    print(f"Applied column mapping: {column_mapping}")

    # 5. Generate Actual Responses and Trajectories
    print("Generating agent responses...")
    actual_responses = []
    for index, record in df_dataset.iterrows():
        prompt = record.get("prompt")
        try:
            response_data = adapter.get_response(prompt)
            actual_responses.append(response_data.get("actual_response", "NO_RESPONSE"))
        except Exception as e:
            print(f"ERROR: Adapter failed for prompt '{prompt}': {e}")
            actual_responses.append("AGENT_EXECUTION_ERROR")

    df_dataset["response"] = actual_responses # Default column name for client.evals.evaluate

    # 6. Define and Run Evaluation using GenAI Client
    print("Running evaluation using vertexai.Client.evals...")
    metrics = _build_metrics(config["metrics"])

    if not metrics:
        print("Warning: No metrics configured for evaluation.")
        return None

    # Determine column names
    reference_column = config.get("column_mapping", {}).get("reference", "reference")
    if reference_column not in df_dataset.columns:
        print(f"Warning: Reference column '{reference_column}' not found in dataset.")
        reference_column = None

    eval_result = client.evals.evaluate(
        dataset=df_dataset,
        metrics=metrics,
        reference_column=reference_column
    )

    # 7. Print results
    print("\n--- Evaluation Results ---")
    try:
        from IPython.display import display
        print("Displaying results in notebook format...")
        eval_result.show()
    except ImportError:
        print("IPython not available, printing tables to console.")
        print("Summary Metrics:")
        print(eval_result.summary_metrics)
        print("\nMetrics Table:")
        print(eval_result.metrics_table.to_string())

    return eval_result
