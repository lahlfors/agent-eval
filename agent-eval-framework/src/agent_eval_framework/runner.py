# In agent-eval-framework/src/agent_eval_framework/runner.py

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
import pathlib
from IPython.display import display

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
    """Builds the list of metric objects for the evaluation.EvalTask."""
    metrics = []
    for metric_spec in metrics_config:
        metric_type = metric_spec.get("type", "computation")
        metric_name = metric_spec["name"]

        if metric_type == "computation":
            metrics.append(metric_name)
        elif metric_type == "pointwise":
            metric_prompt_template = metric_spec.get("metric_prompt_template")
            if not metric_prompt_template and metric_spec.get("use_example_template", False):
                 try:
                     metric_prompt_template = evaluation.MetricPromptTemplateExamples.get_prompt_template(metric_name)
                 except ValueError:
                     print(f"Warning: Pointwise metric '{metric_name}' not found in examples.")

            if not metric_prompt_template:
                 raise ValueError(f"Pointwise metric '{metric_name}' needs a 'metric_prompt_template'.")

            metrics.append(evaluation.PointwiseMetric(
                metric=metric_name,
                metric_prompt_template=metric_prompt_template,
                rating_rubric=metric_spec.get("rating_rubric", {}),
            ))
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
    # .env loading: Assume it's in the CWD (project root)
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
    print(f"Vertex AI SDK initialized for project: {project_id}, location: {location}")

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
        # Path is relative to the project root (CWD)
        local_dataset_path = dataset_path

    print(f"Attempting to load dataset from: {os.path.abspath(local_dataset_path)}")
    if not os.path.exists(local_dataset_path):
        raise FileNotFoundError(f"[Errno 2] No such file or directory: '{os.path.abspath(local_dataset_path)}'")

    with open(local_dataset_path, "r") as f:
        golden_dataset = [json.loads(line) for line in f]

    if config["dataset_path"].startswith("gs://") and os.path.exists(local_dataset_path):
        os.remove(local_dataset_path)

    df_dataset = pd.DataFrame(golden_dataset)

    # 4a. Apply Column Mapping
    column_mapping = config.get("column_mapping", {})
    df_dataset.rename(columns=column_mapping, inplace=True)
    print(f"Applied column mapping: {column_mapping}")
    print(f"Dataset columns after mapping: {df_dataset.columns.tolist()}")

    # 5. Generate Actual Responses
    print("Generating agent responses...")
    actual_responses = []
    for index, record in df_dataset.iterrows():
        prompt = record.get("prompt")
        if not prompt:
            print(f"Warning: Missing 'prompt' in record {index}")
            actual_responses.append("MISSING_PROMPT")
            continue
        try:
            response_data = adapter.get_response(prompt)
            actual_responses.append(response_data.get("actual_response", "NO_RESPONSE"))
        except Exception as e:
            print(f"ERROR: Adapter failed for prompt '{prompt}': {e}")
            actual_responses.append("AGENT_EXECUTION_ERROR")

    df_dataset["response"] = actual_responses

    # 6. Define and Run Evaluation using EvalTask
    print("Running evaluation using vertexai.evaluation.EvalTask...")
    metrics = _build_metrics(config["metrics"])

    if not metrics:
        print("Warning: No metrics configured for evaluation.")
        return None

    eval_task = evaluation.EvalTask(
        dataset=df_dataset,
        metrics=metrics,
        experiment=config.get("experiment_name", "agent-eval-framework-run")
    )

    eval_result = eval_task.evaluate(
        experiment_run_name=config.get("experiment_run_name", "run-" + pd.Timestamp.now().strftime("%Y%m%d%H%M%S"))
    )

    # 7. Print results
    print("\n--- Evaluation Results ---")
    print("Summary Metrics:")
    print(eval_result.summary_metrics)
    print("\nMetrics Table:")
    display(eval_result.metrics_table)

    return eval_result
