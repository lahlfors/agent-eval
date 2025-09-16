# Copyright 2024 Google LLC
# ... (license headers) ...

"""Core logic for orchestrating and executing agent evaluations."""

import os
import json
import yaml
import importlib
import dotenv
import tempfile
import pandas as pd
import vertexai
from vertexai import evaluation
from google.cloud import aiplatform # Import aiplatform
from google.cloud import storage
from typing import List, Dict, Any, Union
import pathlib
import uuid
from datetime import datetime
from .utils.logger import get_logger, set_log_context
from . import otel_config
from IPython.display import display

log = get_logger(__name__)

def load_class(import_str: str) -> type:
    """Dynamically loads a class from a fully qualified string path."""
    module_path, class_name = import_str.rsplit('.', 1)
    try:
        module = importlib.import_module(module_path)
        log.debug(f"Successfully loaded module: {module_path}")
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        log.error(f"Could not load class {import_str}", exc_info=True)
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
            log.info(f"Downloaded {gcs_uri} to {temp_file.name}")
            return temp_file.name
    except Exception as e:
        log.error(f"Failed to download {gcs_uri}", exc_info=True)
        raise RuntimeError(f"Failed to download {gcs_uri}: {e}")

def _build_metrics(metrics_config: List[Dict[str, Any]]) -> List[Union[str, evaluation.CustomMetric, evaluation.PointwiseMetric, evaluation.PairwiseMetric]]:
    """Builds the list of metric objects for the vertexai.evaluation.EvalTask."""
    metrics = []
    for metric_spec in metrics_config:
        metric_type = metric_spec.get("type", "computation")
        metric_name = metric_spec["name"]
        log.debug(f"Building metric: {metric_name}, type: {metric_type}")

        if metric_type == "computation":
            metrics.append(metric_name)
        elif metric_type == "pointwise":
            metric_prompt_template = metric_spec.get("metric_prompt_template")
            if not metric_prompt_template:
                 try:
                     metric_prompt_template = evaluation.MetricPromptTemplateExamples.get_prompt_template(metric_name)
                 except ValueError:
                     raise ValueError(f"Pointwise metric '{metric_name}' needs a 'metric_prompt_template' or be a valid example name.")
            metrics.append(evaluation.PointwiseMetric(
                metric=metric_name,
                metric_prompt_template=metric_prompt_template,
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
    """Runs the full, configuration-driven evaluation pipeline with Vertex AI Experiment tracking."""
    # --- NEW: Setup OpenTelemetry early ---
    otel_config.setup_opentelemetry()
    # ---
    eval_run_id = str(uuid.uuid4())
    set_log_context(eval_run_id=eval_run_id, user_id="agent-eval-framework")
    log.info("Starting evaluation run", extra={"config_path": config_path})

    project_root = pathlib.Path(__file__).resolve().parent.parent.parent.parent
    dotenv_path = project_root / ".env"
    if dotenv_path.exists():
        log.debug(f"Loading environment variables from: {dotenv_path}")
        dotenv.load_dotenv(dotenv_path=dotenv_path, override=True)
    else:
        log.warning(f".env file not found at {dotenv_path}")

    project_id = os.getenv("GCP_PROJECT_ID")
    location = os.getenv("GCP_REGION")
    if not project_id or not location or project_id == "your-project-id-here":
        raise EnvironmentError("GCP_PROJECT_ID and GCP_REGION must be set in the .env file at the project root.")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    log.debug("Configuration loaded", extra={"config": config})

    # --- NEW: Initialize Vertex AI SDK and AI Platform for Experiments ---
    experiment_name = config.get("experiment_name", "default-agent-evals")
    vertexai.init(project=project_id, location=location)
    aiplatform.init(project=project_id, location=location, experiment=experiment_name)
    log.info(f"Vertex AI initialized for project: {project_id}, location: {location}, experiment: {experiment_name}")
    # --- END NEW ---

    adapter_class = load_class(config["agent_adapter_class"])
    adapter = adapter_class(**config.get("agent_config", {}))

    dataset_path = config["dataset_path"]
    local_dataset_path = None
    if dataset_path.startswith("gs://"):
        local_dataset_path = _download_gcs_file(dataset_path)
    else:
        possible_path = project_root / "agent-eval-framework" / dataset_path
        if os.path.exists(possible_path):
             local_dataset_path = possible_path
        elif os.path.exists(dataset_path):
             local_dataset_path = dataset_path
        else:
             log.error(f"Dataset file not found: {possible_path} or {dataset_path}")
             raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with open(local_dataset_path, "r") as f:
        golden_dataset = [json.loads(line) for line in f]
    if str(dataset_path).startswith("gs://") and local_dataset_path:
         os.remove(local_dataset_path)
    df_dataset = pd.DataFrame(golden_dataset)
    log.info(f"Loaded dataset with {len(df_dataset)} records.")

    column_mapping = config.get("column_mapping", {})
    df_dataset.rename(columns=column_mapping, inplace=True)

    prompt_col = config.get("prompt_column", "prompt")
    target_col = config.get("target_column", "reference_response")
    df_dataset.rename(columns={
        prompt_col: "prompt",
        target_col: "reference",
    }, inplace=True, errors='ignore')

    if "prompt" not in df_dataset.columns:
        raise ValueError(f"'{prompt_col}' (mapped to 'prompt') column not found in dataset.")
    if "reference" not in df_dataset.columns:
        log.warning(f"'{target_col}' (mapped to 'reference') column not found. Some metrics may not work.")

    log.info("Generating agent responses...")
    actual_responses = []
    # ... (response generation loop as before) ...
    for index, record in df_dataset.iterrows():
        prompt = record.get("prompt")
        if not prompt:
            log.warning(f"No prompt found in record {index}")
            actual_responses.append("NO_PROMPT")
            continue
        try:
            response_data = adapter.get_response(prompt)
            actual_responses.append(response_data.get("actual_response", "NO_RESPONSE"))
        except Exception as e:
            log.error(f"Adapter failed for prompt '{prompt}'", exc_info=True)
            actual_responses.append("AGENT_EXECUTION_ERROR")
    df_dataset["response"] = actual_responses
    log.info("Finished generating agent responses.")

    metrics = _build_metrics(config["metrics"])

    # --- NEW: Start an Experiment Run ---
    run_name_prefix = config.get("run_name_prefix", "eval")
    run_name = f"{run_name_prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{eval_run_id[:4]}"

    with aiplatform.start_run(run=run_name) as my_run:
        my_run.display_name = run_name
        log.info(f"--- Starting Experiment Run: {my_run.name} ---")

        # Log parameters from the config
        my_run.log_params(config.get("agent_config", {}))
        my_run.log_params({
            "dataset": config["dataset_path"],
            "metrics_config": json.dumps(config["metrics"]),
            "eval_run_id": eval_run_id
        })
        log.info("Logged run parameters.")

        eval_task = evaluation.EvalTask(
            dataset=df_dataset,
            metrics=metrics,
            experiment=experiment_name # Associate with the current experiment
        )

        log.info("Running evaluation using vertexai.evaluation.EvalTask...")
        eval_result = eval_task.evaluate(
            experiment_run_name=my_run.name # Link EvalTask to this run
        )
        log.info("Evaluation complete.")

        # Log summary metrics
        summary_metrics = eval_result.summary_metrics
        my_run.log_metrics(summary_metrics)
        log.info(f"Logged Summary Metrics: {summary_metrics}")

        print("\n--- Evaluation Results ---")
        print(f"Vertex AI Experiment: {experiment_name}, Run: {my_run.name}")
        print("GCS Output Directory for this run:", eval_result.gcs_output_dir)
        print("\nSummary Metrics:")
        print(summary_metrics)
        # --- END NEW ---

        print("\nMetrics Table:")
        display(eval_result.metrics_table)

        try:
            eval_payload = {
                "event_type": "evaluation_result",
                "eval_run_id": eval_run_id,
                "config_path": config_path,
                "experiment_name": experiment_name,
                "run_name": my_run.name,
                "summary_metrics": summary_metrics,
                "metrics_table": eval_result.metrics_table.to_dict(orient='records'),
                "dataset_path": config.get("dataset_path"),
                "gcs_output_dir": eval_result.gcs_output_dir,
            }
            log.info("Evaluation results payload", extra={"payload": eval_payload})
        except Exception as e:
            log.error(f"Error logging eval results: {e}", exc_info=True)

    return eval_result
