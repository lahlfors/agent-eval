# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Core logic for orchestrating and executing agent evaluations."""

import os
import json
import yaml
import importlib
import dotenv
import tempfile
import pandas as pd
import vertexai
from google.cloud import aiplatform
from google.cloud import storage
from typing import List, Dict, Any, Union, Type
import pathlib
import uuid
from datetime import datetime
import traceback # Import traceback
import math
from opentelemetry import trace

# CORRECTED IMPORTS:
# Import the PREVIEW EvalTask for agent evaluation
from vertexai.preview.evaluation import EvalTask
# Import the preview metrics module ONLY for TrajectorySingleToolUse
from vertexai.preview.evaluation import metrics as preview_metrics
# Keep these for other metric types if needed
from vertexai.evaluation import CustomMetric, PointwiseMetric, MetricPromptTemplateExamples

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

def _build_metrics(metrics_config: List[Union[str, Dict[str, Any]]]) -> List[Any]:
    """Builds the list of metric objects for the vertexai.evaluation.EvalTask."""
    metrics = []
    for metric_spec in metrics_config:
        if isinstance(metric_spec, str):
            metric_name = metric_spec
            log.debug(f"Building metric: {metric_name} (from string)")
            metrics.append(metric_name)
            log.info(f"Appended built-in metric: {metric_name}")

        elif isinstance(metric_spec, dict):
            metric_name = metric_spec["name"]
            metric_type = metric_spec.get("type")
            log.debug(f"Building metric: {metric_name}, type: {metric_type} (from dict)")

            if metric_name == "trajectory_single_tool_use":
                tool_name = metric_spec.get("tool_name")
                if not tool_name:
                    raise ValueError(f"'{metric_name}' requires a 'tool_name' field in the config.")
                try:
                    metrics.append(preview_metrics.TrajectorySingleToolUse(tool_name=tool_name))
                    log.info(f"Appended TrajectorySingleToolUse metric for tool: {tool_name}")
                except AttributeError:
                     log.error(f"'{metric_name}' class not found in preview_metrics. Check SDK version.")
                     raise
            elif metric_type == "pointwise":
                metric_prompt_template = metric_spec.get("metric_prompt_template")
                if not metric_prompt_template:
                     try:
                         metric_prompt_template = MetricPromptTemplateExamples.get_prompt_template(metric_name)
                     except ValueError:
                         raise ValueError(f"Pointwise metric '{metric_name}' needs a 'metric_prompt_template' or be a valid example name.")
                metrics.append(PointwiseMetric(
                    metric=metric_name,
                    metric_prompt_template=metric_prompt_template,
                ))
            elif metric_type == "custom_function":
                custom_function = load_class(metric_spec["custom_function_path"])
                metrics.append(CustomMetric(
                    name=metric_name,
                    metric_function=custom_function
                ))
            else:
                 raise ValueError(f"Unsupported metric configuration for: {metric_name} with type: {metric_type}")
        else:
            raise TypeError(f"Invalid metric specification type: {type(metric_spec)}")
    return metrics

def run_evaluation(config_path: str, experiment_run_name: str = None):
    """Runs the full, configuration-driven evaluation pipeline with Vertex AI Experiment tracking."""
    print(f"[DEBUG] run_evaluation called with experiment_run_name: {experiment_run_name}")
    eval_run_id = str(uuid.uuid4())
    set_log_context(eval_run_id=eval_run_id, user_id="agent-eval-framework")
    log.info("Starting evaluation run", extra={"config_path": config_path})

    eval_result = None  # Initialize eval_result
    try:
        project_root = pathlib.Path(__file__).resolve().parent.parent.parent
        dotenv_path = project_root / ".env"
        if dotenv_path.exists():
            log.debug(f"Loading environment variables from: {dotenv_path}")
            dotenv.load_dotenv(dotenv_path=dotenv_path, override=True)
        else:
            log.warning(f".env file not found at {dotenv_path}")

        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION")
        if not project_id or not location or project_id == "your-project-id-here":
            raise EnvironmentError("GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION must be set in the .env file at the project root.")

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
            # Adjust path to be relative to the project root
            possible_path = project_root / dataset_path
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
        target_col = config.get("target_column", "reference")

        if prompt_col not in df_dataset.columns:
             raise ValueError(f"'{prompt_col}' column not found in dataset after mapping.")
        if target_col not in df_dataset.columns:
             log.warning(f"'{target_col}' column not found in dataset after mapping. Some metrics may not work.")

        # NEW: Handle NaN values in columns used for metric API calls
        cols_to_clean = ["prompt", "reference", "response", "predicted_trajectory", "reference_trajectory"]
        for col in cols_to_clean:
            if col in df_dataset.columns:
                if df_dataset[col].isnull().any():
                    log.warning(f"NaN values found in column '{col}', replacing with empty string for API compatibility.")
                    df_dataset[col] = df_dataset[col].fillna('')

                # Ensure trajectory columns are JSON strings
                if col in ["predicted_trajectory", "reference_trajectory"]:
                    def sanitize_traj(traj):
                        if isinstance(traj, str):
                            try:
                                json.loads(traj) # Check if valid JSON
                                return traj
                            except json.JSONDecodeError:
                                return json.dumps({"tool_calls": []}) # Recover from bad string
                        elif isinstance(traj, dict) and "tool_calls" in traj:
                             return json.dumps(traj)
                        elif isinstance(traj, list):
                             return json.dumps({"tool_calls": traj})
                        return json.dumps({"tool_calls": []})
                    df_dataset[col] = df_dataset[col].apply(sanitize_traj)

        metrics = _build_metrics(config["metrics"])

        if not experiment_run_name:
            run_name_prefix = config.get("run_name_prefix", "eval")
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            run_name = f"{run_name_prefix}-{timestamp}-{unique_id}"
        else:
            run_name = experiment_run_name

        # Use the imported EvalTask from vertexai.preview.evaluation
        eval_task = EvalTask(
            dataset=df_dataset,
            metrics=metrics,
            experiment=experiment_name
        )

        log.info("Running evaluation using vertexai.preview.evaluation.EvalTask...")
        eval_result = eval_task.evaluate(
            runnable=adapter,
            experiment_run_name=run_name
        )
        log.info("Evaluation complete.")

        summary_metrics = eval_result.summary_metrics
        log.info(f"Summary Metrics: {summary_metrics}")

        print("\n--- Evaluation Results ---")
        print(f"Vertex AI Experiment: {experiment_name}, Run: {run_name}")
        # print("GCS Output Directory for this run:", eval_result.gcs_output_dir) # Removed
        print("\nSummary Metrics:")
        print(summary_metrics)

        print("\nMetrics Table:")
        display(eval_result.metrics_table)

        try:
            eval_payload = {
                "event_type": "evaluation_result",
                "eval_run_id": eval_run_id,
                "config_path": config_path,
                "experiment_name": experiment_name,
                "run_name": run_name,
                "summary_metrics": summary_metrics,
                "metrics_table": eval_result.metrics_table.to_dict(orient='records'),
                "dataset_path": config.get("dataset_path"),
                # "gcs_output_dir": eval_result.gcs_output_dir, # Removed
            }
            log.info("Evaluation results payload", extra={"payload": eval_payload})
        except Exception as e:
            log.error(f"Error logging eval results: {e}", exc_info=True)
    finally:
        tracer_provider = trace.get_tracer_provider()
        if hasattr(tracer_provider, "shutdown"):
            log.info("Flushing OpenTelemetry traces before exit...")
            tracer_provider.shutdown()
            log.info("Traces flushed.")
        else:
            log.warning("No OpenTelemetry TracerProvider with shutdown found.")

    return eval_result
