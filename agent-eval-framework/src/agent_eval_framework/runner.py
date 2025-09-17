# agent-eval-framework/src/agent_eval_framework/runner.py
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
import traceback
import math
import sys

from opentelemetry import trace
from vertexai.preview.evaluation import EvalTask
from vertexai.preview.evaluation import metrics as preview_metrics
from vertexai.evaluation import CustomMetric, PointwiseMetric, MetricPromptTemplateExamples

from .utils.logger import get_logger, set_log_context
from . import otel_config
from IPython.display import display

log = get_logger(__name__)

def load_class(import_str: str) -> type:
    module_path, class_name = import_str.rsplit('.', 1)
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not load class {import_str}: {e}")

def _download_gcs_file(gcs_uri: str) -> str:
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

def _build_metrics(metrics_config: List[Union[str, Dict[str, Any]]]) -> List[Any]:
    metrics = []
    for metric_spec in metrics_config:
        if isinstance(metric_spec, str):
            metrics.append(metric_spec)
        elif isinstance(metric_spec, dict):
            metric_name = metric_spec["name"]
            metric_type = metric_spec.get("type")
            if metric_name == "trajectory_single_tool_use":
                 metrics.append(preview_metrics.TrajectorySingleToolUse(tool_name=metric_spec["tool_name"]))
            elif metric_type == "pointwise":
                 metric_prompt_template = metric_spec.get("metric_prompt_template")
                 if not metric_prompt_template:
                      metric_prompt_template = MetricPromptTemplateExamples.get_prompt_template(metric_name)
                 metrics.append(PointwiseMetric(metric=metric_name, metric_prompt_template=metric_prompt_template))
            elif metric_type == "custom_function":
                 metrics.append(CustomMetric(name=metric_name, metric_function=load_class(metric_spec["custom_function_path"])))
            else:
                 raise ValueError(f"Unsupported metric: {metric_name} with type: {metric_type}")
        else:
            raise TypeError(f"Invalid metric specification type: {type(metric_spec)}")
    return metrics

def run_evaluation(config_path: str, experiment_run_name: str = None):
    sys.stdout.write(f"[DEBUG] run_evaluation called with experiment_run_name: {experiment_run_name}\n")
    sys.stdout.flush()
    # otel_config.setup_opentelemetry() # Called from conftest.py

    eval_run_id = str(uuid.uuid4())
    set_log_context(eval_run_id=eval_run_id, user_id="agent-eval-framework")

    eval_result = None
    try:
        project_root = pathlib.Path(__file__).resolve().parent.parent.parent
        dotenv_path = project_root / ".env"
        if dotenv_path.exists():
            dotenv.load_dotenv(dotenv_path=dotenv_path, override=True)

        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION")
        if not project_id or not location:
            raise EnvironmentError("GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION must be set.")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        experiment_name = config.get("experiment_name", "default-agent-evals")
        vertexai.init(project=project_id, location=location)
        aiplatform.init(project=project_id, location=location, experiment=experiment_name)
        sys.stdout.write(f"Vertex AI initialized for project: {project_id}, location: {location}, experiment: {experiment_name}\n")
        sys.stdout.flush()

        adapter_class = load_class(config["agent_adapter_class"])
        adapter = adapter_class(**config.get("agent_config", {}))

        dataset_path = config["dataset_path"]
        local_dataset_path = None
        if dataset_path.startswith("gs://"):
            local_dataset_path = _download_gcs_file(dataset_path)
        else:
            possible_path = project_root / dataset_path
            local_dataset_path = str(possible_path) if possible_path.exists() else dataset_path
            if not os.path.exists(local_dataset_path):
                 raise FileNotFoundError(f"Dataset file not found: {local_dataset_path}")

        with open(local_dataset_path, "r") as f:
            golden_dataset = [json.loads(line) for line in f]
        if str(dataset_path).startswith("gs://"): os.remove(local_dataset_path)
        df_dataset = pd.DataFrame(golden_dataset)

        column_mapping = config.get("column_mapping", {})
        df_dataset.rename(columns=column_mapping, inplace=True)

        cols_to_clean = ["prompt", "reference", "response", "predicted_trajectory", "reference_trajectory"]
        for col in cols_to_clean:
            if col in df_dataset.columns:
                if df_dataset[col].isnull().any():
                    df_dataset[col] = df_dataset[col].fillna('')

        def extract_tool_calls(traj_input):
            try:
                if isinstance(traj_input, str): data = json.loads(traj_input)
                elif isinstance(traj_input, dict): data = traj_input
                else: return []
                return data.get("tool_calls", [])
            except: return []

        if "reference_trajectory" in df_dataset.columns:
            df_dataset["reference_trajectory"] = df_dataset["reference_trajectory"].apply(extract_tool_calls)
        else:
             df_dataset["reference_trajectory"] = [[] for _ in range(len(df_dataset))]

        metrics = _build_metrics(config["metrics"])
        run_name = experiment_run_name or f"{config.get('run_name_prefix', 'eval')}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4()}"

        sys.stdout.write("runner.py: Loaded adapter and dataset\n")
        sys.stdout.flush()
        otel_config.log_otel_status("run_evaluation before evaluate")

        eval_task = EvalTask(
             dataset=df_dataset,
             metrics=metrics,
             experiment=experiment_name
        )
        sys.stdout.write("runner.py: EvalTask created\n")
        sys.stdout.flush()

        eval_result = eval_task.evaluate(runnable=adapter, experiment_run_name=run_name)
        sys.stdout.write("runner.py: eval_task.evaluate finished\n")
        sys.stdout.flush()
        otel_config.log_otel_status("run_evaluation after evaluate")

        summary_metrics = eval_result.summary_metrics
        print("\n--- Evaluation Results ---")
        print(f"Vertex AI Experiment: {experiment_name}, Run: {run_name}")
        print("\nSummary Metrics:")
        print(summary_metrics)
        print("\nMetrics Table:")
        display(eval_result.metrics_table)

    except Exception as e:
        sys.stderr.write(f"Error in run_evaluation: {e}\n")
        sys.stderr.flush()
        traceback.print_exc()
        # raise # Re-raising will stop the finally block from hiding the original error
    finally:
        sys.stdout.write("runner.py: Entering finally block\n")
        sys.stdout.flush()
        otel_config.log_otel_status("run_evaluation finally")
        tracer_provider = trace.get_tracer_provider()
        if hasattr(tracer_provider, "shutdown"):
            sys.stdout.write("runner.py: Flushing OpenTelemetry traces before exit...\n")
            sys.stdout.flush()
            tracer_provider.shutdown() # Removed timeout_millis
            sys.stdout.write("runner.py: Traces flushed.\n")
            sys.stdout.flush()
        else:
            sys.stdout.write("runner.py: No OpenTelemetry TracerProvider with shutdown found.\n")
            sys.stdout.flush()
    return eval_result
