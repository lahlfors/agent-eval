# src/agent_eval_framework/runner.py
import asyncio
from datetime import datetime
import json
import os
import pathlib
import uuid
import yaml
import pandas as pd
from google.cloud import aiplatform
import vertexai
from vertexai.preview import evaluation
import dotenv
from IPython.display import display

from agent_eval_framework.utils.class_loader import load_class
from agent_eval_framework.utils.metric_utils import _build_metrics
from agent_eval_framework.utils import otel_config
from agent_eval_framework.utils.gcs_utils import download_gcs_file
from agent_eval_framework.utils.logging_utils import log, set_log_context

def run_evaluation(config_path: str):
    """Runs the full, configuration-driven evaluation pipeline with Vertex AI Experiment tracking."""
    otel_config.setup_opentelemetry()
    eval_run_id = str(uuid.uuid4())
    set_log_context(eval_run_id=eval_run_id, user_id="agent-eval-framework")
    log.info(f"Starting evaluation run: {eval_run_id}", extra={"config_path": config_path})

    # Project root is /Users/laah/Code/walmart/agent-eval
    project_root = pathlib.Path(__file__).resolve().parent.parent.parent.parent
    # Load .env from project root
    dotenv_path = project_root / ".env"
    if dotenv_path.exists():
        log.debug(f"Loading environment variables from: {dotenv_path}")
        dotenv.load_dotenv(dotenv_path=dotenv_path, override=True)
    else:
        log.warning(f".env file not found at {dotenv_path}, relying on existing environment variables.")

    project_id = os.getenv("GCP_PROJECT_ID")
    location = os.getenv("GCP_REGION")
    if not project_id or not location:
        # Replaced placeholder check with a simple existence check
        raise EnvironmentError("GCP_PROJECT_ID and GCP_REGION must be set in the .env file or environment.")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    log.debug("Configuration loaded", extra={"config": json.dumps(config)})

    experiment_name = config.get("experiment_name", "default-agent-evals")
    try:
        vertexai.init(project=project_id, location=location)
        aiplatform.init(project=project_id, location=location, experiment=experiment_name)
        log.info(f"Vertex AI initialized for project: {project_id}, location: {location}, experiment: {experiment_name}")
    except Exception as e:
        log.error(f"Failed to initialize Vertex AI: {e}", exc_info=True)
        raise

    adapter_class = load_class(config["agent_adapter_class"])
    adapter = adapter_class(**config.get("agent_config", {}))

    dataset_path = config["dataset_path"]
    local_dataset_path = None
    is_gcs_path = str(dataset_path).startswith("gs://")
    try:
        if is_gcs_path:
            log.info(f"Downloading dataset from GCS: {dataset_path}")
            local_dataset_path = download_gcs_file(dataset_path)
        else:
            # Paths in config are relative to the config file's location
            config_dir = pathlib.Path(config_path).parent
            possible_path = (config_dir.parent / dataset_path).resolve()
            if possible_path.exists():
                local_dataset_path = str(possible_path)
            else:
                 log.error(f"Dataset file not found: {possible_path}")
                 raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        log.info(f"Loading dataset from: {local_dataset_path}")
        with open(local_dataset_path, "r") as f:
            golden_dataset = [json.loads(line) for line in f]
    finally:
        if is_gcs_path and local_dataset_path and os.path.exists(local_dataset_path):
             os.remove(local_dataset_path)
             log.info(f"Removed temporary file: {local_dataset_path}")

    df_dataset = pd.DataFrame(golden_dataset)
    log.info(f"Loaded dataset with {len(df_dataset)} records.")

    column_mapping = config.get("column_mapping", {})
    df_dataset.rename(columns=column_mapping, inplace=True)

    prompt_col = "prompt"
    if prompt_col not in df_dataset.columns:
        raise ValueError(f"'{prompt_col}' column not found in dataset after mapping.")
    reference_col = "reference"
    if reference_col not in df_dataset.columns:
        log.warning(f"'{reference_col}' column not found, adding empty column.")
        df_dataset[reference_col] = ""

    df_eval = pd.DataFrame()
    df_eval["prompt"] = df_dataset[prompt_col]
    df_eval["reference"] = df_dataset[reference_col]
    if 'reference_trajectory' in df_dataset.columns:
         df_eval['reference_trajectory'] = df_dataset['reference_trajectory']

    log.info("Generating agent responses...")
    predictions = []
    actual_trajectories = []
    for index, record in df_eval.iterrows():
        prompt = record.get("prompt")
        if not prompt:
            log.warning(f"No prompt found in record {index}"); predictions.append(""); actual_trajectories.append([]); continue
        try:
            response_data = adapter.get_response(prompt)
            predictions.append(response_data.get("actual_response", "NO_RESPONSE"))
            actual_trajectories.append(response_data.get("predicted_trajectory", []))
        except Exception as e:
            log.error(f"Adapter failed for prompt '{prompt}'", exc_info=True)
            predictions.append("AGENT_EXECUTION_ERROR"); actual_trajectories.append([])
    df_eval["response"] = predictions
    if 'reference_trajectory' in df_eval.columns:
        df_eval["predicted_trajectory"] = actual_trajectories
    log.info("Finished generating agent responses.")

    metrics = _build_metrics(config["metrics"])
    run_name_prefix = config.get("run_name_prefix", "eval")
    run_name = f"{run_name_prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{eval_run_id[:4]}"

    with aiplatform.start_run(run_name) as my_run:
        log.info(f"Started Vertex AI Experiment Run: {run_name} (ID: {my_run.name})")

        params_to_log = {}
        for key, value in config.items():
            if isinstance(value, (list, dict)):
                try: params_to_log[key] = json.dumps(value, sort_keys=True)
                except TypeError: params_to_log[key] = str(value)
            else: params_to_log[key] = value
        params_to_log["eval_run_id"] = eval_run_id
        my_run.log_params(params_to_log)
        log.info("Logged configuration parameters to Vertex AI Experiment run.")

        eval_task = evaluation.EvalTask(dataset=df_eval, metrics=metrics)
        log.info("Running evaluation using vertexai.evaluation.EvalTask...")
        eval_result = eval_task.evaluate()
        log.info("Evaluation complete.")

        summary_metrics = eval_result.summary_metrics
        my_run.log_metrics(summary_metrics)
        log.info(f"Logged Summary Metrics: {summary_metrics}")

        print("\n--- Evaluation Results ---"); print(f"Vertex AI Experiment: {experiment_name}, Run: {my_run.name}"); print("\nSummary Metrics:"); print(summary_metrics)
        print("\nMetrics Table:"); display(eval_result.metrics_table)

    log.info(f"Finished evaluation run {eval_run_id}")
    return eval_result
