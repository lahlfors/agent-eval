# Copyright 2025 Google LLC
# ... (license headers) ...

import pytest
import os
import sys
import pathlib
from collections import defaultdict
from dotenv import load_dotenv
import pandas as pd
from google.cloud.aiplatform import initializer

@pytest.fixture(scope="session", autouse=True)
def setup_env():
    load_dotenv(override=True)

def test_shopping_agent_vertex_eval(mocker):
    # Mock GCP calls
    mocker.patch('vertexai.init')
    mocker.patch('google.cloud.aiplatform.init')
    mock_eval_result = mocker.Mock()
    mock_eval_result.summary_metrics = {"some_metric": 1.0}
    mock_eval_result.metrics_table = pd.DataFrame() # Mock metrics_table as an empty DataFrame

    mock_eval_task_class = mocker.patch('vertexai.preview.evaluation.EvalTask')
    mock_eval_task_instance = mock_eval_task_class.return_value
    mock_eval_task_instance.evaluate.return_value = mock_eval_result
    mocker.patch('google.cloud.aiplatform.start_run')
    mocker.patch('vertexai.preview.evaluation.utils.create_evaluation_service_client')


    # Mock data loading
    from agent_eval_framework.runner import run_evaluation
    config_path = "agent-eval-framework/config/adk_eval_config.yaml"
    assert os.path.exists(config_path), f"Config file not found: {config_path}"
    print(f"Running evaluation with config: {config_path}")
    eval_result = run_evaluation(config_path)
    assert eval_result is not None
    print("Eval result summary:", eval_result.summary_metrics)
