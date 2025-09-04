# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law of aS KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import pytest
import dotenv
from google.cloud import aiplatform
from vertexai import agent_engines


# Function to call the live agent
def call_live_agent(prompt: str) -> str:
    """Calls the deployed agent and returns the response."""
    agent_engine_id = os.getenv("AGENT_ENGINE_ID")
    if not agent_engine_id:
        raise ValueError("AGENT_ENGINE_ID environment variable not set.")

    agent_engine = agent_engines.get(agent_engine_id)
    # Use a static user_id for evaluation purposes
    session = agent_engine.create_session(user_id="evaluation_user")

    response_parts = []
    response = agent_engine.query(
        user_id=session["user_id"], session_id=session["id"], message=prompt
    )
    for part in response.parts:
        if "text" in part:
            response_parts.append(part["text"])

    return "".join(response_parts)

@pytest.fixture(scope="session", autouse=True)
def load_env():
    """Loads environment variables from .env file."""
    dotenv.load_dotenv()

def test_vertex_evaluation():
    """
    Tests the agent using the Vertex AI Evaluation Service.
    """
    # 1. Setup
    project_id = os.getenv("GCP_PROJECT_ID")
    location = os.getenv("GCP_REGION")
    if not project_id or not location:
        pytest.skip("GCP_PROJECT_ID and GCP_REGION environment variables must be set.")

    aiplatform.init(project=project_id, location=location)

    # 2. Load Golden Dataset
    eval_dataset_path = os.path.join(
        os.path.dirname(__file__), "vertex_eval_data", "golden_record.jsonl"
    )
    with open(eval_dataset_path, "r") as f:
        golden_dataset = [json.loads(line) for line in f]

    # 3. Generate Actual Responses
    for record in golden_dataset:
        prompt = record["prompt"]
        # This calls the live agent for each prompt in the dataset
        try:
            record["actual_response"] = call_live_agent(prompt)
        except Exception as e:
            pytest.fail(f"call_live_agent failed for prompt '{prompt}': {e}")


    # 4. Define and Run Evaluation Task
    eval_task = aiplatform.evaluate.EvalTask(
        dataset=golden_dataset,
        metrics=["rouge"],
        response_column="actual_response",
        reference_column="reference_response",
    )

    result = eval_task.evaluate()

    # 5. Print and Assert
    print("Evaluation results:")
    print(result.metrics_table)
    assert result is not None
    assert "rouge" in result.metrics_table.columns
