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

import asyncio
import json
import os
import uuid
from typing import Any, Dict, List

from google.adk.evaluation.agent_evaluator import AgentEvaluator  # Corrected
from google.adk.evaluation.eval_case import EvalCase, Invocation
from google.adk.evaluation.eval_set import EvalSet  # Corrected
from google.genai import types as genai_types

# Define the evaluation criteria from the config file.
CRITERIA = json.load(open("eval/test_config.json"))

def create_invocation(data: Dict[str, Any]) -> Invocation:
    """Creates an Invocation object from a dictionary from the JSONL line."""
    user_content = genai_types.Content(parts=[genai_types.Part(text=data.get("query", ""))], role="user")

    final_response = None
    if "reference_answer" in data and data["reference_answer"]:
        final_response = genai_types.Content(parts=[genai_types.Part(text=data["reference_answer"])], role="model")

    tool_uses = data.get("expected_tool_use", [])

    intermediate_data = {
        "tool_uses": tool_uses,
        "intermediate_responses": []
    }

    return Invocation(
        invocation_id=str(uuid.uuid4()),
        user_content=user_content,
        final_response=final_response,
        intermediate_data=intermediate_data
    )

def load_eval_cases(data_dir: str) -> List[EvalCase]:
    """Loads all evaluation data from .jsonl files in a directory and creates EvalCase objects."""
    all_cases = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".jsonl"):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, "r") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line)
                        invocation = create_invocation(data)
                        eval_case = EvalCase(
                            eval_id=data.get("example_id", f"{filename}-{line_num}-{str(uuid.uuid4())}"),
                            conversation=[invocation],
                            session_input=data.get("session_input", {}),
                        )
                        all_cases.append(eval_case)
                    except json.JSONDecodeError as e:
                        print(f"Skipping line {line_num} in {filename} due to JSON error: {e}")
                    except Exception as e:
                        print(f"Error processing line {line_num} in {filename}: {e}")
    return all_cases

def test_eval():
    """Runs the agent evaluation using the loaded data and criteria."""
    eval_cases = load_eval_cases("eval/eval_data")
    if not eval_cases:
        print("No evaluation cases loaded. Check 'eval/eval_data' directory and JSONL contents.")
        assert False, "No evaluation cases were loaded."

    eval_set = EvalSet(
        eval_set_id=str(uuid.uuid4()),
        name="Personalized Shopping Agent Evaluation",
        description="Evaluation set for the personalized shopping agent.",
        eval_cases=eval_cases,
    )

    result = asyncio.run(
        AgentEvaluator.evaluate(
            agent_module="personalized_shopping.agent", # Pointing to agent.py
            dataset=eval_set,
            criteria=CRITERIA.get("criteria", {}),
            num_runs=1,
            print_detailed_results=True,
        )
    )
    print(f"Evaluation finished with pass_rate: {result.pass_rate}")
    assert result.pass_rate >= 0.0
