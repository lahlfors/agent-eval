# Agent Evaluation Framework

A generic, configuration-driven framework for evaluating AI agents using the Google Vertex AI Evaluation Service.

## Overview

This framework provides a reusable and extensible pipeline for running evaluations on any AI agent. The core principle is to decouple the generic evaluation logic from the specific implementation of any given agent.

You provide:
1.  An **Agent Adapter** that "wraps" your agent in a standard interface.
2.  A **Configuration File** (`config.yaml`) that tells the framework where to find your adapter, your dataset, and which metrics to run.
3.  A **Golden Dataset** in JSONL format.

The framework handles the rest: loading data, calling your agent via the adapter, running the Vertex AI evaluation, and printing the results.

## Installation

This framework is designed to be installed as a dependency in your agent's project using a package manager like Poetry or Pip.

For local development, you can install it as a path-based dependency. For example, in your project's `pyproject.toml`:

```toml
[tool.poetry.dependencies]
agent-eval-framework = {path = "../path/to/agent-eval-framework", develop = true}
```

## Usage

To evaluate your agent using this framework, follow these steps:

### 1. Create an Agent Adapter

Create a Python class that inherits from `agent_eval_framework.adapters.BaseAgentAdapter` and implements the `get_response` method. This method is responsible for calling your agent and returning its response.

**Example:**
```python
# my_agent_project/my_adapter.py
from agent_eval_framework.adapters import BaseAgentAdapter

class MyCustomAdapter(BaseAgentAdapter):
    def __init__(self, api_key: str):
        # Initialize your agent here, e.g., with an API key
        self.my_agent_client = SomeAgentAPI(api_key=api_key)

    def get_response(self, prompt: str) -> dict[str, any]:
        # Call your agent
        response_text, tool_calls = self.my_agent_client.get_answer(prompt)

        # Return the response in the required format
        return {
            "actual_response": response_text,
            "actual_trajectory": tool_calls
        }
```

### 2. Create a Configuration File

Create a `config.yaml` file to configure the evaluation run.

**Example `config.yaml`:**
```yaml
# The full import path to your custom adapter class.
agent_adapter_class: "my_agent_project.my_adapter.MyCustomAdapter"

# Configuration for your adapter's __init__ method.
agent_config:
  api_key: "your-secret-api-key" # It's recommended to load this from an env var

# Path to the golden dataset (local or GCS).
dataset_path: "path/to/your/golden_dataset.jsonl" # or gs://bucket/path

# List of metrics to run.
metrics:
  - "rouge"
  - "trajectory_exact_match"

# (Optional) Map your dataset's column names to the framework's names.
column_mapping:
  prompt: "question"
  reference_response: "ground_truth_answer"
  reference_trajectory: "expected_tool_calls"
```

### 3. Create a Runner Script

Create a simple Python script to start the evaluation.

**Example `run_my_eval.py`:**
```python
from agent_eval_framework.runner import run_evaluation

if __name__ == "__main__":
    # Ensure GCP environment variables are set (GCP_PROJECT_ID, GCP_REGION)
    run_evaluation(config_path="path/to/your/config.yaml")
```

### 4. Run the Evaluation
```bash
python run_my_eval.py
```

The framework will then execute the full pipeline and print a table with the evaluation results.
