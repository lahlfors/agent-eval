# Evaluating a New Agent

This document provides a step-by-step guide on how to use the `agent-eval-framework` to run a comprehensive evaluation on a new, custom agent.

The overall process involves:
1.  **Creating an Agent Adapter**: A Python class that acts as a "bridge" between the framework and your agent.
2.  **Preparing a Dataset**: A JSONL file containing your test cases (prompts and expected outcomes).
3.  **Creating a Configuration File**: A YAML file that tells the framework how to run the evaluation.
4.  **Running the Evaluation**: Executing a script to start the process and view the results.

---

### **Step 1: Create an Agent Adapter**

The first step is to create a Python class that allows the evaluation framework to communicate with your specific agent.

1.  **Create a new Python file** (e.g., `my_agent_adapter.py`).
2.  **Import the base class**: In your new file, import `BaseAgentAdapter` from the framework:
    ```python
    from agent_eval_framework.adapters import BaseAgentAdapter
    ```
3.  **Define your adapter class**: Create a new class that inherits from `BaseAgentAdapter`.
    ```python
    class MyAgentAdapter(BaseAgentAdapter):
        # ...
    ```
4.  **Implement the `__init__` method**: The constructor should handle any setup your agent needs, such as accepting API keys or initializing a client.
    ```python
    def __init__(self, api_key: str):
        self.api_key = api_key
        # Initialize your agent client here
        # self.my_agent_client = ...
    ```
5.  **Implement the `get_response` method**: This is the core of the adapter. The framework will call this function with a prompt, and it must return a dictionary containing the agent's response and the sequence of tools it used (the trajectory).

    ```python
    def get_response(self, prompt: str) -> dict:
        # 1. Call your agent with the prompt
        # agent_output, tool_calls = self.my_agent_client.query(prompt)
        
        # 2. Format the response into the required dictionary
        return {
            "actual_response": "The final text response from your agent.",
            "actual_trajectory": [
                {"tool_name": "tool1", "tool_input": {"arg": "value"}},
                {"tool_name": "tool2", "tool_input": {"arg": "value"}},
            ]
        }
    ```

---

### **Step 2: Prepare Your Evaluation Dataset**

The evaluation framework requires a dataset of test cases in the JSONL (JSON Lines) format, where each line in the file is a separate, complete JSON object.

1.  **Create a `.jsonl` file** (e.g., `my_eval_dataset.jsonl`).
2.  **Add your test cases**: Each line in the file should be a JSON object that represents one test case. The object must contain a **prompt**, the **expected response** (`reference`), and the **expected tool calls** (`reference_trajectory`).

**Example `my_eval_dataset.jsonl`:**
```json
{"prompt": "What is the capital of France?", "reference": "The capital of France is Paris.", "reference_trajectory": [{"tool_name": "search", "tool_input": {"query": "capital of France"}}]}
{"prompt": "What is 2 + 2?", "reference": "2 + 2 is 4.", "reference_trajectory": []}
```

*   **`prompt`**: The exact input to send to the agent.
*   **`reference`**: The ideal final answer you expect from the agent.
*   **`reference_trajectory`**: A list of dictionaries representing the sequence of expected tool calls. If no tool calls are expected, use an empty list `[]`.

---

### **Step 3: Create a Configuration File**

The `config.yaml` file tells the evaluation framework where to find your agent adapter, your dataset, and which metrics to use.

1.  **Create a `config.yaml` file.**
2.  **Define the evaluation**: Add the following sections to the file:

    ```yaml
    # 1. Fully qualified path to your custom adapter class
    agent_adapter_class: 'path.to.your.file.MyAgentAdapter'

    # 2. Configuration for your agent (passed to your adapter's __init__)
    agent_config:
      api_key: 'YOUR_API_KEY' # Example parameter

    # 3. Path to your dataset (can be local or a gs:// URI)
    dataset_path: 'data/my_eval_dataset.jsonl'

    # 4. (Optional) Mapping for columns if your dataset uses different names
    column_mapping:
      prompt: 'my_prompt_column'
      reference: 'my_reference_column'
      reference_trajectory: 'my_trajectory_column'

    # 5. Define the metrics to run
    metrics:
      - 'exact_match'
      - 'bleu'
      - 'rouge_l'
      - name: 'trajectory_exact_match'
        type: 'custom_function'
        custom_function_path: 'agent_eval_framework.metrics.trajectory_metrics.trajectory_exact_match'

    # 6. (Optional) Name for the Vertex AI Experiment
    experiment_name: 'my-new-agent-evaluation'
    ```

---

### **Step 4: Run the Evaluation**

With the adapter, dataset, and configuration in place, you can now run the evaluation.

1.  **Create a new Python script** (e.g., `run_my_eval.py`).
2.  **Add the following code**:

    ```python
    import os
    from agent_eval_framework.runner import run_evaluation

    # Make sure your .env file has GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION
    if __name__ == "__main__":
        config_path = "path/to/your/config.yaml"
        
        print(f"Starting evaluation with config: {config_path}")
        eval_result = run_evaluation(config_path=config_path)
        
        if eval_result:
            print("\\n--- Evaluation Complete ---")
            print(eval_result.summary_metrics)
    ```
3.  **Run the script from your terminal**:
    ```bash
    python run_my_eval.py
    ```

The framework will then execute the full pipeline and print a table of the results, which will also be saved to Vertex AI Experiments.
