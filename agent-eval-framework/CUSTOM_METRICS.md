# How to Define and Use a Custom Evaluation Metric

This document explains how to create your own custom evaluation metric, configure it in your `config.yaml`, and run it using the `agent-eval-framework`.

Custom metrics allow you to assess specific aspects of your agent's performance that are not covered by standard metrics like BLEU or ROUGE.

The process involves two main steps:
1.  **Defining the Metric**: Writing a Python function that takes the agent's output and the reference data and returns a score.
2.  **Configuring the Metric**: Updating your `config.yaml` to tell the framework where to find your custom metric function.

---

### **Step 1: Define Your Custom Metric Function**

A custom metric is a Python function that accepts four arguments and returns a dictionary containing the calculated score.

1.  **Create a Python file** for your custom metrics (e.g., `my_custom_metrics.py`).
2.  **Define your function**: The function signature must accept `prompt`, `reference`, `actual_response`, and `actual_trajectory`.

    ```python
    def my_custom_metric(prompt: str, reference: str, actual_response: str, actual_trajectory: list) -> dict:
        """
        Calculates a custom score based on the agent's output.

        Args:
            prompt (str): The input prompt given to the agent.
            reference (str): The expected or "golden" response.
            actual_response (str): The final text response from the agent.
            actual_trajectory (list): The sequence of tool calls made by the agent.

        Returns:
            A dictionary with a "score" key (float or int) and an optional "reason" key (str).
        """
        score = 0.0
        reason = "Initial score"

        # Example: Check if the response contains a specific keyword
        if "expected_keyword" in actual_response.lower():
            score = 1.0
            reason = "The response contained the expected keyword."
        else:
            score = 0.0
            reason = "The response did not contain the expected keyword."

        return {"score": score, "reason": reason}
    ```

**Function Requirements:**
*   It **must** accept `prompt`, `reference`, `actual_response`, and `actual_trajectory` as arguments.
*   It **must** return a dictionary containing at least a `"score"` key with a numeric value.
*   It can optionally return a `"reason"` key with a string explaining the score.

---

### **Step 2: Configure the Custom Metric in Your YAML File**

After defining your function, you need to tell the evaluation framework how to find and use it.

1.  **Open your `config.yaml` file.**
2.  **Add a new entry** to the `metrics` list. This entry must be a dictionary with three keys:
    *   `name`: A unique name for your metric (e.g., `keyword_match`).
    *   `type`: Must be set to `custom_function`.
    *   `custom_function_path`: The fully qualified Python path to your function (e.g., `path.to.your.file.my_custom_metric`).

**Example `config.yaml`:**
```yaml
# ... (other configurations)

metrics:
  - 'exact_match'
  - 'bleu'
  - name: 'keyword_match'
    type: 'custom_function'
    custom_function_path: 'my_project.my_custom_metrics.my_custom_metric'
  - name: 'trajectory_exact_match'
    type: 'custom_function'
    custom_function_path: 'agent_eval_framework.metrics.trajectory_metrics.trajectory_exact_match'

# ... (other configurations)
```

In this example:
*   The framework will look for a function named `my_custom_metric` inside a file located at `my_project/my_custom_metrics.py`.
*   The results for this metric will be displayed under the name `keyword_match`.

---

---

### **Built-in Tokenomics & Cost Analysis Metrics**

The framework includes pre-built custom metrics for tokenomics and cost modeling:

1. **`cost_savings_multiplier`** (`agent_eval_framework.metrics.tokenomics_metrics.cost_savings_multiplier`):
   - Computes the cost multiplier advantage of candidate model (e.g. Gemini 3.7 Flash) over the baseline model (e.g. Claude 3.7 Sonnet).

2. **`token_cost_usd`** (`agent_eval_framework.metrics.tokenomics_metrics.token_cost_usd`):
   - Calculates the exact USD cost per agent execution.

3. **`multi_turn_token_growth_rate`** (`agent_eval_framework.metrics.tokenomics_metrics.multi_turn_token_growth_rate`):
   - Measures token accumulation efficiency and checks for payload bloat across multi-turn tool trajectories.

```yaml
metrics:
  - name: 'cost_savings_multiplier'
    type: 'custom_function'
    custom_function_path: 'agent_eval_framework.metrics.tokenomics_metrics.cost_savings_multiplier'
  - name: 'token_cost_usd'
    type: 'custom_function'
    custom_function_path: 'agent_eval_framework.metrics.tokenomics_metrics.token_cost_usd'
```