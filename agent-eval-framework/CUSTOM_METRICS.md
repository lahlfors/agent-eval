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

### **Step 3: Run the Evaluation**

Run your evaluation script as you normally would. The framework will automatically import your custom function, execute it for each test case, and include the results in the final summary.

```bash
python run_my_eval.py
```

The output will now include a new column for your `keyword_match` metric, showing the score for each prompt. The summary metrics will also include an aggregate score (e.g., the average) for your custom metric.