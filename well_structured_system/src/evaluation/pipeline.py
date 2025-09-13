"""Core orchestration logic for the evaluation pipeline.

This module contains the primary `run_evaluation` function that connects all
the different components of the evaluation framework (adapters, datasets, and
metrics) to execute a complete evaluation run.
"""

from typing import Any, Callable, List, Dict
from .adapters import SystemAdapter
import statistics

def run_evaluation(
    adapter: SystemAdapter,
    dataset: List[Dict[str, Any]],
    metrics: List[Callable[[Any, Any], float]]
) -> Dict[str, float]:
    """Runs a full evaluation pipeline against a system under test.

    This function orchestrates the evaluation process by iterating through a
    dataset, using a system adapter to get the output for each input, and then
    computing a set of specified metrics by comparing the system's output to
    the ground truth. Finally, it aggregates and returns the results.

    Args:
        adapter: An initialized instance of a `SystemAdapter` subclass that
                 provides a standard interface to the system being tested.
        dataset: A list of dictionaries, where each dictionary represents a
                 single test case. Each dictionary must contain an 'input' key
                 and a 'ground_truth' key.
        metrics: A list of metric functions to compute. Each function must
                 accept two arguments (ground_truth, actual_output) and return
                 a float score. The function's `__name__` is used as the key
                 in the results dictionary.

    Returns:
        A dictionary containing the aggregated results. The keys are the names
        of the metric functions, and the values are the average scores for
        each metric across the entire dataset.
    """
    results = {metric.__name__: [] for metric in metrics}
    print(f"Starting evaluation with {len(dataset)} examples and {len(metrics)} metrics...")

    for i, record in enumerate(dataset):
        print(f"Processing example {i+1}/{len(dataset)}...")

        input_data = record.get("input")
        ground_truth = record.get("ground_truth")

        if input_data is None or ground_truth is None:
            print(f"  Skipping record {i+1} due to missing 'input' or 'ground_truth'.")
            continue

        # 1. Get the actual output from the system under test via the adapter
        actual_output = adapter.execute(input_data)

        if actual_output is None:
            print(f"  Skipping metrics for record {i+1} because the adapter returned None.")
            continue

        # 2. Compute and store the score for each metric
        for metric in metrics:
            try:
                score = metric(ground_truth, actual_output)
                results[metric.__name__].append(score)
            except Exception as e:
                print(f"  Error computing metric '{metric.__name__}' for record {i+1}: {e}")

    # 3. Aggregate the results by averaging the scores for each metric
    aggregated_results = {}
    for metric_name, scores in results.items():
        if scores:
            aggregated_results[metric_name] = statistics.mean(scores)
        else:
            aggregated_results[metric_name] = 0.0

    print("Evaluation finished.")
    return aggregated_results
