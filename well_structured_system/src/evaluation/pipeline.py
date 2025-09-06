from typing import Any, Callable, List, Dict
from .adapters import SystemAdapter
import statistics

def run_evaluation(
    adapter: SystemAdapter,
    dataset: List[Dict[str, Any]],
    metrics: List[Callable[[Any, Any], float]]
) -> Dict[str, float]:
    """
    Runs a full evaluation pipeline.

    Args:
        adapter: An instance of a SystemAdapter for the system under test.
        dataset: A list of dictionaries, where each dict must contain 'input'
                 and 'ground_truth' keys.
        metrics: A list of metric functions to compute. Each function should
                 accept (ground_truth, actual_output) and return a float score.

    Returns:
        A dictionary containing the aggregated results for each metric,
        where the key is the metric function's name and the value is the
        average score across the dataset.
    """
    # Initialize a dictionary to store the scores for each metric
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

    # 3. Aggregate the results (e.g., by averaging)
    aggregated_results = {}
    for metric_name, scores in results.items():
        if scores:
            aggregated_results[metric_name] = statistics.mean(scores)
        else:
            aggregated_results[metric_name] = 0.0 # Or float('nan'), or None

    print("Evaluation finished.")
    return aggregated_results
