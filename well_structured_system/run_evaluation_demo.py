"""Demonstrates the local evaluation pipeline from `well_structured_system`.

This script serves as an executable demonstration of the evaluation framework
defined within the `well_structured_system/src/evaluation` directory. It shows
how to evaluate a simple Python function (`sample_tool`) by performing the
following steps:

1.  **Define a System**: A sample function to be tested is defined.
2.  **Define a Dataset**: A list of dictionaries is created, with each entry
    containing the input for the function and the expected ground truth output.
3.  **Adapt the System**: The function is wrapped in a `LibraryAdapter` to make
    it compatible with the evaluation pipeline.
4.  **Select Metrics**: A list of metric functions (e.g., `exact_match`) is
    chosen for the evaluation.
5.  **Run Evaluation**: The `run_evaluation` function is called with the
    adapter, dataset, and metrics.
6.  **Print Results**: The script prints the final, aggregated results.

Two separate demos are run to showcase both the `exact_match` and
`jaccard_similarity` metrics.
"""

import sys
import os

# Add the 'src' directory to the Python path to allow direct imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from evaluation import LibraryAdapter, run_evaluation, exact_match, jaccard_similarity

# 1. Define the system/tool to be evaluated.
def sample_tool(text: str) -> str:
    """A simple sample tool that converts text to uppercase."""
    return text.upper()

# 2. Define the evaluation datasets.
dataset_strings = [
    {"input": "hello world", "ground_truth": "HELLO WORLD"},
    {"input": "python", "ground_truth": "PYTHON"},
    {"input": "test", "ground_truth": "TESTING"}, # Should fail
    {"input": "data", "ground_truth": "DATA"},
]

dataset_sets = [
    {"input": {"a", "b", "c"}, "ground_truth": {"a", "b", "c"}}, # Jaccard = 1.0
    {"input": {"a", "b"}, "ground_truth": {"b", "c"}},       # Jaccard = 1/3
    {"input": {"x", "y"}, "ground_truth": {"z"}},             # Jaccard = 0.0
]

def set_converter_tool(s: set) -> set:
    """A dummy tool that just returns the input set, for testing set metrics."""
    return s

def main():
    """Main function to run the evaluation demonstrations."""
    print("--- Running Evaluation Demo 1: Exact Match ---")

    # 3. Adapt the tool to the pipeline's interface.
    adapter = LibraryAdapter(tool_function=sample_tool)
    # 4. Select the metrics to run.
    metrics_to_run = [exact_match]
    # 5. Run the evaluation.
    results = run_evaluation(adapter=adapter, dataset=dataset_strings, metrics=metrics_to_run)
    # 6. Print the results.
    print("\n--- Aggregated Results ---")
    print(results)
    print("--------------------------------------------\n")


    print("--- Running Evaluation Demo 2: Jaccard Similarity ---")
    set_adapter = LibraryAdapter(tool_function=set_converter_tool)
    set_metrics = [jaccard_similarity]
    set_results = run_evaluation(adapter=set_adapter, dataset=dataset_sets, metrics=set_metrics)
    print("\n--- Aggregated Results ---")
    print(set_results)
    print("--------------------------------------------\n")


if __name__ == "__main__":
    main()
