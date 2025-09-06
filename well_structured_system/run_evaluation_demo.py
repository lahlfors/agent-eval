# Add src to the Python path to allow direct imports
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from evaluation import LibraryAdapter, run_evaluation, exact_match, jaccard_similarity

# --- DEMONSTRATION ---

# 1. Define the system/tool to be evaluated.
# This is a simple function that converts a string to uppercase.
def sample_tool(text: str) -> str:
    """A simple tool that converts text to uppercase."""
    return text.upper()

# 2. Define the evaluation dataset.
# Each record has an 'input' for the tool and a 'ground_truth' for the expected output.
dataset = [
    {"input": "hello world", "ground_truth": "HELLO WORLD"},  # Should be an exact match
    {"input": "python", "ground_truth": "PYTHON"},            # Should be an exact match
    {"input": "test", "ground_truth": "TESTING"},             # Should not match
    {"input": "data", "ground_truth": "DATA"},                # Should be an exact match
]

# A second dataset for the Jaccard similarity metric
dataset_sets = [
    {"input": {"a", "b", "c"}, "ground_truth": {"a", "b", "c"}}, # Jaccard = 1.0
    {"input": {"a", "b"}, "ground_truth": {"b", "c"}},       # Jaccard = 1/3
    {"input": {"x", "y"}, "ground_truth": {"z"}},             # Jaccard = 0.0
]

def set_converter_tool(s: set) -> set:
    """A dummy tool that just returns the input set."""
    return s

def main():
    """
    Main function to demonstrate the evaluation pipeline.
    """
    print("--- Running Evaluation Demo 1: Exact Match ---")

    # 3. Adapt the tool to the pipeline's interface.
    adapter = LibraryAdapter(tool_function=sample_tool)

    # 4. Select the metrics to run.
    metrics_to_run = [exact_match]

    # 5. Run the evaluation.
    results = run_evaluation(adapter=adapter, dataset=dataset, metrics=metrics_to_run)

    # 6. Print the results.
    print("\\n--- Aggregated Results ---")
    print(results)
    # Expected output for exact_match: 0.75 (3 out of 4 were correct)
    print("--------------------------------------------\\n")


    print("--- Running Evaluation Demo 2: Jaccard Similarity ---")
    set_adapter = LibraryAdapter(tool_function=set_converter_tool)
    set_metrics = [jaccard_similarity]
    set_results = run_evaluation(adapter=set_adapter, dataset=dataset_sets, metrics=set_metrics)
    print("\\n--- Aggregated Results ---")
    print(set_results)
    # Expected output for jaccard_similarity: mean(1.0, 0.333, 0.0) = 0.444
    print("--------------------------------------------\\n")


if __name__ == "__main__":
    main()
