from typing import Any, Set

def exact_match(ground_truth: str, actual_output: str) -> float:
    """
    Calculates exact match between two strings.
    Returns 1.0 if they are identical, 0.0 otherwise.
    """
    if not isinstance(ground_truth, str) or not isinstance(actual_output, str):
        return 0.0
    return 1.0 if ground_truth.strip() == actual_output.strip() else 0.0


def jaccard_similarity(ground_truth: Set[Any], actual_output: Set[Any]) -> float:
    """
    Calculates the Jaccard similarity between two sets.
    Jaccard similarity is the size of the intersection divided by the size
    of the union of the two sets.
    """
    if not isinstance(ground_truth, set) or not isinstance(actual_output, set):
        return 0.0

    intersection_size = len(ground_truth.intersection(actual_output))
    union_size = len(ground_truth.union(actual_output))

    if union_size == 0:
        return 1.0  # Both sets are empty, they are identical

    return intersection_size / union_size


# You can add more metric functions here following the same pattern.
# For example:
# def rouge_l(ground_truth: str, actual_output: str) -> float:
#     # Implementation for ROUGE-L
#     pass

# def bleu_score(ground_truth: str, actual_output: str) -> float:
#     # Implementation for BLEU score
#     pass
