"""
A modular and extensible evaluation pipeline for testing AI systems.
"""

from .adapters import SystemAdapter, LibraryAdapter, ApiAdapter
from .metrics import exact_match, jaccard_similarity
from .pipeline import run_evaluation

__all__ = [
    "SystemAdapter",
    "LibraryAdapter",
    "ApiAdapter",
    "exact_match",
    "jaccard_similarity",
    "run_evaluation",
]
