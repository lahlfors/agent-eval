"""
A module demonstrating performance patterns like asyncio and caching.
"""

from .cache import Cache, InMemoryCache
from .main_logic import handle_request, handle_parallel_request

__all__ = [
    "Cache",
    "InMemoryCache",
    "handle_request",
    "handle_parallel_request",
]
