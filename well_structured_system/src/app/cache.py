"""Provides an abstract base class and a concrete implementation for caching.

This module defines a generic caching interface (`Cache`) and a simple
in-memory implementation (`InMemoryCache`). The use of an abstract base class
allows the application to be decoupled from the specific caching technology,
making it easy to swap in a different backend (like Redis) in the future.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

class Cache(ABC):
    """Abstract base class for a generic asynchronous caching mechanism.

    This class defines the essential `get` and `set` methods that any cache
    implementation must provide. By defining a standard interface, the rest of
    the application can use the cache without being tied to a specific
    implementation. The methods are asynchronous to support non-blocking I/O,
    which is crucial for network-based caches (e.g., Redis).
    """
    @abstractmethod
    async def get(self, key: str) -> Any:
        """Retrieves an item from the cache based on its key.

        Args:
            key: The unique identifier for the item in the cache.

        Returns:
            The cached item if found, otherwise None.
        """
        pass

    @abstractmethod
    async def set(self, key: str, value: Any):
        """Saves an item to the cache.

        If an item with the same key already exists, it should be overwritten.

        Args:
            key: The unique identifier for the item to be stored.
            value: The item to store in the cache.
        """
        pass

class InMemoryCache(Cache):
    """A simple, in-memory implementation of the Cache interface.

    This class provides a basic caching mechanism that uses a Python dictionary
    for storage. It is useful for development, testing, or scenarios where a
    shared, persistent cache is not required. All data is lost when the
    application instance is destroyed.

    Attributes:
        _cache: A dictionary holding the cached keys and values.
    """
    def __init__(self):
        """Initializes the InMemoryCache."""
        self._cache: Dict[str, Any] = {}
        print("Initialized InMemoryCache.")

    async def get(self, key: str) -> Any:
        """Retrieves an item from the internal in-memory dictionary.

        Args:
            key: The key of the item to retrieve.

        Returns:
            The cached value if the key exists, otherwise None.
        """
        value = self._cache.get(key)
        if value:
            print(f"CACHE HIT for key: '{key}'")
        else:
            print(f"CACHE MISS for key: '{key}'")
        return value

    async def set(self, key: str, value: Any):
        """Saves an item to the internal in-memory dictionary.

        Args:
            key: The key under which to store the value.
            value: The value to be stored.
        """
        print(f"CACHE SET for key: '{key}'")
        self._cache[key] = value
