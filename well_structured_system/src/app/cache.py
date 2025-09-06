from abc import ABC, abstractmethod
from typing import Any, Dict

class Cache(ABC):
    """
    An abstract base class for a generic caching mechanism.
    It defines the contract that all cache implementations must follow.
    Methods are async to support non-blocking I/O with network-based caches.
    """
    @abstractmethod
    async def get(self, key: str) -> Any:
        """
        Retrieves an item from the cache.
        Returns the item if found, otherwise None.
        """
        pass

    @abstractmethod
    async def set(self, key: str, value: Any):
        """
        Saves an item to the cache.
        """
        pass


class InMemoryCache(Cache):
    """
    A simple, concrete implementation of the Cache interface that uses an
    in-memory dictionary for storage.
    """
    def __init__(self):
        self._cache: Dict[str, Any] = {}
        print("Initialized InMemoryCache.")

    async def get(self, key: str) -> Any:
        """
        Retrieves an item from the in-memory dictionary.
        """
        value = self._cache.get(key)
        if value:
            print(f"CACHE HIT for key: '{key}'")
        else:
            print(f"CACHE MISS for key: '{key}'")
        return value

    async def set(self, key: str, value: Any):
        """
        Saves an item to the in-memory dictionary.
        """
        print(f"CACHE SET for key: '{key}'")
        self._cache[key] = value
