# personalized_shopping/tools/search.py
import os
from ..shared_libraries.web_agent_site.engine.engine import SearchEngine

# Lazily initialize a global SearchEngine instance
_search_engine = None

def get_search_engine():
    global _search_engine
    if _search_engine is None:
        print("Initializing GCS-aware SearchEngine...")
        _search_engine = SearchEngine() # This is fast, index download is deferred
    return _search_engine

def search(query: str):
    """
    Performs a search query using the GCS-backed SearchEngine.
    """
    search_engine = get_search_engine()
    print(f"Performing search for query: {query}")
    try:
        # The SearchEngine.search() method will handle lazy loading/caching from GCS
        results = search_engine.search(query)
        return results
    except Exception as e:
        print(f"Error during search: {e}")
        return f"Error during search: {e}"
