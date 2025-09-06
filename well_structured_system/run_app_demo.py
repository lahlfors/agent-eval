# Add src to the Python path to allow direct imports
import sys
import os
import asyncio
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from app import InMemoryCache, handle_request, handle_parallel_request

async def main():
    """
    Main async function to demonstrate the app's performance patterns.
    """
    # --- Caching Demonstration ---
    print("--- DEMO 1: Caching for Sequential Tool Calls ---")
    cache = InMemoryCache()
    query = "daily sales report"

    # First call: should be slow and result in a cache miss
    start_time_1 = time.time()
    result_1 = await handle_request(query, cache)
    end_time_1 = time.time()
    print(f"Result 1: '{result_1}'")
    print(f"Time for first call (cache miss): {end_time_1 - start_time_1:.2f}s")

    # Second call: should be very fast and result in a cache hit
    start_time_2 = time.time()
    result_2 = await handle_request(query, cache)
    end_time_2 = time.time()
    print(f"Result 2: '{result_2}'")
    print(f"Time for second call (cache hit): {end_time_2 - start_time_2:.2f}s")

    # --- Concurrency Demonstration ---
    print("\\n--- DEMO 2: Concurrent Execution with asyncio.gather ---")
    await handle_parallel_request()


if __name__ == "__main__":
    asyncio.run(main())
