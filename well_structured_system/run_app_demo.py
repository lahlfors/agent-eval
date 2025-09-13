"""Demonstrates the functionality of the `well_structured_system` application.

This script serves as an executable demonstration of the core features of the
`well_structured_system`, showcasing:
1.  **Caching**: It runs the same request twice to show the performance gain
    from caching. The first call is slow (a cache miss), and the second is
    nearly instantaneous (a cache hit).
2.  **Concurrency**: It runs multiple independent tools in parallel using
    `asyncio.gather` to demonstrate efficient handling of concurrent operations.
"""

import sys
import os
import asyncio
import time

# Add the 'src' directory to the Python path to allow direct imports
# of modules like 'app'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from app import InMemoryCache, handle_request, handle_parallel_request

async def main():
    """Main asynchronous function to run the application demonstrations."""
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
    print("\n--- DEMO 2: Concurrent Execution with asyncio.gather ---")
    await handle_parallel_request()


if __name__ == "__main__":
    # Entry point to run the main async function
    asyncio.run(main())
