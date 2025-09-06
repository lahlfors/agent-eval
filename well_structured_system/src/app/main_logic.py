import asyncio
import time
from .cache import Cache
from .tools import metadata_tool, report_tool, independent_tool

async def handle_request(query: str, cache: Cache) -> str:
    """
    Handles a user request by orchestrating sequential tool calls and
    using a cache to improve performance for repeated requests.
    """
    print(f"\nHandling request for query: '{query}'")

    # 1. Check the cache first
    cached_result = await cache.get(query)
    if cached_result:
        return cached_result

    # 2. If not in cache, run the sequential tool pipeline
    print("  No cached result. Executing tool pipeline...")
    metadata = await metadata_tool(query)
    final_result = await report_tool(metadata)

    # 3. Store the result in the cache before returning
    await cache.set(query, final_result)

    return final_result


async def handle_parallel_request():
    """
    Demonstrates running multiple independent tools concurrently.
    """
    print("\nHandling request to run 3 independent tools...")

    # Create tasks to be run concurrently
    tasks = [
        independent_tool(1),
        independent_tool(2),
        independent_tool(3),
    ]

    # Use asyncio.gather to run them in parallel
    start_time = time.time()
    print("  Starting parallel execution with asyncio.gather...")
    results = await asyncio.gather(*tasks)
    end_time = time.time()

    print(f"  Finished parallel execution in {end_time - start_time:.2f}s")
    print("  Results:")
    for res in results:
        print(f"    - {res}")

    return results
