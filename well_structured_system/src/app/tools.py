import asyncio
import time

async def metadata_tool(query: str) -> str:
    """
    A mock tool that simulates a slow, CPU-bound or I/O-bound operation
    to fetch metadata.
    """
    print(f"  [Tool] Starting metadata_tool for query: '{query}'...")
    # Simulate a 1-second network or database call
    await asyncio.sleep(1)
    result = f"Metadata for '{query}'"
    print(f"  [Tool] Finished metadata_tool.")
    return result


async def report_tool(metadata: str) -> str:
    """
    A mock tool that simulates generating a report, which depends on metadata.
    """
    print(f"  [Tool] Starting report_tool with metadata: '{metadata}'...")
    # Simulate a 1.5-second external API call
    await asyncio.sleep(1.5)
    result = f"Report generated based on '{metadata}'"
    print(f"  [Tool] Finished report_tool.")
    return result


async def independent_tool(tool_id: int) -> str:
    """
    A mock tool that is independent of others, used to demonstrate
    parallel execution.
    """
    start_time = time.time()
    print(f"  [Tool {tool_id}] Starting independent_tool...")
    # Simulate a 1-second operation
    await asyncio.sleep(1)
    end_time = time.time()
    result = f"Independent tool {tool_id} finished in {end_time - start_time:.2f}s"
    print(f"  [Tool {tool_id}] Finished.")
    return result
