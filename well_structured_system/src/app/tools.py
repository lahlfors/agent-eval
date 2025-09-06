"""A collection of mock asynchronous tools for demonstration purposes.

This module defines several `async` functions that simulate the behavior of
tools that might be used in an agent or application. These tools use
`asyncio.sleep` to represent I/O-bound operations (like network requests or
database calls), making them suitable for showcasing concurrent execution.
"""

import asyncio
import time

async def metadata_tool(query: str) -> str:
    """Simulates a tool that fetches metadata for a given query.

    This asynchronous function mimics a network or database call that takes
    1 second to complete. It represents the first step in a sequential chain
    of tool calls.

    Args:
        query: The input query for which to fetch metadata.

    Returns:
        A string containing the fetched metadata.
    """
    print(f"  [Tool] Starting metadata_tool for query: '{query}'...")
    await asyncio.sleep(1)
    result = f"Metadata for '{query}'"
    print(f"  [Tool] Finished metadata_tool.")
    return result

async def report_tool(metadata: str) -> str:
    """Simulates a tool that generates a report based on metadata.

    This asynchronous function depends on the output of `metadata_tool`. It
    mimics a 1.5-second operation, such as calling an external API or
    performing a complex calculation, to generate a final report.

    Args:
        metadata: The metadata string, typically from `metadata_tool`.

    Returns:
        A string representing the final generated report.
    """
    print(f"  [Tool] Starting report_tool with metadata: '{metadata}'...")
    await asyncio.sleep(1.5)
    result = f"Report generated based on '{metadata}'"
    print(f"  [Tool] Finished report_tool.")
    return result

async def independent_tool(tool_id: int) -> str:
    """Simulates a standalone tool to demonstrate parallel execution.

    This asynchronous function represents a task that can be run concurrently
    with other independent tools. It takes 1 second to complete and returns a
    message indicating its completion.

    Args:
        tool_id: A unique identifier for the tool instance.

    Returns:
        A string confirming the tool's completion and its execution time.
    """
    start_time = time.time()
    print(f"  [Tool {tool_id}] Starting independent_tool...")
    await asyncio.sleep(1)
    end_time = time.time()
    result = f"Independent tool {tool_id} finished in {end_time - start_time:.2f}s"
    print(f"  [Tool {tool_id}] Finished.")
    return result
