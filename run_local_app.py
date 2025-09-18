# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A local runner for the Personalized Shopping Agent.

This script provides a simple way to interact with the shopping agent directly
from the command line, without needing to deploy it or run a web server. It
initializes the agent, sets up console-based OpenTelemetry tracing for
debugging, and runs a single interaction with the agent.
"""

import asyncio
import os
from google.adk.apps import App
from personalized_shopping.agent import root_agent

# OpenTelemetry imports for console tracing
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

def setup_console_tracing():
    """Sets up OpenTelemetry to export trace spans to the console.

    This provides a simple, human-readable way to see the execution trace
    of the agent and its tools directly in the terminal.
    """
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(
        SimpleSpanProcessor(ConsoleSpanExporter())
    )
    trace.set_tracer_provider(tracer_provider)
    print("ADK Console tracing enabled.")

async def run_my_app():
    """Initializes and runs a single session with the shopping agent."""
    setup_console_tracing()

    my_app = App(name="my_shopping_app", root_agent=root_agent)

    async with my_app.create_session() as session:
        print("Session created.")
        async for event in session.send_message("Hello, who are you?"):
            if event.content:
                print(f"Agent: {event.content.parts[0].text}")
        # Add more interactions as needed

if __name__ == "__main__":
    asyncio.run(run_my_app())
