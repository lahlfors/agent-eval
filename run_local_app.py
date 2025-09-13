# Example script to run an ADK App locally with console tracing

import asyncio
import os
from google.adk.apps import App
from personalized_shopping.agent import root_agent # Import your agent

# OpenTelemetry imports for console tracing
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

def setup_console_tracing():
    """Sets up OpenTelemetry to export spans to the console."""
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(
        SimpleSpanProcessor(ConsoleSpanExporter())
    )
    trace.set_tracer_provider(tracer_provider)
    print("ADK Console tracing enabled.")

async def run_my_app():
    setup_console_tracing() # Enable tracing

    my_app = App(name="my_shopping_app", root_agent=root_agent)

    async with my_app.create_session() as session:
        print("Session created.")
        async for event in session.send_message("Hello, who are you?"):
            if event.content:
                print(f"Agent: {event.content.parts[0].text}")
        # Add more interactions as needed

if __name__ == "__main__":
    asyncio.run(run_my_app())
