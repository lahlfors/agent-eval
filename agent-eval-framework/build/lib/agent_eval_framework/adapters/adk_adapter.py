# Inside ADKAgentAdapter in adapters.py
import asyncio
from google.adk.apps import App # *** IMPORT THE BASE App ***
from personalized_shopping import root_agent
from opentelemetry import trace

class ADKAgentAdapter:
    def __init__(self, **kwargs):
        self.app = App(agent=root_agent, name="eval-shopping-app")
        # Enable ADK's OpenTelemetry instrumentation, which will use
        # the globally configured OTLP exporter from env variables.
        self.app.enable_otel()
        print("ADKAgentAdapter: Base App initialized and OTel enabled.")

    async def get_response_async(self, prompt: str):
         # ADK's tracer, configured to export via OTLP
        tracer = trace.get_tracer("agent-eval-framework.adapter")
        with tracer.start_as_current_span("ADKAgentAdapter.get_response"):
            async with self.app.create_session() as session:
                response_text = ""
                async for event in session.send_message(prompt):
                    if event.content and event.content.parts:
                        response_text += event.content.parts[0].text
                return {"actual_response": response_text}

    def get_response(self, prompt: str):
        return asyncio.run(self.get_response_async(prompt))
