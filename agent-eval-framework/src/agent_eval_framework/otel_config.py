import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from .utils.logger import get_logger

log = get_logger(__name__)

def setup_opentelemetry():
    """Initializes OpenTelemetry to export traces to Google Cloud."""
    if hasattr(trace.get_tracer_provider(), "shutdown"): # Check if a real provider is configured
        log.info("OpenTelemetry appears to be already configured.")
        return

    try:
        resource = Resource(attributes={
            "service.name": os.getenv("OTEL_SERVICE_NAME", "agent-eval-framework")
        })

        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)

        otlp_exporter = OTLPSpanExporter(
            endpoint="https://otel.googleapis.com:443",  # Google Cloud Trace OTLP endpoint
            insecure=False
        )

        tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        log.info("✅ OpenTelemetry configured successfully for Google Cloud Trace.")

    except ImportError:
        log.warning("⚠️ OpenTelemetry libraries not found. Tracing will be disabled.")
    except Exception as e:
        log.error(f"🔥 Failed to configure OpenTelemetry: {e}", exc_info=True)
