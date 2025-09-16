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

        # Make the endpoint configurable via environment variable
        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://otel.googleapis.com:443")
        # Allow insecure connections for local collectors if needed
        insecure_flag = os.getenv("OTEL_EXPORTER_OTLP_INSECURE", "False").lower() in ('true', '1', 't')

        log.info(f"Configuring OTLP exporter with endpoint: {endpoint} (insecure: {insecure_flag})")

        otlp_exporter = OTLPSpanExporter(
            endpoint=endpoint,
            insecure=insecure_flag
        )

        tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        log.info("✅ OpenTelemetry configured successfully.")

    except ImportError:
        log.warning("⚠️ OpenTelemetry libraries not found. Tracing will be disabled.")
    except Exception as e:
        log.error(f"🔥 Failed to configure OpenTelemetry: {e}", exc_info=True)
