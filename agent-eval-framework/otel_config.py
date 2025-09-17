# agent-eval-framework/otel_config.py
import os
import sys
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.sdk.trace import sampling
from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider

# Import the Google Cloud Trace Exporter
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

# This import is problematic, but I'm creating the file as requested.
# It might need to be adjusted later if it causes an error.
from .utils.logger import get_logger

log = get_logger(__name__)

def log_otel_status(stage: str):
    provider = trace.get_tracer_provider()
    sys.stdout.write(f"OTEL STATUS [{stage}]: Provider type: {type(provider)}\n")
    if isinstance(provider, SDKTracerProvider):
        # Corrected attribute access
        if hasattr(provider, "_span_processors") and provider._span_processors:
             sys.stdout.write(f"OTEL STATUS [{stage}]: Active span processors: {provider._span_processors}\n")
        else:
             sys.stdout.write(f"OTEL STATUS [{stage}]: SDKTracerProvider has no active span processors\n")
    else:
        sys.stdout.write(f"OTEL STATUS [{stage}]: Provider is NOT an SDK TracerProvider!\n")
    sys.stdout.flush()

def setup_opentelemetry():
    sys.stdout.write("Attempting to setup OpenTelemetry...\n")
    try:
        current_provider = trace.get_tracer_provider()
        sys.stdout.write(f"OpenTelemetry: Current provider in setup_opentelemetry: {type(current_provider)}\n")
        if hasattr(current_provider, "shutdown") and not isinstance(current_provider, trace.ProxyTracerProvider):
            sys.stdout.write("OpenTelemetry appears to be already configured with a real provider.\n")
            # We will not re-configure if a real provider is already in place.
            return

        sys.stdout.write("OpenTelemetry: Configuring new provider.\n")

        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            sys.stdout.write("⚠️ GOOGLE_CLOUD_PROJECT not set, OpenTelemetry disabled.\n")
            return

        service_name = os.getenv("OTEL_SERVICE_NAME", "agent-eval-framework")
        resource = Resource.create({
            SERVICE_NAME: service_name,
            "gcp.project_id": project_id,
        })
        tracer_provider = TracerProvider(resource=resource, sampler=sampling.ALWAYS_ON)
        trace.set_tracer_provider(tracer_provider)
        sys.stdout.write(f"OpenTelemetry: Set TracerProvider: {type(trace.get_tracer_provider())}\n")

        # 1. Console Exporter for local debugging
        console_exporter = ConsoleSpanExporter()
        tracer_provider.add_span_processor(SimpleSpanProcessor(console_exporter))
        sys.stdout.write("Added ConsoleSpanExporter for local trace visibility.\n")

        # 2. Google Cloud Trace Exporter
        try:
            gcp_exporter = CloudTraceSpanExporter(project_id=project_id)
            tracer_provider.add_span_processor(BatchSpanProcessor(gcp_exporter))
            sys.stdout.write("Added CloudTraceSpanExporter for Google Cloud Trace.\n")
        except Exception as e:
            sys.stderr.write(f"🔥 Failed to configure CloudTraceSpanExporter: {e}\n")
            sys.stderr.flush()

        sys.stdout.write(f"✅ OpenTelemetry configured successfully for service: {service_name}, project: {project_id}\n")

    except ImportError as e:
        sys.stderr.write(f"⚠️ OpenTelemetry related libraries not found. Tracing will be disabled. Error: {e}\n")
        sys.stderr.flush()
    except Exception as e:
        sys.stderr.write(f"🔥 Failed to configure OpenTelemetry: {e}\n")
        sys.stderr.flush()
    finally:
        sys.stdout.flush()
