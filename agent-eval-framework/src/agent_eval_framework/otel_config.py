# agent-eval-framework/otel_config.py
import os
import importlib
from typing import Dict, Any

# Standard OpenTelemetry
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.sdk.resources import Resource, SERVICE_NAME

# OTLP Exporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Google Cloud Credentials
import google.auth
import google.auth.transport.grpc
import google.auth.transport.requests
import grpc
from google.auth.transport.grpc import AuthMetadataPlugin

# Utility
from .utils.logger import get_logger

log = get_logger(__name__)

def setup_opentelemetry():
    """Initializes OpenTelemetry to export traces to Google Cloud and Console."""
    if hasattr(trace.get_tracer_provider(), "shutdown"):
        current_provider = trace.get_tracer_provider()
        if not isinstance(current_provider, trace.ProxyTracerProvider):
             log.info("OpenTelemetry appears to be already configured.")
             return
        log.info("Replacing existing ProxyTracerProvider.")

    try:
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            log.warning("⚠️ GOOGLE_CLOUD_PROJECT not set, OpenTelemetry tracing to GCP will be disabled.")
            return

        service_name = os.getenv("OTEL_SERVICE_NAME", "agent-eval-framework")

        resource = Resource.create({
            SERVICE_NAME: service_name,
            "gcp.project_id": project_id,
        })

        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)

        # 1. Console Exporter for local debugging
        console_exporter = ConsoleSpanExporter()
        tracer_provider.add_span_processor(SimpleSpanProcessor(console_exporter))
        log.info("Added ConsoleSpanExporter for local trace visibility.")

        # 2. OTLP Exporter for Google Cloud Trace
        try:
            credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/trace.append"])
            request = google.auth.transport.requests.Request()
            auth_metadata_plugin = AuthMetadataPlugin(credentials=credentials, request=request)
            channel_creds = grpc.composite_channel_credentials(
                grpc.ssl_channel_credentials(),
                grpc.metadata_call_credentials(auth_metadata_plugin),
            )

            endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "telemetry.googleapis.com:443")
            log.info(f"Configuring OTLP exporter for GCP with endpoint: {endpoint}")

            otlp_exporter = OTLPSpanExporter(
                endpoint=endpoint,
                credentials=channel_creds
            )
            tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            log.info("Added OTLPSpanExporter for Google Cloud Trace.")

        except Exception as e:
            log.error(f"🔥 Failed to configure OTLP Exporter for GCP: {e}", exc_info=True)

        log.info("✅ OpenTelemetry configured successfully for service: %s, project: %s", service_name, project_id)

    except ImportError as e:
        log.warning("⚠️ OpenTelemetry related libraries not found. Tracing will be disabled. Error: %s", e)
    except Exception as e:
        log.error(f"🔥 Failed to configure OpenTelemetry: {e}", exc_info=True)
