# agent-eval-framework/otel_config.py
import os
import sys
import importlib
from typing import Dict, Any

# Standard OpenTelemetry
from opentelemetry import trace # <--- Added missing import
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.sdk.trace import sampling
from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider

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

def log_otel_status(stage: str):
    provider = trace.get_tracer_provider()
    sys.stdout.write(f"OTEL STATUS [{stage}]: Provider type: {type(provider)}\n")
    if isinstance(provider, SDKTracerProvider):
        # We can't reliably print the processors, but we know it's an SDK provider
         sys.stdout.write(f"OTEL STATUS [{stage}]: Is SDKTracerProvider: True\n")
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
            # return # Allow re-configuration if called again
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

        console_exporter = ConsoleSpanExporter()
        tracer_provider.add_span_processor(SimpleSpanProcessor(console_exporter))
        sys.stdout.write("Added ConsoleSpanExporter for local trace visibility.\n")

        try:
            credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/trace.append"])
            request = google.auth.transport.requests.Request()
            auth_metadata_plugin = AuthMetadataPlugin(credentials=credentials, request=request)
            channel_creds = grpc.composite_channel_credentials(
                grpc.ssl_channel_credentials(),
                grpc.metadata_call_credentials(auth_metadata_plugin),
            )
            endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "telemetry.googleapis.com:443")
            sys.stdout.write(f"Configuring OTLP exporter for GCP with endpoint: {endpoint}\n")
            otlp_exporter = OTLPSpanExporter(endpoint=endpoint, credentials=channel_creds)
            tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            sys.stdout.write("Added OTLPSpanExporter for Google Cloud Trace.\n")
        except Exception as e:
            sys.stderr.write(f"🔥 Failed to configure OTLP Exporter for GCP: {e}\n")
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
