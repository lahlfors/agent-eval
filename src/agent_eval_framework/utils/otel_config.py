# src/agent_eval_framework/utils/otel_config.py
import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
# Corrected import path for the GCP Cloud Trace Span Exporter
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from .logging_utils import log

_otel_initialized = False

def setup_opentelemetry():
    global _otel_initialized
    if _otel_initialized:
        return
    project_id = os.getenv("GCP_PROJECT_ID")
    if not project_id:
        log.warning("GCP_PROJECT_ID not set, OTEL Cloud Trace exporter not configured.")
        return
    try:
        if isinstance(trace.get_tracer_provider(), trace.ProxyTracerProvider):
            resource = Resource(attributes={SERVICE_NAME: "agent-eval-framework"})
            provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(provider)

            # Use the corrected exporter
            gcp_exporter = CloudTraceSpanExporter(project_id=project_id)
            provider.add_span_processor(BatchSpanProcessor(gcp_exporter))
            log.info(f"OpenTelemetry CloudTraceSpanExporter initialized for project: {project_id}")
            _otel_initialized = True
        else:
            log.info("OpenTelemetry TracerProvider already configured.")
            _otel_initialized = True
    except Exception as e:
        log.error(f"Failed to setup OpenTelemetry: {e}", exc_info=True)

def get_tracer(name: str):
    return trace.get_tracer(name)
