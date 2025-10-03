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

"""Configures OpenTelemetry for exporting traces and logs."""

import os
import sys
import logging
import google.auth
import google.auth.transport.requests
from opentelemetry import trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

# OTel Logging
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from pythonjsonlogger import jsonlogger

def setup_logging(log_provider: LoggerProvider):
    """Configures the OTLP Log Exporter and JSON formatting."""
    openobserve_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT")
    if openobserve_endpoint:
        headers = os.environ.get("OTEL_EXPORTER_OTLP_LOGS_HEADERS")
        log_exporter = OTLPLogExporter(
            endpoint=f"{openobserve_endpoint}",
            headers=parse_headers(headers) if headers else None
        )
        log_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
        sys.stdout.write(f"otel_config.py: OTLPLogExporter configured for endpoint: {openobserve_endpoint}\n")
    else:
        sys.stdout.write("otel_config.py: OTEL_EXPORTER_OTLP_LOGS_ENDPOINT not set. OTLP log exporting will be disabled.\n")

    LoggingInstrumentor().instrument()
    handler = LoggingHandler(level=logging.INFO, logger_provider=log_provider)
    logging.getLogger().addHandler(handler)
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter()
    console_handler.setFormatter(formatter)
    if not any(isinstance(h, logging.StreamHandler) for h in logging.getLogger().handlers):
        logging.getLogger().addHandler(console_handler)
    sys.stdout.write("otel_config.py: JSON logging configured.\n")

from openinference.instrumentation.vertexai import VertexAIInstrumentor

def setup_opentelemetry():
    """Sets up OpenTelemetry for the application."""
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "laah-genai")
    resource_attributes = {SERVICE_NAME: "agent-eval-framework"}
    if project_id:
        resource_attributes["gcp.project_id"] = project_id

    resource = Resource.create(resource_attributes)

    log_provider = LoggerProvider(resource=resource)
    set_logger_provider(log_provider)
    setup_logging(log_provider)

    trace_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(trace_provider)
    setup_tracing(trace_provider)

    # Instrument for Vertex AI
    VertexAIInstrumentor().instrument()

    sys.stdout.flush()

def setup_tracing(trace_provider: TracerProvider):
    """Configures the OTLP Trace Exporter based on environment variables."""
    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")

    if otlp_endpoint:
        exporter = None
        # If the endpoint is for Google Cloud, we need to configure auth.
        if "googleapis.com" in otlp_endpoint:
            try:
                sys.stdout.write(f"otel_config.py: Google Cloud endpoint detected. Configuring AuthorizedSession.\n")

                quota_project = os.getenv("GOOGLE_CLOUD_PROJECT", "laah-genai")
                credentials, project_id = google.auth.default(
                    quota_project_id=quota_project,
                    scopes=["https://www.googleapis.com/auth/cloud-platform", "https://www.googleapis.com/auth/trace.append"]
                )
                sys.stdout.write(f"otel_config.py: Got credentials for quota project: {quota_project}\n")

                # Use AuthorizedSession to handle auth and project headers
                authed_session = google.auth.transport.requests.AuthorizedSession(credentials)

                sys.stdout.write(f"otel_config.py: Configuring OTLP HTTP exporter for endpoint: {otlp_endpoint}\n")
                exporter = OTLPSpanExporter(
                    endpoint=otlp_endpoint,
                    session=authed_session  # Pass the session here
                )
            except Exception as e:
                sys.stderr.write(f"otel_config.py: Failed to configure Google Cloud OTLP HTTP Exporter. Error: {e}\n")
                # Fallback to console exporter
                sys.stdout.write("otel_config.py: Falling back to ConsoleSpanExporter.\n")
                exporter = ConsoleSpanExporter()
        else:
            # For non-Google endpoints, just use headers from env
            headers = parse_headers(os.environ.get("OTEL_EXPORTER_OTLP_TRACES_HEADERS", ""))
            sys.stdout.write(f"otel_config.py: Configuring OTLP HTTP exporter for non-Google endpoint: {otlp_endpoint}\n")
            exporter = OTLPSpanExporter(
                endpoint=otlp_endpoint,
                headers=headers if headers else None
            )
    else:
        sys.stdout.write("otel_config.py: No OTLP endpoint configured. Using ConsoleSpanExporter.\n")
        exporter = ConsoleSpanExporter()

    if exporter:
        trace_provider.add_span_processor(BatchSpanProcessor(exporter))

def parse_headers(header_string: str) -> dict:
    """Parses a comma-separated string of key=value pairs into a dictionary."""
    if not header_string:
        return {}
    headers = {}
    for header in header_string.split(','):
        key, value = header.strip().split('=', 1)
        headers[key] = value
    return headers

def log_otel_status(context: str = ""):
    """Logs the current OpenTelemetry status for debugging purposes."""
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    provider = trace.get_tracer_provider()
    sys.stdout.write(f"otel_config.py: OTEL STATUS [{context}]: GOOGLE_CLOUD_PROJECT={project_id}\n")
    sys.stdout.write(f"otel_config.py: OTEL STATUS [{context}]: Provider type: {type(provider)}\n")
    if hasattr(provider, 'span_processors'):
        sys.stdout.write(f"otel_config.py: OTEL STATUS [{context}]: Span Processors: {provider.span_processors}\n")
    else:
        sys.stdout.write(f"otel_config.py: OTEL STATUS [{context}]: Provider has no span_processors attribute.\n")
    sys.stdout.flush()