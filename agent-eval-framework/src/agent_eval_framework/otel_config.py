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
from opentelemetry._logs import set_logger_provider
import logging
import google.auth
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
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

    # Instrument to add OTel trace context to logs
    LoggingInstrumentor().instrument()

    # Create a handler that will send logs to the OTel LoggerProvider
    handler = LoggingHandler(level=logging.INFO, logger_provider=log_provider)

    # Configure the root logger to use the OTel handler
    logging.getLogger().addHandler(handler)

    # For console logging, ensure it's formatted as JSON
    # This is useful for local development and debugging
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter()
    console_handler.setFormatter(formatter)

    # Configure root logger to use the console handler
    # Avoid adding duplicate handlers if already configured
    if not any(isinstance(h, logging.StreamHandler) for h in logging.getLogger().handlers):
        logging.getLogger().addHandler(console_handler)

    sys.stdout.write("otel_config.py: JSON logging configured.\n")


def setup_opentelemetry():
    """Sets up OpenTelemetry for the application.

    Initializes and configures the TracerProvider and LoggerProvider.
    - Traces can be exported to Google Cloud Trace, a custom OTLP endpoint, or the console.
    - Logs can be exported to a custom OTLP endpoint (e.g., OpenObserve).
    Configuration is driven by environment variables.
    """
    # General service resource configuration
    resource = Resource.create({
        SERVICE_NAME: "agent-eval-framework"
    })

    # Set up and register the logger provider
    log_provider = LoggerProvider(resource=resource)
    set_logger_provider(log_provider)
    setup_logging(log_provider)

    # Set up and register the tracer provider
    trace_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(trace_provider)
    setup_tracing(trace_provider)

    sys.stdout.flush()

def setup_tracing(trace_provider: TracerProvider):
    """Configures the OTLP Trace Exporter based on environment variables."""
    # OTEL_EXPORTER_OTLP_TRACES_ENDPOINT is the standard env var for OTLP exporters.
    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    use_gcp_trace = os.environ.get("USE_GCP_TRACE", "false").lower() == "true"

    if otlp_endpoint:
        sys.stdout.write(f"otel_config.py: Configuring OTLPSpanExporter for endpoint: {otlp_endpoint}\n")
        headers = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_HEADERS")
        exporter = OTLPSpanExporter(
            endpoint=otlp_endpoint,
            headers=parse_headers(headers) if headers else None
        )
    elif use_gcp_trace:
        try:
            _, project_id = google.auth.default()
            if not project_id:
                project_id = os.getenv("GOOGLE_CLOUD_PROJECT")

            if project_id:
                sys.stdout.write(f"otel_config.py: Configuring CloudTraceSpanExporter for project: {project_id}\n")
                exporter = CloudTraceSpanExporter(project_id=project_id)
            else:
                sys.stdout.write("otel_config.py: USE_GCP_TRACE is true but GOOGLE_CLOUD_PROJECT not set. Using ConsoleSpanExporter.\n")
                exporter = ConsoleSpanExporter()
        except Exception as e:
            sys.stdout.write(f"otel_config.py: Error configuring GCP Trace. Using ConsoleSpanExporter. Error: {e}\n")
            exporter = ConsoleSpanExporter()
    else:
        sys.stdout.write("otel_config.py: No OTLP endpoint or GCP Trace configured. Using ConsoleSpanExporter.\n")
        exporter = ConsoleSpanExporter()

    trace_provider.add_span_processor(BatchSpanProcessor(exporter))


def parse_headers(header_string: str) -> dict:
    """Parses a comma-separated string of key=value pairs into a dictionary."""
    headers = {}
    for header in header_string.split(','):
        key, value = header.strip().split('=', 1)
        headers[key] = value
    return headers

def log_otel_status(context: str = ""):
    """Logs the current OpenTelemetry status for debugging purposes.

    This function prints the configured Google Cloud project ID, the type of the
    current tracer provider, and information about any registered span processors.

    Args:
        context: An optional string to identify the context in which the
            status is being logged.
    """
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    provider = trace.get_tracer_provider()
    sys.stdout.write(f"otel_config.py: OTEL STATUS [{context}]: GOOGLE_CLOUD_PROJECT={project_id}\n")
    sys.stdout.write(f"otel_config.py: OTEL STATUS [{context}]: Provider type: {type(provider)}\n")
    if hasattr(provider, 'span_processors'):
        sys.stdout.write(f"otel_config.py: OTEL STATUS [{context}]: Span Processors: {provider.span_processors}\n")
    else:
        sys.stdout.write(f"otel_config.py: OTEL STATUS [{context}]: Provider has no span_processors attribute.\n")
    sys.stdout.flush()
