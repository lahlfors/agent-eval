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

"""Configures OpenTelemetry for exporting traces to Google Cloud Trace."""

import os
import sys
import google.auth
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

def setup_opentelemetry():
    """Sets up OpenTelemetry for the application to export to Google Cloud Trace.

    This function initializes the OpenTelemetry tracer provider and configures it
    to use the CloudTraceSpanExporter. It automatically detects the Google Cloud
    project ID from the environment. If the project ID cannot be determined,
    tracing will be disabled.
    """
    try:
        credentials, project_id = google.auth.default()
        if not project_id: # Sometimes project_id is not in credentials
            project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    except Exception as e:
        sys.stdout.write(f"otel_config.py: WARNING: Error getting Google Cloud credentials: {e}\n")
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")

    if not project_id:
        sys.stdout.write("otel_config.py: CRITICAL: GOOGLE_CLOUD_PROJECT not set. Tracing to GCP will be disabled.\n")
        sys.stdout.flush()
        return

    sys.stdout.write(f"otel_config.py: Setting up OpenTelemetry for project: {project_id}\n")

    # Set up resource - identifies the service producing traces
    resource = Resource.create({
        SERVICE_NAME: "agent-eval-framework",
        "gcp.project_id": project_id
    })

    # Set up trace provider
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)

    # Configure Cloud Trace Exporter
    try:
        cloud_trace_exporter = CloudTraceSpanExporter(project_id=project_id)
        # Register the exporter with the TraceProvider using a BatchSpanProcessor
        trace.get_tracer_provider().add_span_processor(
            BatchSpanProcessor(cloud_trace_exporter)
        )
        sys.stdout.write(f"otel_config.py: CloudTraceSpanExporter configured for project {project_id}.\n")
    except Exception as e:
        sys.stdout.write(f"otel_config.py: CRITICAL: Failed to configure CloudTraceSpanExporter: {e}\n")

    sys.stdout.flush()

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
