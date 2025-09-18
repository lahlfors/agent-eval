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

"""A structured, context-aware JSON logging module.

This module provides a logging setup that formats log records as JSON objects.
It uses context variables to automatically enrich log entries with tracing
identifiers like session ID, user ID, and evaluation run ID. It also supports
forwarding logs to an OpenObserve endpoint if configured via environment variables.
"""

import logging
import json
import contextvars
from datetime import datetime
import os
import sys
import urllib.request
import urllib.error

# Context variables for tracing information.
session_id_var = contextvars.ContextVar('session_id', default=None)
user_id_var = contextvars.ContextVar('user_id', default=None)
eval_run_id_var = contextvars.ContextVar('eval_run_id', default=None)


class OpenObserveHandler(logging.Handler):
    """A logging handler that sends log records to an OpenObserve API endpoint.

    This handler formats log records into JSON and sends them to the specified
    OpenObserve HTTP endpoint using basic authentication.

    Attributes:
        endpoint_url: The URL of the OpenObserve _json endpoint.
        user: The username for basic authentication.
        password: The password for basic authentication.
        opener: A urllib.request opener configured with authentication.
    """
    def __init__(self, endpoint_url: str, user: str, password: str):
        """Initializes the OpenObserveHandler.

        Args:
            endpoint_url: The full URL for the OpenObserve _json API.
            user: The username for authentication.
            password: The password for authentication.
        """
        super().__init__()
        self.endpoint_url = endpoint_url
        self.user = user
        self.password = password
        # For sending credentials
        password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        password_mgr.add_password(None, self.endpoint_url, self.user, self.password)
        self.opener = urllib.request.build_opener(urllib.request.HTTPBasicAuthHandler(password_mgr))

    def emit(self, record: logging.LogRecord):
        """Formats the record and sends it to OpenObserve.

        The log record is sent as a single-item list in a JSON array, as
        expected by the OpenObserve _json endpoint.

        Args:
            record: The log record to be emitted.
        """
        try:
            log_entry_json = self.format(record)  # This will be a JSON string from JsonFormatter

            # The _json endpoint expects a JSON array of records.
            payload = f"[{log_entry_json}]".encode('utf-8')

            req = urllib.request.Request(
                self.endpoint_url,
                data=payload,
                headers={'Content-Type': 'application/json', "User-Agent": "Python-Logging-Handler"}
            )

            # Use the opener to handle basic authentication
            with self.opener.open(req, timeout=5) as response:
                if response.status >= 300:
                    print(f"Error sending log to OpenObserve: {response.status} {response.read()}", file=sys.stderr)

        except Exception as e:
            print(f"Failed to send log to OpenObserve: {e}", file=sys.stderr)


class JsonFormatter(logging.Formatter):
    """Formats log records as structured JSON objects.

    This formatter automatically includes context variables (session_id, user_id,
    eval_run_id) and any custom data passed in the `extra` dictionary of a
    logging call.
    """
    RESERVED_ATTRS = (
        'args', 'asctime', 'created', 'exc_info', 'exc_text', 'filename',
        'funcName', 'levelname', 'levelno', 'lineno', 'module',
        'msecs', 'message', 'msg', 'name', 'pathname', 'process',
        'processName', 'relativeCreated', 'stack_info', 'thread', 'threadName'
    )

    def format(self, record: logging.LogRecord) -> str:
        """Converts a log record to a JSON string.

        Args:
            record: The LogRecord instance.

        Returns:
            A JSON string representing the log record.
        """
        log_object = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "logger_name": record.name,
        }

        # Add context variables for tracing
        log_object['session_id'] = session_id_var.get()
        log_object['user_id'] = user_id_var.get()
        log_object['eval_run_id'] = eval_run_id_var.get()

        # Add any extra data passed to the logging call
        extra_data = {}
        for key, value in record.__dict__.items():
            if key not in self.RESERVED_ATTRS and not key.startswith('_'):
                extra_data[key] = value
        if extra_data:
            log_object['extra'] = extra_data

        if record.exc_info:
            log_object['exception'] = self.formatException(record.exc_info)
        if record.stack_info:
            log_object['stack_info'] = self.formatStack(record.stack_info)

        return json.dumps(log_object)

def get_logger(name: str) -> logging.Logger:
    """Gets a logger configured for structured JSON logging.

    This function acts as a facade for logger instantiation. It ensures that
    each logger is a singleton and is configured with a `JsonFormatter`. It
    also adds a console handler and, if configured via environment variables,
    an `OpenObserveHandler`.

    Args:
        name: The name of the logger (typically `__name__`).

    Returns:
        A configured Logger instance.
    """
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    formatter = JsonFormatter()

    # Always add a console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Add OpenObserve handler if endpoint is configured
    oo_endpoint = os.getenv("OPENOBSERVE_ENDPOINT")
    oo_user = os.getenv("OPENOBSERVE_USER")
    oo_password = os.getenv("OPENOBSERVE_PASSWORD")

    if oo_endpoint and oo_user and oo_password:
        try:
            oo_handler = OpenObserveHandler(endpoint_url=oo_endpoint, user=oo_user, password=oo_password)
            oo_handler.setFormatter(formatter)
            logger.addHandler(oo_handler)
            logger.info(f"OpenObserve logging enabled to endpoint: {oo_endpoint}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenObserve handler: {e}", exc_info=True)

    return logger

def set_log_context(session_id: str = None, user_id: str = None, eval_run_id: str = None):
    """Sets tracing identifiers for the current asynchronous context.

    These values will be automatically included in all subsequent log messages
    emitted within the same context.

    Args:
        session_id: The session identifier.
        user_id: The user identifier.
        eval_run_id: The evaluation run identifier.
    """
    if session_id:
        session_id_var.set(session_id)
    if user_id:
        user_id_var.set(user_id)
    if eval_run_id:
        eval_run_id_var.set(eval_run_id)

def get_log_context() -> dict:
    """Retrieves the current tracing identifiers from the context.

    Returns:
        A dictionary containing the current session_id, user_id, and
        eval_run_id.
    """
    return {
        "session_id": session_id_var.get(),
        "user_id": user_id_var.get(),
        "eval_run_id": eval_run_id_var.get(),
    }
