# Copyright 2025 Google LLC
# ... (license headers) ...

import logging
import json
import contextvars
from datetime import datetime
import os
import sys
import urllib.request
import urllib.error

# 1. Context Variables for tracing
session_id_var = contextvars.ContextVar('session_id', default=None)
user_id_var = contextvars.ContextVar('user_id', default=None)
eval_run_id_var = contextvars.ContextVar('eval_run_id', default=None)


class OpenObserveHandler(logging.Handler):
    """
    A logging handler that sends log records to an OpenObserve API endpoint.
    """
    def __init__(self, endpoint_url: str, user: str, password: str):
        super().__init__()
        self.endpoint_url = endpoint_url
        self.user = user
        self.password = password
        # For sending credentials
        password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        password_mgr.add_password(None, self.endpoint_url, self.user, self.password)
        self.opener = urllib.request.build_opener(urllib.request.HTTPBasicAuthHandler(password_mgr))

    def emit(self, record: logging.LogRecord):
        """
        Formats the record and sends it to OpenObserve as a single-item batch.
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

# 1. Context Variables for tracing
session_id_var = contextvars.ContextVar('session_id', default=None)
user_id_var = contextvars.ContextVar('user_id', default=None)
eval_run_id_var = contextvars.ContextVar('eval_run_id', default=None)

class JsonFormatter(logging.Formatter):
    """
    Formats log records as structured JSON objects, automatically including
    context variables and any custom data passed in the `extra` dictionary.
    """
    RESERVED_ATTRS = (
        'args', 'asctime', 'created', 'exc_info', 'exc_text', 'filename',
        'funcName', 'levelname', 'levelno', 'lineno', 'module',
        'msecs', 'message', 'msg', 'name', 'pathname', 'process',
        'processName', 'relativeCreated', 'stack_info', 'thread', 'threadName'
    )

    def format(self, record: logging.LogRecord) -> str:
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
    """
    Acts as the facade. Returns a logger instance configured to output
    structured JSON logs to the console and/or OpenObserve.
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
    """ Sets the session_id, user_id, and eval_run_id for the current logging context. """
    if session_id:
        session_id_var.set(session_id)
    if user_id:
        user_id_var.set(user_id)
    if eval_run_id:
        eval_run_id_var.set(eval_run_id)

def get_log_context() -> dict:
    """ Retrieves the current logging context. """
    return {
        "session_id": session_id_var.get(),
        "user_id": user_id_var.get(),
        "eval_run_id": eval_run_id_var.get(),
    }
