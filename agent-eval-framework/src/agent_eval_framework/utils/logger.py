# Copyright 2025 Google LLC
# ... (license headers) ...

import logging
import json
import contextvars
from datetime import datetime
import os
import sys

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
    structured JSON logs to the console.
    """
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout) # Log to stdout
    formatter = JsonFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
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
