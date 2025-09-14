import logging
import json
import contextvars
from datetime import datetime

# 1. Context Variables for tracing
session_id_var = contextvars.ContextVar('session_id', default=None)
user_id_var = contextvars.ContextVar('user_id', default=None)


class JsonFormatter(logging.Formatter):
    """
    Formats log records as structured JSON objects, automatically including
    context variables and any custom data passed in the `extra` dictionary.
    """
    # Define the standard attributes of a LogRecord
    RESERVED_ATTRS = (
        'args', 'asctime', 'created', 'exc_info', 'exc_text', 'filename',
        'funcName', 'levelname', 'levelno', 'lineno', 'module',
        'msecs', 'message', 'msg', 'name', 'pathname', 'process',
        'processName', 'relativeCreated', 'stack_info', 'thread', 'threadName'
    )

    def format(self, record: logging.LogRecord) -> str:
        # Start with the basic message
        log_object = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "logger_name": record.name,
        }

        # Add context variables for tracing
        log_object['session_id'] = session_id_var.get()
        log_object['user_id'] = user_id_var.get()

        # Add any extra data passed to the logging call
        # Iterate over the record's dict and add any keys that are not
        # part of the standard LogRecord attributes.
        for key, value in record.__dict__.items():
            if key not in self.RESERVED_ATTRS and not key.startswith('_'):
                log_object[key] = value

        # Add exception info if present
        if record.exc_info:
            log_object['exception'] = self.formatException(record.exc_info)

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

    handler = logging.StreamHandler()
    formatter = JsonFormatter()
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger

# --- Context Management Helpers ---

def set_log_context(session_id: str, user_id: str):
    """
    Sets the session_id and user_id for the current logging context.
    """
    session_id_var.set(session_id)
    user_id_var.set(user_id)

def get_log_context() -> dict:
    """
    Retrieves the current logging context.
    """
    return {
        "session_id": session_id_var.get(),
        "user_id": user_id_var.get(),
    }
