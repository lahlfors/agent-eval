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

"""Standardized logging module for the GenAI Evaluation Framework.

This module provides a simplified interface for obtaining a logger instance.
The actual log configuration (e.g., formatters, handlers, exporters) is
managed centrally by the `otel_config` module. This ensures that all logs
are structured and processed consistently according to the OpenTelemetry setup.
"""

import logging
import os
import contextvars

# Context variables for tracing information. These can be used for logging
# outside of an active OTel span, but for tracing, OTel's context is preferred.
session_id_var = contextvars.ContextVar('session_id', default=None)
user_id_var = contextvars.ContextVar('user_id', default=None)
eval_run_id_var = contextvars.ContextVar('eval_run_id', default=None)

def get_logger(name: str) -> logging.Logger:
    """Gets a logger instance.

    The logger's configuration is determined by the global logging setup
    in `otel_config.py`. This function simply retrieves the logger by name.
    The root logger is configured by OpenTelemetry to handle structured
    logging and exporting.

    Args:
        name: The name of the logger (typically `__name__`).

    Returns:
        A Logger instance.
    """
    logger = logging.getLogger(name)
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(log_level)
    return logger

def set_log_context(session_id: str = None, user_id: str = None, eval_run_id: str = None):
    """Sets tracing identifiers for the current asynchronous context.

    These values are available for logging purposes. OpenTelemetry's context
    propagation is used for trace correlation.

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
