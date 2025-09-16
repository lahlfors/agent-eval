# src/agent_eval_framework/utils/logger.py
import logging
import os
import sys

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

class ContextFilter(logging.Filter):
    _context = {"eval_run_id": None, "user_id": None}
    def filter(self, record):
        record.eval_run_id = ContextFilter._context.get("eval_run_id", "-")
        record.user_id = ContextFilter._context.get("user_id", "-")
        return True
    @staticmethod
    def set_context(eval_run_id=None, user_id=None):
        if eval_run_id:
            ContextFilter._context["eval_run_id"] = eval_run_id
        if user_id:
            ContextFilter._context["user_id"] = user_id

def get_logger(name: str):
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    logger.setLevel(LOG_LEVEL)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(eval_run_id)s] [%(user_id)s] - %(name)s - %(message)s')
    handler.setFormatter(formatter)
    handler.addFilter(ContextFilter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger

def set_log_context(eval_run_id=None, user_id=None):
    ContextFilter.set_context(eval_run_id=eval_run_id, user_id=user_id)
    # Get a logger to log the context change
    log = get_logger("agent_eval_framework.logger")
    log.debug(f"Set log context: eval_run_id={ContextFilter._context['eval_run_id']}, user_id={ContextFilter._context['user_id']}")

# Re-export from logging_utils to maintain compatibility with runner.py
from .logging_utils import log, setup_logging # Keep these for now
