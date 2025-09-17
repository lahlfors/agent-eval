# agent-eval-framework/tests/conftest.py
import pytest
import os
import dotenv
import pathlib
import sys
# Import otel_config to make setup_opentelemetry available
from agent_eval_framework import otel_config

def pytest_configure(config):
    """
    Loads environment variables from .env file at the project root
    before any tests are run.
    """
    project_root = pathlib.Path(__file__).resolve().parent.parent
    dotenv_path = project_root / ".env"
    if dotenv_path.exists():
        sys.stdout.write(f"conftest.py: Loading environment variables from: {dotenv_path}\n")
        dotenv.load_dotenv(dotenv_path=dotenv_path, override=True)
    else:
        sys.stdout.write(f"conftest.py: .env file not found at {dotenv_path}\n")
    sys.stdout.flush()

    sys.stdout.write("conftest.py: Setting up OpenTelemetry...\n")
    sys.stdout.flush()
    otel_config.setup_opentelemetry()  # <--- Ensure this is called
    sys.stdout.write("conftest.py: OpenTelemetry setup complete.\n")
    sys.stdout.flush()
    otel_config.log_otel_status("After conftest.py setup")
