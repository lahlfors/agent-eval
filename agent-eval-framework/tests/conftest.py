# agent-eval-framework/tests/conftest.py
import pytest
import os
import dotenv
import pathlib

def pytest_configure(config):
    """
    Loads environment variables from .env file at the project root
    before any tests are run.
    """
    project_root = pathlib.Path(__file__).resolve().parent.parent.parent
    dotenv_path = project_root / ".env"
    if dotenv_path.exists():
        print(f"conftest.py: Loading environment variables from: {dotenv_path}")
        dotenv.load_dotenv(dotenv_path=dotenv_path, override=True)
        # Optional: Verify they are loaded
        # print(f"[DEBUG] GOOGLE_CLOUD_PROJECT in conftest: {os.getenv('GOOGLE_CLOUD_PROJECT')}")
        # print(f"[DEBUG] GOOGLE_CLOUD_LOCATION in conftest: {os.getenv('GOOGLE_CLOUD_LOCATION')}")
    else:
        print(f"conftest.py: .env file not found at {dotenv_path}")
