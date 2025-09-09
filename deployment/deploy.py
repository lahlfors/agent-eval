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

"""
Deploys the Personalized Shopping Agent to Vertex AI Agent Engine.
"""

import glob
import os
import shutil
import vertexai
from dotenv import load_dotenv
from vertexai.preview import reasoning_engines # CORRECTED IMPORT
from vertexai.preview.reasoning_engines import AdkApp

from personalized_shopping.agent import root_agent

# --- Initialization ---
load_dotenv()
cloud_project = os.getenv("GOOGLE_CLOUD_PROJECT")
cloud_location = os.getenv("GOOGLE_CLOUD_LOCATION")
storage_bucket = os.getenv("GOOGLE_CLOUD_STORAGE_BUCKET")

print(f"cloud_project={cloud_project}")
print(f"cloud_location={cloud_location}")
print(f"storage_bucket={storage_bucket}")

vertexai.init(
    project=cloud_project,
    location=cloud_location,
    staging_bucket=f"gs://{storage_bucket}",
)

print("-" * 50)
print("Deploying app begin...")

# --- Pydantic Schema Rebuild Workaround ---
print("Applying Pydantic schema rebuild workaround for memory tool...")
for tool in root_agent.tools:
    if hasattr(tool, "name") and "preload_memory_tool" in tool.name:
        if hasattr(tool, "model_rebuild"):
            tool.model_rebuild()
            print("Schema rebuild applied successfully.")
            break

# --- Robust Package Finding ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, "..")
    dist_dir = os.path.join(project_root, "dist")

    whl_files = glob.glob(os.path.join(dist_dir, "personalized_shopping-*.whl"))
    if not whl_files:
        raise FileNotFoundError(
            "Could not find the .whl package in the '../dist/' directory. "
            "Please run 'poetry build' in the project root first."
        )

    latest_whl = max(whl_files, key=os.path.getmtime)
    # Path relative to the script location
    AGENT_WHL_FILE = os.path.relpath(latest_whl, start=script_dir)
    print(f"Found agent package at: {AGENT_WHL_FILE}")

except Exception as e:
    print(f"Error finding the wheel package: {e}")
    exit(1)

# --- Deployment ---
print("Deploying agent to agent engine...")
app = AdkApp(agent=root_agent)

try:
    remote_app = reasoning_engines.ReasoningEngine.create(
        app,
        display_name="Personalized Shopping Agent",
        description="An agent for personalized shopping experiences.",
        extra_packages=[AGENT_WHL_FILE],
        requirements=[
            "google-cloud-aiplatform[adk,agent-engines]>=1.93.0,<2.0.0",
            # Dependencies from your pyproject.toml
            "google-genai>=1.9.0,<2.0.0",
            # google-adk is part of google-cloud-aiplatform
            "pydantic>=2.10.6,<3.0.0",
            "pyserini>=0.43.0,<0.44.0",
            "cleantext>=1.1.4,<1.2.0",
            "Flask>=3.1.0,<3.2.0",
            "spacy>=3.8.2,<3.9.0",
            "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
            "thefuzz>=0.22.1,<0.23.0",
            "gym==0.23.0",
            "torch>=2.5.1,<2.6.0",
            "torchvision>=0.20.1,<0.21.0",
            "gdown>=5.2.0,<6.0.0",
            "tabulate>=0.9.0,<0.10.0",
            "absl-py>=2.2.1,<2.3.0",
            "pyyaml>=6.0.2,<7.0.0",
        ],
        sys_version="3.12",  # Ensure container uses Python 3.12
    )
    print("Deploying agent to agent engine finished.")
    print("-" * 50)

    # --- Verification ---
    print(f"Agent deployed successfully!\nResource Name: {remote_app.resource_name}")
    print("-" * 50)
    print("Testing deployment with a simple query...")
    try:
        # Using stream_query as it's the standard for AdkApp
        async def test_query():
            async for event in remote_app.async_stream_query(
                user_id="test-user-123", message="Hello!"
            ):
                print(event)
        import asyncio
        asyncio.run(test_query())
        print("\nTest query finished!")
    except Exception as e:
        print(f"\nTest query failed: {e}")

except Exception as e:
    print(f"Error during deployment: {e}")

print("-" * 50)
