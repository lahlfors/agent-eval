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

This script handles the process of deploying the `personalized_shopping` agent
as a live, callable service on Google Cloud Vertex AI.

The script performs the following steps:
1.  Loads necessary configuration (project ID, location, storage bucket) from
    the .env file.
2.  Initializes the Vertex AI SDK.
3.  Dynamically finds the agent's packaged .whl file in the `dist/` directory.
4.  Applies a crucial workaround to rebuild the memory tool's Pydantic schema,
    preventing a cloud build failure.
5.  Calls `agent_engines.create` to deploy the application to the Vertex AI
    Agent Engine service.
6.  Prints the resource name of the deployed agent.
7.  Runs a simple test query against the newly deployed agent to confirm it is
    working correctly.
"""

import glob
import os

import vertexai
from dotenv import load_dotenv
# CORRECT: Import agent_engines from the top-level vertexai library
from vertexai import agent_engines
from vertexai.preview.reasoning_engines import AdkApp

# CORRECT: Import the root_agent directly from the package
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


# --- CRITICAL: Pydantic Schema Rebuild Workaround ---
# This resolves the "400 Build failed" error by forcing a rebuild of the
# memory tool's schema after all types have been fully defined locally.
# This prevents a forward reference issue in the cloud build environment.
print("Applying Pydantic schema rebuild workaround for memory tool...")
for tool in root_agent.tools:
    # The built-in memory tool is a FunctionTool instance
    if "preload_memory_tool" in tool.name:
        tool.model_rebuild()
        print("Schema rebuild applied successfully.")
        break


# --- Robust Package Finding ---
# Dynamically find the .whl file instead of using a hardcoded path.
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dist_dir = os.path.join(script_dir, "..", "dist")
    
    whl_files = glob.glob(os.path.join(dist_dir, "*.whl"))
    if not whl_files:
        raise FileNotFoundError(
            "Could not find the .whl package in the '../dist/' directory. "
            "Please run 'poetry build' in the project root first."
        )
    
    # Make the path relative to the current directory (deployment/)
    AGENT_WHL_FILE = os.path.relpath(whl_files[0], start=script_dir)
    print(f"Found agent package at: {AGENT_WHL_FILE}")

except Exception as e:
    print(f"Error finding the wheel package: {e}")
    exit(1)


# --- Deployment ---

# --- 4. Deploy to Agent Engine ---
print("Deploying agent to agent engine...")
app = AdkApp(agent=root_agent)

# --- CORRECTED CALL ---
remote_app = agent_engines.create(
    app,  # Pass 'app' as the first positional argument
    display_name="Personalized Shopping Agent",
    description="An agent for personalized shopping experiences.",
    extra_packages=[AGENT_WHL_FILE],
    # requirements=["google-cloud-aiplatform[agent_engines,adk]"], # Consider adding if not already in your .whl
)
print("Deploying agent to agent engine finished.")
print("-" * 50)


# --- Verification ---

print("Testing deployment with a simple query...")
try:
    response = remote_app.query(input="Hello!")
    print("\nTest query successful! Response:")
    print(response)
except Exception as e:
    print(f"\nTest query failed: {e}")

print("Testing deployment finished!")
print("-" * 50)

